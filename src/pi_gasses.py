"""The pi_gasses module provides an implementation for get_values_for_boxes for SpO2, EtCO2, and
FiO2 that the physiological_indicators module can use to get values for the bounding box
detections it made."""

from typing import List
import warnings
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bounding_box import BoundingBox
import oxygen_saturation
import end_tidal_carbon_dioxide
import fraction_of_inspired_oxygen

CHAR_CLASSIFICATION_MODEL = models.regnet_y_400mf()
num_ftrs = CHAR_CLASSIFICATION_MODEL.fc.in_features
CHAR_CLASSIFICATION_MODEL.num_classes = 10
CHAR_CLASSIFICATION_MODEL.fc = nn.Linear(num_ftrs, 10)
CHAR_CLASSIFICATION_MODEL.load_state_dict(
    torch.load("../models/zero_to_nine_char_classifier_regnet_y_400mf.pt")
)
CHAR_CLASSIFICATION_MODEL.eval()


def get_values_for_boxes(
    boxes: List[BoundingBox], image: Image.Image, strategy: str
) -> list:
    """Implements a strategy for getting the values for the gas boxes from the physiological
    indicator section.

    Args :
        boxes - the boxes in that section.
        image - the image that the boxes come from.
        strategy - Determines constants in other functions. One of ["SpO2", "EtCO2", "FiO2"].

    Returns : The actual values for the specific gas section in a list of objects.
    """
    warnings.filterwarnings("ignore")
    strategy = set_strategy(strategy)
    observations = cluster_into_observations(boxes, strategy)
    predicted_chars = predict_values(observations, image)
    observations = create_gas_objects(observations, predicted_chars, strategy)
    observations = impute_naive_value(observations, strategy)
    observations = flag_jumps_as_implausible(observations, strategy)
    observations = impute_value_for_erroneous_observations(observations)
    warnings.filterwarnings("default")
    return observations


def set_strategy(strategy: str):
    """Sets consts and functions that will be used to extract the values from the specific section.

    Args :
        strategy - A string containing the name of the section.

    Returns : The module with the strategy implemented that we need to get values for the boxes.
    """
    strategies = {
        "SpO2": oxygen_saturation,
        "EtCO2": end_tidal_carbon_dioxide,
        "FiO2": fraction_of_inspired_oxygen,
    }
    return strategies[strategy]


def cluster_into_observations(
    boxes: List[BoundingBox], strategy
) -> List[List[BoundingBox]]:
    """Clusters the individual boxes into groups that are likely to go together.

    This function uses KMeans clustering by defining a lower and upper limit to the plausible
    number of clusters (strategies for computing this depend upon the section). Then uses
    the silhouette score to determine the optimal number of observations that need to be made,
    and then returns the clusters with the optimal K value.

    Args :
        boxes - The detected boxes from the YOLOv8 model.
        strategy - The module with the strategy implemented that we need.

    Returns : Boxes clusted into discrete observations.
    """
    lower_lim, upper_lim = strategy.get_limits_for_number_of_clusters(len(boxes))

    x_centers = [box.get_x_center() for box in boxes]
    x_centers = np.array(x_centers).reshape(-1, 1)

    scores = {}
    for val in range(lower_lim, upper_lim + 1):
        kmeans = KMeans(n_init=10, n_clusters=val).fit(x_centers)
        preds = kmeans.fit_predict(x_centers)
        scores[val] = silhouette_score(x_centers, preds)

    best_cluster_value = max(zip(scores.values(), scores.keys()))[1]
    kmeans = KMeans(n_init=10, n_clusters=best_cluster_value).fit(x_centers)
    preds = kmeans.fit_predict(x_centers)
    masks = [[True if x == u else False for x in preds] for u in np.unique(preds)]
    sorted_boxes = [[x for ix, x in enumerate(boxes) if m[ix]] for m in masks]
    sorted_boxes.sort(
        key=lambda cluster: np.mean([bb.get_x_center() for bb in cluster])
    )
    return sorted_boxes


def predict_values(
    observations: List[List[BoundingBox]], image: Image.Image
) -> List[List[int]]:
    """Uses a CNN to classify the individual images.

    Args :
        observations - The clustered BoundingBoxes
        image - The PIL image of the physiological indicators section.

    Returns : A list conntaining the predicted chars for each box.
    """
    chars = []

    for cluster in observations:
        cluster_chars = []
        cluster.sort(key=lambda bb: bb.get_x_center())
        for bbox in cluster:
            single_char_img = image.crop(bbox.get_box())
            number = classify_image(single_char_img)
            cluster_chars.append(number)
        chars.append(cluster_chars)
    return chars


def classify_image(image: Image.Image):
    """Uses a CNN to classify a PIL Image."""
    datatransform = transforms.Compose(
        [
            transforms.Resize(size=(40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_image = datatransform(image)
    pred = CHAR_CLASSIFICATION_MODEL(input_image.unsqueeze(0)).tolist()[0]
    return np.argmax(pred)


def create_gas_objects(
    boxes: List[List[BoundingBox]], chars: List[List[int]], strategy
) -> List:
    """Creates the proper object to store the Boxes alongside the chars that we predicted for them.

    Args :
        boxes - The bounding boxes for the objects.
        chars - The predicted chars for the objects.
        strategy - The module with the strategy implemented that we need.

    Returns : A list of objects that couple the box with the chars, and that can store other data
    like whether or not the objects chars are plausible, and a finalized value.
    """
    objs = []
    for index, cluster_boxes in enumerate(boxes):
        new_obj = strategy.get_dataclass(
            chars=chars[index],
            boxes=cluster_boxes,
            percent=-1,
            timestamp=-1,
            implausible=False,
        )
        objs.append(new_obj)
    return objs


def impute_naive_value(observations: List, strategy) -> List:
    """Uses a naive method to impute the value of a gas object based on the chars.

    This works for most values, but not all. However, this first pass is needed to
    impute the values for erroneous observations in the following steps.

    Returns : A list of gas objects with values.
    """
    lowest_plausible_value = strategy.LOWEST_PLAUSIBLE_VALUE
    highest_plausible_value = strategy.HIGHEST_PLAUSIBLE_VALUE
    for obs in observations:
        naive_value = int("".join([str(x) for x in obs.chars]))
        if lowest_plausible_value <= naive_value <= highest_plausible_value:
            obs.percent = naive_value
        else:
            obs.implausible = True
    return observations


def flag_jumps_as_implausible(observations: List, strategy) -> List:
    """Flags values that are implausible.

    There are two ways an observation can get flagged.
        1) The value is not in the plausible range.
        2) The value jumpped from the previous value > x percentage points, where x is defined by
           the gas.
    """
    jump_threshold = strategy.JUMP_THRESHOLD

    for index, obs in enumerate(observations):
        if index == 0 or index == len(observations) - 1:
            continue
        previous_obs = observations[index - 1]
        next_obs = observations[index + 1]

        jump_to_next = (
            abs(obs.percent - next_obs.percent) if not next_obs.implausible else 0
        )
        jump_from_last = (
            abs(obs.percent - previous_obs.percent)
            if not previous_obs.implausible
            else 0
        )

        if (jump_to_next + jump_from_last) / 2 > jump_threshold:
            obs.implausible = True

    return observations


def impute_value_for_erroneous_observations(observations: List) -> List:
    """Imputes a value to erroneous observations using linear regression.

    Creates a linear regression model with the previous two values and next two values and predicts
    the current value based on the output.
    """

    for index, obs in enumerate(observations):
        if not obs.implausible:
            continue

        surrounding_observations = []
        for surrounding_index in range(index - 2, index + 2):
            try:
                if observations[surrounding_index].implausible:
                    continue
                surrounding_observations.append(
                    (surrounding_index - 2, observations[surrounding_index].percent)
                )
            except IndexError:
                pass

        if len(surrounding_observations) > 0:
            x_values = [[x[0]] for x in surrounding_observations]
            y_values = [[y[1]] for y in surrounding_observations]
            linreg = LinearRegression().fit(x_values, y_values)
            observations[index].percent = int(
                round(linreg.predict([[0]]).tolist()[0][0], 0)
            )

    return observations
