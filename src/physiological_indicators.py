"""The physiological_indicators module extracts the data from the
physiological indicator section of the Rwandan flowsheet using YOLOv8."""

from typing import Dict, List, Tuple
from functools import cache
import warnings
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import deshadow
import tiles
import oxygen_saturation
import end_tidal_carbon_dioxide
import fraction_of_inspired_oxygen
import tidal_volume_x_respiratory_rate
import volume
import respiratory_rate
from bounding_box import BoundingBox


@cache
def load_char_classification_model():
    """Loads the chararacter classification model."""
    char_classification_model = models.regnet_y_1_6gf()
    num_ftrs = char_classification_model.fc.in_features
    char_classification_model.num_classes = 10
    char_classification_model.fc = nn.Linear(num_ftrs, 10)
    char_classification_model.load_state_dict(
        torch.load("../models/zero_to_nine_char_classifier_regnet_1_6_gf.pt")
    )
    char_classification_model.eval()

    return char_classification_model


@cache
def load_x_vs_rest_model():
    """Loads the X vs Rest model."""
    x_vs_rest_classification_model = models.regnet_y_800mf()
    num_ftrs = x_vs_rest_classification_model.fc.in_features
    x_vs_rest_classification_model.num_classes = 2
    x_vs_rest_classification_model.fc = nn.Linear(num_ftrs, 2)
    x_vs_rest_classification_model.load_state_dict(
        torch.load("../models/x_vs_rest_classifier_regnet_y_800mf.pt")
    )
    x_vs_rest_classification_model.eval()

    return x_vs_rest_classification_model


CHAR_CLASSIFICATION_MODEL = load_char_classification_model()
X_VS_REST_MODEL = load_x_vs_rest_model()
SINGLE_CHAR_MODEL = YOLO("../models/single_char_physio_detector_yolov8l.pt")
PHYSIOLOGICAL_INDICATOR_TILE_DATA = {"ROWS": 2, "COLUMNS": 8, "STRIDE": 1 / 2}


def extract_physiological_indicators(image: Image.Image) -> Dict[str, list]:
    """Extracts the physiological indicators from a cropped image
    of the physiological indicators section of a Rwandan Flowsheet.

    Args :
        image - a PIL image of the physiological indicators section.

    Returns : A dictionary with keys for each row of the PI section,
              and a list of timestamped values for that section.
    """
    img = deshadow.deshadow_and_normalize_image(image)
    predictions = tiles.tile_predict(
        SINGLE_CHAR_MODEL,
        img,
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["ROWS"],
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["COLUMNS"],
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["STRIDE"],
        0.5,
        strategy="not_iou",
    )
    rows = cluster_into_rows(predictions, img.size[1])

    indicators = {}
    for name, boxes in rows.items():
        indicators[name] = get_values_for_boxes(name, boxes, image)
    return indicators


def cluster_into_rows(
    predictions: List[BoundingBox], im_height: int
) -> Dict[str, List[BoundingBox]]:
    """Clusters the observations into rows so that different strategies can
    be used to impute and correct the values identified by the CNN.

    Args :
        predictions - the bounding box predictions from the single char model.

    Returns : A dictionary where the name of the section maps to a list of
              BoundingBoxes for that section.
    """
    warnings.filterwarnings("ignore")

    y_centers = [bb.get_y_center() for bb in predictions]
    y_centers = np.array(y_centers).reshape(-1, 1)
    best_cluster_value = find_number_of_rows(predictions, im_height)
    clusters = get_clusters(predictions, best_cluster_value, y_centers)

    rows = {}
    for cluster in clusters:
        label = get_label_for_cluster(cluster, im_height)
        rows[label] = cluster

    warnings.filterwarnings("default")
    return rows


def find_number_of_rows(predictions: List[BoundingBox], im_height: int) -> int:
    """Finds the number of rows that have been filled out on the sheet."""
    has_one_cluster = check_if_section_has_only_one_row(predictions, im_height)
    if has_one_cluster:
        best_cluster_value = 1
    else:
        y_centers = [bb.get_y_center() for bb in predictions]
        scores = compute_silhouette_scores(y_centers)
        best_cluster_value = max(zip(scores.values(), scores.keys()))[1]

    return best_cluster_value


def compute_silhouette_scores(y_centers: List[float]) -> Dict[int, float]:
    """Gets the kmeans silhouette scores for all plausible values of k."""
    y_centers = np.array(y_centers).reshape(-1, 1)
    max_rows = 7
    scores = {}
    for num_rows in range(2, max_rows + 1):
        if num_rows == 0:
            continue
        kmeans = KMeans(n_init=10, n_clusters=num_rows).fit(y_centers)
        preds = kmeans.fit_predict(y_centers)
        scores[num_rows] = silhouette_score(y_centers, preds)

    return scores


def check_if_section_has_only_one_row(
    predictions: List[BoundingBox], im_height: int
) -> bool:
    """Checks if a section has only one row since KMeans needs 2 clusers to run.

    Essentially, 10% is around 60 of the height of a full row. If the largest
    difference between the max and min value is less than 10%, it is very likely
    there is only one row.
    """
    y_centers = [bb.get_y_center() for bb in predictions]
    max_val = max(y_centers)
    min_val = min(y_centers)
    max_allowable_diff = 0.1 * im_height
    return abs(max_val - min_val) < max_allowable_diff


def get_clusters(
    predictions: List[BoundingBox], best_cluster_value: int, y_centers=List[float]
):
    """Returns a list of clusters based on the best cluster value for k."""
    kmeans = KMeans(n_init=10, n_clusters=best_cluster_value).fit(y_centers)
    preds = kmeans.fit_predict(y_centers)
    clusters = []
    for cluster_value in np.unique(preds):
        indices_in_cluster = [ix for ix, x in enumerate(preds) if x == cluster_value]
        clusters.append([predictions[ix] for ix in indices_in_cluster])

    return clusters


def get_label_for_cluster(cluster: List[BoundingBox], im_height: int) -> str:
    """Applies a label to a cluster.

    There are a lot of magic numbers in here. Essentially, the values are the
    plausible y locations where the centroid of the top end of the box can be.

    This was established empirically, and the clusters don't overlap. These
    values are normalized to the image height.
    """
    top_centroid = np.mean([bb.top for bb in cluster]) / im_height
    if top_centroid < 0.137687:
        return "SpO2"
    if 0.137687 < top_centroid <= 0.265647:
        return "EtCO2"
    if 0.265647 < top_centroid <= 0.390272:
        return "FiO2"
    if 0.390272 < top_centroid <= 0.521633:
        return "Tidal_VolxF"
    if 0.521633 < top_centroid <= 0.656327:
        return "Temp_C"
    if 0.656327 < top_centroid <= 0.787619:
        return "Diuresis"
    if 0.787619 < top_centroid:
        return "Blood_Loss"
    return "Unknown"


def get_values_for_boxes(
    section_name: str, boxes: List[BoundingBox], image: Image.Image
) -> list:
    """Imputes the values for the bounding boxes in the section.

    Uses different strategies for each section for classifying the
    characters, clustering characters into observations, flagging
    incorrect values, and imputing values to flagged observations.

    Args :
        section_name - used to select the strategy.
        boxes - the boxes in that section.
        image - the image that the boxes come from.

    Returns : The actual values for that section in a list of objects.
    """
    strategies = {
        "SpO2": get_values_for_gas_boxes,
        "EtCO2": get_values_for_gas_boxes,
        "FiO2": get_values_for_gas_boxes,
        "Tidal_VolxF": get_values_for_tidal_volume,
    }
    try:
        if section_name != "Tidal_VolxF":
            return strategies[section_name](boxes, image, section_name)
        else:
            return strategies[section_name](boxes, image)
    except KeyError:
        print(f"{section_name} has not been implemented yet.")
        return None


def get_values_for_gas_boxes(
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
    strategies = {
        "SpO2": oxygen_saturation,
        "EtCO2": end_tidal_carbon_dioxide,
        "FiO2": fraction_of_inspired_oxygen,
    }
    strategy = strategies[strategy]
    observations = cluster_into_observations(boxes, strategy)
    observations = impute_values_to_clusters(observations, image, strategy)

    warnings.filterwarnings("default")
    return observations


def get_values_for_tidal_volume(
    boxes: List[BoundingBox], image: Image.Image
) -> List[tidal_volume_x_respiratory_rate.TidalVolumeXRespiratoryRate]:
    """Implements a strategy for getting the values for the tidal volume boxes from the
    physiological indicator section.

    Args :
        boxes - the BoundingBoxes in that section.
        image - the image of the physiological indicators section.

    Returns : A list of TidalVolume objects.
    """
    warnings.filterwarnings("ignore")
    strategies = {"tidal_vol": volume, "resp_rate": respiratory_rate}
    observations = cluster_into_observations(boxes, tidal_volume_x_respiratory_rate)
    tidal_vol_obs, resp_rate_obs = separate_tidal_vol_x_f_observations(
        observations, image
    )
    observations = {
        "tidal_vol": tidal_vol_obs,
        "resp_rate": resp_rate_obs,
    }

    for part in ["tidal_vol", "resp_rate"]:
        strategy = strategies[part]
        obs = list(filter(lambda x: x is not None, observations[part]))
        observations[part] = impute_values_to_clusters(obs, image, strategy)
    observations = list(zip(observations["tidal_vol"], observations["resp_rate"]))

    tidal_volume_objects = []
    for tidal_vol, resp_rate in observations:
        tidal_vol_x_resp_rate = (
            tidal_volume_x_respiratory_rate.TidalVolumeXRespiratoryRate(
                volume=tidal_vol, respiratory_rate=resp_rate, timestamp=-1
            )
        )
        tidal_volume_objects.append(tidal_vol_x_resp_rate)
    warnings.filterwarnings("default")
    return tidal_volume_objects


def impute_values_to_clusters(
    observations: List[List[BoundingBox]], image: Image.Image, strategy
):
    """Imputes values to clustered BoundingBoxes given a strategy.

    Args :
        observations - The BoundingBoxes clustered into obeservations.
        image - The PIL Image of the physiological indicators section.
        strategy - The module with data for the particular row's strategy..

    Returns : A list of objects containing the data.
    """
    predicted_chars = predict_values(observations, image)
    observations = create_gas_objects(observations, predicted_chars, strategy)
    observations = impute_naive_value(observations, strategy)
    observations = flag_jumps_as_implausible(observations, strategy)
    observations = impute_value_for_erroneous_observations(observations)

    return observations


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
        if val < 2:
            continue
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


def separate_tidal_vol_x_f_observations(
    observations: List[List[BoundingBox]], image: Image.Image
) -> Tuple[List[List[BoundingBox]], List[List[BoundingBox]]]:
    """Separates the tidal volume numbers that are before the x from the respiratory rate numbers
    that come after the x in the TidalVolxF row of the physiological indicators section.

    Args :
        observations - The bounding box clusters identified by cluster_into_observations.
        image - A PIL image of the physiological indicators section.

    Returns : Two lists of BoundingBoxes separated into (tidal_vol, resp_rate).
    """
    tidal_vol_bboxes = []
    resp_rate_bboxes = []

    for cluster in observations:
        cluster = sorted(cluster, key=lambda box: box.get_x_center())
        bbox_crops = [image.crop(bb.get_box()) for bb in cluster]
        probabilities_that_bbox_is_x = [
            classify_image(crop, model="x_vs_rest") for crop in bbox_crops
        ]
        x_index = np.argmax(probabilities_that_bbox_is_x)
        plausible_x_indices = [2, 3, 4, 5]  # never observed x outside these indices.
        if x_index in plausible_x_indices:
            tidal_vol_bboxes.append(cluster[0:x_index])
            resp_rate_bboxes.append(cluster[x_index + 1 :])
            continue
        if len(cluster) == 6:
            # if it is length 6 (the standard length), its probably VVVxRR.
            tidal_vol_bboxes.append(cluster[0:3])
            resp_rate_bboxes.append(cluster[4:])
            continue
        # if it isn't length 6 (the standard length), we don't know how to split it...
        tidal_vol_bboxes.append(None)
        resp_rate_bboxes.append(None)

    return tidal_vol_bboxes, resp_rate_bboxes


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


def classify_image(image: Image.Image, model: str = "char"):
    """Uses a CNN to classify a PIL Image.

    Args:
        image - A PIL image of the physiological indicators section.
        model - A string determining the model to use ["char", "x_vs_rest"] defaults to "char".
    """
    datatransform = transforms.Compose(
        [
            transforms.Resize(size=(40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_image = datatransform(image)
    if model == "char":
        model = CHAR_CLASSIFICATION_MODEL
    elif model == "x_vs_rest":
        model = X_VS_REST_MODEL
    else:
        raise ValueError(f"Invalid parameter for model:{model}")
    pred = model(input_image.unsqueeze(0)).tolist()[0]
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
        new_obj = strategy.get_dataclass()(
            chars=chars[index],
            boxes=cluster_boxes,
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
            obs.value = naive_value
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

        try:
            jump_to_next = (
                abs(obs.value - next_obs.value) if not next_obs.implausible else 0
            )
        except TypeError:
            jump_to_next = np.nan

        try:
            jump_from_last = (
                abs(obs.value - previous_obs.value)
                if not previous_obs.implausible
                else 0
            )
        except TypeError:
            jump_from_last = np.nan

        average_jump = np.nanmean([jump_from_last, jump_to_next])
        if not np.isnan(average_jump) and average_jump > jump_threshold:
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
                    (surrounding_index - 2, observations[surrounding_index].value)
                )
            except IndexError:
                pass

        if len(surrounding_observations) > 0:
            x_values = [[x[0]] for x in surrounding_observations]
            y_values = [[y[1]] for y in surrounding_observations]
            linreg = LinearRegression().fit(x_values, y_values)
            observations[index].value = int(
                round(linreg.predict([[0]]).tolist()[0][0], 0)
            )

    return observations


def show_detections(image: Image.Image) -> Image.Image:
    """Shows a color coded version of the physiological indicator detections."""
    img = deshadow.deshadow_and_normalize_image(image)
    predictions = tiles.tile_predict(
        SINGLE_CHAR_MODEL,
        img,
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["ROWS"],
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["COLUMNS"],
        PHYSIOLOGICAL_INDICATOR_TILE_DATA["STRIDE"],
        0.5,
        strategy="not_iou",
    )
    rows = cluster_into_rows(predictions, img.size[1])

    section_colors = {
        "SpO2": "#b37486",
        "EtCO2": "#5b9877",
        "FiO2": "#e6bd57",
        "Tidal_VolxF": "#7c98a6",
        "Temp_C": "#e7a29c",
        "Diuresis": "#534d6b",
        "Blood_Loss": "#d67d53",
    }

    draw = ImageDraw.Draw(img)
    for key in list(rows.keys()):
        color = section_colors[key]
        for pred in rows[key]:
            draw.rectangle(pred.get_box(), outline=color, width=2)

    return img
