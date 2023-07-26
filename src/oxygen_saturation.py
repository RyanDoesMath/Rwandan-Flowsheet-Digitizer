"""The oxygen_saturation module provides an implementation for
get_values_for_boxes that the physiological_indicators module can
use to get values for the bounding box detections it made."""

from dataclasses import dataclass
from typing import List
from functools import lru_cache
import warnings
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bounding_box import BoundingBox

CHAR_CLASSIFICATION_MODEL = models.regnet_y_400mf()
num_ftrs = CHAR_CLASSIFICATION_MODEL.fc.in_features
CHAR_CLASSIFICATION_MODEL.num_classes = 10
CHAR_CLASSIFICATION_MODEL.fc = nn.Linear(num_ftrs, 10)
CHAR_CLASSIFICATION_MODEL.load_state_dict(
    torch.load("../models/zero_to_nine_char_classifier_regnet_y_400mf.pt")
)
CHAR_CLASSIFICATION_MODEL.eval()


@dataclass
class OxygenSaturation:
    """Dataclass for spo2."""

    chars: list
    boxes: list
    percent: int
    timestamp: int
    implausible: bool = False


def get_values_for_boxes(boxes: List[BoundingBox], image: Image.Image) -> list:
    """Implements a strategy for getting the values for the spo2
    boxes from the physiological indicator section.

    Args :
        boxes - the boxes in that section.
        image - the image that the boxes come from.

    Returns : The actual values for the spo2 section in a list of objects.
    """
    warnings.filterwarnings("ignore")
    observations = cluster_into_observations(boxes)
    observations = predict_values(observations, image)
    observations = impute_naive_value(observations)
    observations = flag_jumps_as_implausible(observations)
    observations = impute_value_for_erroneous_observations(observations)
    warnings.filterwarnings("default")
    return observations


def cluster_into_observations(boxes: List[BoundingBox]) -> List[List[BoundingBox]]:
    """Clusters the individual boxes into groups that are likely to go together."""
    lower_lim = len(boxes) // 3 - int(0.2 * len(boxes) // 3)
    upper_lim = len(boxes) // 2 + int(0.2 * len(boxes) // 2)

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

    return sorted_boxes


def predict_values(
    observations: List[BoundingBox], image: Image.Image
) -> List[OxygenSaturation]:
    """Uses a CNN to classify the individual images.

    Returns : A list of OxygenSaturation objects with no percent or timestamp.
    """
    values = []
    for cluster in observations:
        cluster_chars = []
        cluster_boxes = []
        for bbox in cluster:
            single_char_img = image.crop(bbox.get_box())
            number = classify_image(single_char_img)
            cluster_chars.append(number)
            cluster_boxes.append(bbox)
        obs = OxygenSaturation(
            chars=cluster_chars,
            boxes=cluster_boxes,
            percent=-1,
            timestamp=-1,
            implausible=False,
        )
        values.append(obs)
    return values


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


def impute_naive_value(observations: List[OxygenSaturation]) -> List[OxygenSaturation]:
    """Uses a naive method to impute the value based on the chars.

    This works for most values, but not all. However, this first pass is needed to
    impute the values for erroneous observations in the following steps.

    Returns : A list of OxygenSaturations with percent values.
    """
    lowest_plausible_value = 75
    highest_plausible_value = 100
    for obs in observations:
        naive_value = int("".join([str(x) for x in obs.chars]))
        if lowest_plausible_value <= naive_value <= highest_plausible_value:
            obs.percent = naive_value
        else:
            obs.implausible = True
    return observations


def flag_jumps_as_implausible(
    observations: List[OxygenSaturation],
) -> List[OxygenSaturation]:
    """Flags values that are implausible.

    There are two ways an observation can get flagged.
        1) The value is not in the range 75-100.
        2) The value jumpped from the previous value > 8 percentage points.

    Looking through available data, there has never been a jump greater than 4%.
    So, to 'derate' this threshold, we increase it to 7%. This is enough to
    catch errors made in the tens place (IE: 89 instead of 99).
    """
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

        if (jump_to_next + jump_from_last) / 2 > 7:
            obs.implausible = True

    return observations


def impute_value_for_erroneous_observations(
    observations: List[OxygenSaturation],
) -> List[OxygenSaturation]:
    """Imputes a value to erroneous observations using linear regression.

    This function will use the previous two and next two observations if neither
    are erroneous to create an averaged estimation of what the value should be.
    Then, the function checks all the available one-edit distance values from the
    observation (in this case, a delete, swap, or insert is an edit).
    Finally, the one-edit value that is closest to the regression estimation is
    chosen, or, in the case where there are no one-edit values that are in the
    range (75, 100), the regression estimation itself is rounded and used.
    """

    for index, obs in enumerate(observations):
        if not obs.implausible:
            continue

        try:
            t_minus_one_is_plausible = not observations[index - 1].implausible
            t_minus_1 = observations[index - 1].percent
        except IndexError:
            t_minus_one_is_plausible = False
            t_minus_1 = 0

        try:
            t_minus_two_is_plausible = not observations[index - 2].implausible
            t_minus_2 = observations[index - 2].percent
        except IndexError:
            t_minus_two_is_plausible = False
            t_minus_2 = 0

        try:
            t_plus_one_is_plausible = not observations[index + 1].implausible
            t_plus_1 = observations[index + 1].percent
        except IndexError:
            t_plus_one_is_plausible = False
            t_plus_1 = 0

        try:
            t_plus_two_is_plausible = not observations[index + 2].implausible
            t_plus_2 = observations[index + 2].percent
        except IndexError:
            t_minus_two_is_plausible = False
            t_plus_2 = 0

        forward_estimate = forward_regression(
            t_minus_1, t_minus_2, t_minus_one_is_plausible, t_minus_two_is_plausible
        )
        backward_estimate = backward_regression(
            t_plus_1, t_plus_2, t_plus_one_is_plausible, t_plus_two_is_plausible
        )
        obs.percent = correct_erroneous_observation(
            obs, forward_estimate, backward_estimate
        )

    return observations


def forward_regression(
    t_minus_1: int,
    t_minus_2: int,
    t_minus_1_is_plausible: bool = True,
    t_minus_2_is_plausible: bool = True,
) -> float:
    """Estimates a value for an SpO2 based on the two previous values.

    Args :
        t_minus_1 - the last value.
        t_minus_2 - the value before last.

    Returns: An estimated value based on the previous two values.
    """
    if t_minus_1_is_plausible and t_minus_2_is_plausible:
        beta_1 = 0.5904
        beta_2 = 0.2844
        intercept = 12.2984
        return intercept + t_minus_1 * beta_1 + t_minus_2 * beta_2

    if t_minus_1_is_plausible:
        beta_1 = 0.8331
        intercept = 16.3866
        return intercept + t_minus_1 * beta_1

    if t_minus_2_is_plausible:
        beta_2 = 0.7616
        intercept = 23.42
        return intercept + t_minus_2 * beta_2

    return np.nan


def backward_regression(
    t_plus_one: int,
    t_plus_two: int,
    t_plus_one_is_plausible: bool = True,
    t_plus_two_is_plausible: bool = True,
) -> float:
    """Performs linear regression with the next two values to try to impute the current one.

    Args :
        t_plus_one - The next value.
        t_plus_two - The value after next.

    Returns : An estimated value based on the next two values.
    """
    if t_plus_one_is_plausible and t_plus_two_is_plausible:
        beta_1 = 0.5952
        beta_2 = 0.2686
        intercept = 13.3739
        return intercept + t_plus_one * beta_1 + t_plus_two * beta_2

    if t_plus_one_is_plausible:
        beta_1 = 0.8149
        intercept = 18.1468
        return intercept + t_plus_one * beta_1

    if t_plus_two_is_plausible:
        beta_2 = 0.7536
        intercept = 24.1745
        return intercept + t_plus_two * beta_2

    return np.nan


def correct_erroneous_observation(
    observation: OxygenSaturation, forward_estimate: float, backward_estimate: float
):
    """Corrects a single erroneous observation."""
    observation = remove_until_length_three(observation)
    estimate = np.nanmean([forward_estimate, backward_estimate])
    possible_correct_values = []
    current_prediction = "".join([str(x) for x in observation.chars])
    for val in range(75, 101):
        edit_dist = levenshtein_dist(str(val), current_prediction)
        if edit_dist in [0, 1]:
            possible_correct_values.append(val)

    if np.isnan(estimate):
        return None
    if len(possible_correct_values) == 0:
        return int(round(estimate, 0))
    distance_from_predicted_value = {
        k: abs(k - estimate) for k in possible_correct_values
    }
    return min(distance_from_predicted_value, key=distance_from_predicted_value.get)


def levenshtein_dist(string_1: str, string_2: str) -> int:
    """This function will calculate the levenshtein distance between two input
    strings a and b

    Args :
        string_1 (str) - The first string you want to compare
        string_2 (str) - The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0

    https://towardsdatascience.com/text-similarity-w-levenshtein-distance-in-python-2f7478986e75
    """

    @lru_cache(None)  # for memorization
    def min_dist(str1, str2):

        if str1 == len(string_1) or str2 == len(string_2):
            return len(string_1) - str1 + len(string_2) - str2

        # no change required
        if string_1[str1] == string_2[str2]:
            return min_dist(str1 + 1, str2 + 1)

        return 1 + min(
            min_dist(str1, str2 + 1),  # insert character
            min_dist(str1 + 1, str2),  # delete character
            min_dist(str1 + 1, str2 + 1),  # replace character
        )

    return min_dist(0, 0)


def argmin(target_list: list) -> int:
    """Returns the index of the minimum value of a list."""
    return min(range(len(target_list)), key=lambda x: target_list[x])


def remove_until_length_three(observation: OxygenSaturation) -> OxygenSaturation:
    """Removes boxes from observation until there are three left."""
    confs = [box.confidence for box in observation.boxes]
    if len(observation.boxes) <= 3:
        return observation
    while len(observation.boxes) > 3:
        del_index = argmin(confs)
        del observation.boxes[del_index]
        del confs[del_index]
    return observation
