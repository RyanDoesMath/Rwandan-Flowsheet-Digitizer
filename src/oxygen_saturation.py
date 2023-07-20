"""The oxygen_saturation module provides an implementation for
get_values_for_boxes that the physiological_indicators module can
use to get values for the bounding box detections it made."""

from dataclasses import dataclass
from typing import List
import warnings
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bounding_box import BoundingBox
from physiological_indicators import classify_image


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

    x_centers = [box.x_center for box in boxes]
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
            single_char_img = image.crop(bbox.box)
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
    So, to 'derate' this threshold, we double it to 8. This is enough to
    catch errors made in the tens place (IE: 89 instead of 99).
    """
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