"""The physiological_indicators module extracts the data from the
physiological indicator section of the Rwandan flowsheet using YOLOv8."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import deshadow
import tiles
from bounding_box import BoundingBox

SINGLE_CHAR_MODEL = YOLO("../models/single_char_pi_detector_yolov8l.pt")
PHYSIOLOGICAL_INDICATOR_TILE_DATA = {"ROWS": 4, "COLUMNS": 17, "STRIDE": 1 / 2}


@dataclass
class OxygenSaturation:
    """Dataclass for spo2."""

    percent: int
    timestamp: int


@dataclass
class EndTidalCarbonDioxide:
    """Dataclass for etco2."""

    percent: int
    timestamp: int


@dataclass
class FractionOfInspiredOxygen:
    """Dataclass for fio2."""

    percent: int
    timestamp: int


@dataclass
class TidalVolume:
    """Dataclass for Tidal Vol x F."""

    tidal_volume_ml: int
    respiratory_rate: int
    timestamp: int


@dataclass
class Temperature:
    """Dataclass for temp in celcius."""

    temp_c: int
    timestamp: int


@dataclass
class Diuresis:
    """Dataclass for diuresis (urine output)."""

    diuresis_ml: int
    timestamp: int


@dataclass
class BloodLoss:
    """Dataclass for estimated blood loss."""

    blood_loss_ml: int
    timestamp: int


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
    )
    predictions = [
        BoundingBox(l, t, r, b, co, cl) for l, t, r, b, cl, co in predictions
    ]
    rows = cluster_into_rows(predictions, img.size[1])
    return rows


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

    y_centers = [bb.y_center for bb in predictions]
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
        y_centers = [bb.y_center for bb in predictions]
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
    y_centers = [bb.y_center for bb in predictions]
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
