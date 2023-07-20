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
        BoundingBox(l, t, r, b, cl, co) for l, t, r, b, cl, co in predictions
    ]
    rows = cluster_into_rows(predictions, img.size[1])


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
    y_centers = [bb.y_center for bb in predictions]
    best_cluster_value = find_number_of_rows(predictions, im_height)
    kmeans = KMeans(n_init=10, n_clusters=best_cluster_value).fit(
        np.array(y_centers).reshape(-1, 1)
    )
    preds = kmeans.fit_predict(np.array(y_centers).reshape(-1, 1))


def find_number_of_rows(predictions: List[BoundingBox], im_height: int) -> int:
    """Finds the number of rows that have been filled out on the sheet."""
    has_one_cluster = check_if_section_has_only_one_row(predictions, im_height)
    if has_one_cluster:
        pass
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
    """Checks if a section has only one row since KMeans needs 2 clusers to run."""
