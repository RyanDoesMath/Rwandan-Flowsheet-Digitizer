"""The physiological_indicators module extracts the data from the
physiological indicator section of the Rwandan flowsheet using YOLOv8."""

from typing import Dict, List
import warnings
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import deshadow
import tiles
import pi_gasses
from bounding_box import BoundingBox

SINGLE_CHAR_MODEL = YOLO("../models/single_char_physio_detector_yolov8s.pt")
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
        "SpO2": pi_gasses.get_values_for_boxes,
        "EtCO2": pi_gasses.get_values_for_boxes,
    }
    try:
        return strategies[section_name](boxes, image, section_name)
    except KeyError:
        print(f"{section_name} has not been implemented yet.")
        return None


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
