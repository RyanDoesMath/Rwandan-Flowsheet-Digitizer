"""The sectioning module contains methods for identifying and cropping the
individual sections for other modules to be able to use."""

from typing import List
from PIL import Image, ImageDraw
from ultralytics import YOLO
from bounding_box import BoundingBox

SECTIONING_MODEL = YOLO("../models/section_cropper_yolov8.pt")


def extract_sections(image) -> dict:
    """Uses a yolov8 model to crop out the 7 sections of the Rwandan flowsheet.

    Args :
        image - A PIL image of the whole flowsheet post-homography correction.

    Returns :
        A dictionary with the following keys:
        [
            'iv_drugs',
            'inhaled_drugs',
            'iv_fluids',
            'transfusion',
            'blood_pressure_and_heart_rate',
            'physiological_indicators',
            'checkboxes'
        ]
        and PIL image values corresponding to the key section.
    """
    preds = SECTIONING_MODEL(image, verbose=False)
    preds = make_preds_into_boundingboxes(preds)
    preds = filter_section_predictions(preds)

    names = {
        0.0: "iv_drugs",
        1.0: "inhaled_drugs",
        2.0: "iv_fluids",
        3.0: "transfusion",
        4.0: "blood_pressure_and_heart_rate",
        5.0: "physiological_indicators",
        6.0: "checkboxes",
    }

    sections = {}
    for box in preds:
        crop_coords = box.get_box()
        name = names[box.predicted_class]
        sections[name] = image.crop(crop_coords)
    return sections


def make_preds_into_boundingboxes(preds) -> List[BoundingBox]:
    """Creates BoundingBox objects for every prediction.

    Args :
        preds - the yolov8 predictions.

    Returns : A list of BoundingBox objects.
    """
    bboxes = []
    for box in preds[0].boxes.data.tolist():
        bboxes.append(BoundingBox(box[0], box[1], box[2], box[3], box[5], box[4]))
    return bboxes


def filter_section_predictions(preds: List[BoundingBox]) -> List[BoundingBox]:
    """Chooses the prediction with higher confidence if there are duplicates.

    Parameters:
        preds - a list of BoundingBoxes of the detections.

    Returns : a filtered dataframe with a unique row per predicted section.
    """
    max_preds = []
    unique_classes = list(set([p.predicted_class for p in preds]))
    for unique_class in unique_classes:
        boxes_of_class = list(
            filter(lambda box: box.predicted_class == unique_class, preds)
        )
        boxes_of_class = sorted(boxes_of_class, key=lambda x: x.confidence)
        max_preds.append(boxes_of_class[0])

    return max_preds


def show_detections(image: Image.Image) -> Image.Image:
    """Draws rectangles on the image where detections are made."""
    colors = {
        0.0: "#b37486",
        1.0: "#e7a29c",
        2.0: "#7c98a6",
        3.0: "#d67d53",
        4.0: "#5b9877",
        5.0: "#534d6b",
        6.0: "#e6bd57",
    }

    preds = SECTIONING_MODEL(image, verbose=False)
    preds = make_preds_into_boundingboxes(preds)
    preds = filter_section_predictions(preds)

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for box in preds:
        draw.rectangle(box.get_box(), outline=colors[box.predicted_class], width=2)

    return img_copy
