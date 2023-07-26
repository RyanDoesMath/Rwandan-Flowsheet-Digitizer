"""A module for extracting the checkbox data from the checkbox section of a
Rwandan intraoperative Flowsheet."""

from typing import List, Dict
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
import tiles
from bounding_box import BoundingBox

CHECKBOX_TILE_DATA = {"ROWS": 1, "COLUMNS": 8, "STRIDE": 1 / 2}
CHECKBOX_MODEL = YOLO("../models/checkbox_yolov8m.pt")


def extract_checkboxes(image: Image.Image) -> Dict[str:bool]:
    """Reads the checkbox data from an image of the checkbox section.

    Parameters :
        image - a PIL image of the checkbox section.

    Returns : A dictionary of checkboxs names as keys, and  checked or
              not checked as values..
    """
    detections = tiles.tile_predict(
        CHECKBOX_MODEL,
        image,
        rows=CHECKBOX_TILE_DATA["ROWS"],
        columns=CHECKBOX_TILE_DATA["COLUMNS"],
        stride=CHECKBOX_TILE_DATA["STRIDE"],
        overlap_tolerance=0.3,
        remove_non_square=True,
        strategy="iou",
    )
    values = read_checkbox_values(detections)
    return values


def read_checkbox_values(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Creates values from the raw detections.

    Parameters:
        detections - A list of BoundingBox detections.

    Returns: A dictionary with (checkbox name(str):True/False(bool)).
    """
    checkbox_vals = {}

    return checkbox_vals


def read_patient_safety_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the patient safety boxes into a dictionary.

    The patient safety column contains the following boxes:
        Eye Protection
        Warming
        TED Stockings
        Safety Checklist
    """


def read_mask_ventilation_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the mask ventilation boxes into a dictionary.

    The mask ventilation column contains the following boxes:
        Easy Ventilation
        Ventilation w/ Adjunct (oral airway)
        Difficult Ventilation
    """


def read_airway_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the airway boxes into a dictionary.

    The airway column contains the following boxes:
        Natural; Face Mask N*
        LMA N*
        ETT N*
        Trach N*
    """


def read_airway_placement_aid_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the airway placement aid boxes into a dictionary.

    The airway placement aid column contains the following boxes:
        Used
        Not Used
        Fibroscope
        Brochoscope
        Other
    """


def read_lra_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the lra boxes into a dictionary.

    The lra column contains the following boxes:
        Used
        Not Used
    """


def read_tubes_and_lines_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the tubes and lines boxes into a dictionary.

    The tubes and lines column contains the following boxes:
        Peripheral IV Line
        Central IV Line
        Urinary Catheter
        Gastric Tube
    """


def read_monitoring_details_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the monitoring details boxes into a dictionary.

    The monitoring details column contains the following boxes:
        ECG
        NIBP
        SpO2
        EtCO2
        Stethoscope
        Temperature
        NMT
        Urine Output
        Arterial BP
        Other
    """


def read_patient_position_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the patient position boxes into a dictionary.

    The patient position column contains the following boxes:
        Supine
        Prone
        Litholomy
        Sitting
        Trendelenburg
        Fowler
        Lateral
    """


def add_section_column_to_df(dataframe):
    """Adds the checkbox section of the detection to the dataframe.

    Parameters:
        dataframe - the dataframe to add the section column to.

    Returns: The dataframe with a column for the checkbox section.
    """
    sections = [
        ["patient_safety"] * 4,
        ["mask_ventilation"] * 3,
        ["airway"] * 4,
        ["airway_placement_aid"] * 3,
        ["airway_placement_aid_used"] * 1,
        ["airway_placement_aid_not_used"] * 1,
        ["lra_used"] * 1,
        ["lra_not_used"] * 1,
        ["tubes_and_lines"] * 4,
        ["monitoring_details_left"] * 5,
        ["monitoring_details_right"] * 5,
        ["patient_position_left"] * 4,
        ["patient_position_right"] * 3,
    ]
    section_order = [
        [1] * 4,
        [2] * 3,
        [3] * 4,
        [4] * 3,
        [5] * 1,
        [6] * 1,
        [7] * 1,
        [8] * 1,
        [9] * 4,
        [10] * 5,
        [11] * 5,
        [12] * 4,
        [13] * 3,
    ]
    sections = [x for y in sections for x in y]
    section_order = [x for y in section_order for x in y]
    dataframe["section"] = sections
    dataframe["section_order"] = section_order
    return dataframe


def add_box_name_column_to_df(dataframe):
    """Adds the checkbox name of the detection to the dataframe.

    Parameters:
        dataframe - the dataframe to add the name column to.

    Returns: The dataframe with a column specifying the box name.
    """
    names = [
        "eye_protection",
        "warming",
        "ted_stockings",
        "safety_checklist",
        "easy_ventilation",
        "ventilation_with_adjunct",
        "difficult_ventilation",
        "natural_face_mask",
        "lma",
        "ett",
        "trach",
        "fibroscope",
        "brochoscope",
        "apa_other",
        "apa_used",
        "apa_not_used",
        "lra_used",
        "lra_not_used",
        "peripheral_iv_line",
        "central_iv_line",
        "urinary_catheter",
        "gastric_tube",
        "ecg",
        "nibp",
        "spo2",
        "etco2",
        "stethoscope",
        "temperature",
        "nmt",
        "urine_output",
        "arterial_bp",
        "md_other",
        "supine",
        "prone",
        "litholomy",
        "sitting",
        "trendelenburg",
        "fowler",
        "lateral",
    ]
    dataframe = dataframe.sort_values(["section_order", "y1"])
    dataframe["name"] = names
    return dataframe


def show_detections(
    image: Image.Image,
    overlap_tolerance: float = 0.3,
    remove_non_square: bool = True,
    strategy: str = "iou",
) -> Image.Image:
    """Shows an image of the detections.

    Args :
        image - A PIL image of the checkbox section.
        overlap_tolerance - the threshold over which to remove an overlapping box.
        remove_non_square - whether to remove blatently non-sqauare detections.
        strategy - the strategy for overlap detection.


    Returns : A PIL image with colored rectangles showing detections.
    """
    img = image.copy()
    preds = tiles.tile_predict(
        CHECKBOX_MODEL,
        img,
        rows=CHECKBOX_TILE_DATA["ROWS"],
        columns=CHECKBOX_TILE_DATA["COLUMNS"],
        stride=CHECKBOX_TILE_DATA["STRIDE"],
        overlap_tolerance=overlap_tolerance,
        remove_non_square=remove_non_square,
        strategy=strategy,
    )

    draw = ImageDraw.Draw(img)
    red = "#FF6D60"
    green = "#98D8AA"
    colors = {0.0: red, 1.0: green}
    for pred in preds:
        draw.rectangle(pred.get_box(), outline=colors[pred.predicted_class], width=2)
    return img
