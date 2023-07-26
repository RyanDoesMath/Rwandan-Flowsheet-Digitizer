"""A module for extracting the checkbox data from the checkbox section of a
Rwandan intraoperative Flowsheet."""

from typing import List, Dict
from PIL import Image, ImageDraw
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

    At the moment this assumes all boxes have been detected, but this will
    be changed to deal with missing boxes.

    Parameters:
        detections - A list of BoundingBox detections.

    Returns: A dictionary with (checkbox name(str):True/False(bool)).
    """
    total_number_of_boxes = 39
    if len(detections) < total_number_of_boxes:
        raise ValueError("Not all boxes have been detected.")

    detections = sorted(detections, key=lambda bb: bb.confidence)[
        :total_number_of_boxes
    ]
    detections = sorted(detections, key=lambda bb: bb.get_x_center)
    checkbox_sections = [
        read_patient_safety_boxes,
        read_mask_ventilation_boxes,
        read_airway_boxes,
        read_airway_placement_aid_boxes,
        read_lra_boxes,
        read_tubes_and_lines_boxes,
        read_monitoring_details_boxes,
        read_patient_position_boxes,
    ]
    checkbox_vals = {}
    for section_func in checkbox_sections:
        section_dict = section_func(detections)
        checkbox_vals.update(section_dict)

    return checkbox_vals


def read_patient_safety_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the patient safety boxes into a dictionary.

    The patient safety column contains the following boxes:
        Eye Protection
        Warming
        TED Stockings
        Safety Checklist
    """
    section_start = 0
    section_end = 4
    patient_safety_boxes = detections[section_start:section_end]
    patient_safety_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "eye_protection": bool(patient_safety_boxes[0].predicted_class),
        "warming": bool(patient_safety_boxes[1].predicted_class),
        "ted_stockings": bool(patient_safety_boxes[2].predicted_class),
        "safety_checklist": bool(patient_safety_boxes[3].predicted_class),
    }


def read_mask_ventilation_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the mask ventilation boxes into a dictionary.

    The mask ventilation column contains the following boxes:
        Easy Ventilation
        Ventilation w/ Adjunct (oral airway)
        Difficult Ventilation
    """
    section_start = 4
    section_end = 7
    mask_ventilation_boxes = detections[section_start:section_end]
    mask_ventilation_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "easy_ventilation": bool(mask_ventilation_boxes[0].predicted_class),
        "ventilation_with_adjunct": bool(mask_ventilation_boxes[1].predicted_class),
        "difficult_ventilation": bool(mask_ventilation_boxes[2].predicted_class),
    }


def read_airway_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the airway boxes into a dictionary.

    The airway column contains the following boxes:
        Natural; Face Mask N*
        LMA N*
        ETT N*
        Trach N*
    """
    section_start = 7
    section_end = 11
    airway_boxes = detections[section_start:section_end]
    airway_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "natural_face_mask": bool(airway_boxes[0].predicted_class),
        "lma": bool(airway_boxes[1].predicted_class),
        "ett": bool(airway_boxes[2].predicted_class),
        "trach": bool(airway_boxes[3].predicted_class),
    }


def read_airway_placement_aid_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the airway placement aid boxes into a dictionary.

    The airway placement aid column contains the following boxes:
        Used
        Not Used
        Fibroscope
        Brochoscope
        Other
    """
    section_start = 11
    section_end = 16
    airway_placement_boxes = detections[section_start : section_end - 2]
    used_not_used_boxes = detections[section_start + 3 : section_end]
    airway_placement_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "fibroscope": bool(airway_placement_boxes[0].predicted_class),
        "brochoscope": bool(airway_placement_boxes[1].predicted_class),
        "apa_other": bool(airway_placement_boxes[2].predicted_class),
        "apa_used": bool(used_not_used_boxes[0].predicted_class),
        "apa_not_used": bool(used_not_used_boxes[1].predicted_class),
    }


def read_lra_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the lra boxes into a dictionary.

    The lra column contains the following boxes:
        Used
        Not Used
    """
    section_start = 16
    section_end = 18
    lra_boxes = detections[section_start:section_end]

    return {
        "lra_used": bool(lra_boxes[0].predicted_class),
        "lra_not_used": bool(lra_boxes[1].predicted_class),
    }


def read_tubes_and_lines_boxes(detections: List[BoundingBox]) -> Dict[str:bool]:
    """Reads the tubes and lines boxes into a dictionary.

    The tubes and lines column contains the following boxes:
        Peripheral IV Line
        Central IV Line
        Urinary Catheter
        Gastric Tube
    """
    section_start = 18
    section_end = 22
    tubes_and_lines_boxes = detections[section_start:section_end]
    tubes_and_lines_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "peripheral_iv_line": bool(tubes_and_lines_boxes[0].predicted_class),
        "central_iv_line": bool(tubes_and_lines_boxes[1].predicted_class),
        "urinary_catheter": bool(tubes_and_lines_boxes[2].predicted_class),
        "gastric_tube": bool(tubes_and_lines_boxes[3].predicted_class),
    }


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
    section_start = 22
    section_end = 32
    md_left_col_boxes = detections[section_start : section_end - 5]
    md_right_col_boxes = detections[section_start + 5 : section_end]
    md_left_col_boxes.sort(key=lambda bb: bb.get_y_center)
    md_right_col_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "ecg": bool(md_left_col_boxes[0].predicted_class),
        "nibp": bool(md_left_col_boxes[1].predicted_class),
        "spo2": bool(md_left_col_boxes[2].predicted_class),
        "etco2": bool(md_left_col_boxes[3].predicted_class),
        "stethoscope": bool(md_left_col_boxes[4].predicted_class),
        "temperature": bool(md_right_col_boxes[0].predicted_class),
        "nmt": bool(md_right_col_boxes[1].predicted_class),
        "urine_output": bool(md_right_col_boxes[2].predicted_class),
        "arterial_bp": bool(md_right_col_boxes[3].predicted_class),
        "monitoring_details_other": bool(md_right_col_boxes[4].predicted_class),
    }


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
    section_start = 32
    section_end = 39
    pp_left_col_boxes = detections[section_start : section_end - 3]
    pp_right_col_boxes = detections[section_start + 4 : section_end]
    pp_left_col_boxes.sort(key=lambda bb: bb.get_y_center)
    pp_right_col_boxes.sort(key=lambda bb: bb.get_y_center)

    return {
        "supine": bool(pp_left_col_boxes[0].predicted_class),
        "prone": bool(pp_left_col_boxes[1].predicted_class),
        "litholomy": bool(pp_left_col_boxes[2].predicted_class),
        "sitting": bool(pp_left_col_boxes[3].predicted_class),
        "trendelenburg": bool(pp_right_col_boxes[0].predicted_class),
        "fowler": bool(pp_right_col_boxes[1].predicted_class),
        "lateral": bool(pp_right_col_boxes[2].predicted_class),
    }


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
