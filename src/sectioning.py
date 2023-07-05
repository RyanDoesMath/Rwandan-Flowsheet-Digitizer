"""The sectioning module contains methods for identifying and cropping the
individual sections for other modules to be able to use."""

from ultralytics import YOLO
from PIL import Image

SECTIONING_MODEL = YOLO("../models/section_cropper_yolov8.pt")


def extract_sections(image) -> dict:
    """Uses a yolov8 model to crop out the 7 sections of the Rwandan flowsheet.

    Parameters :
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
    pass
