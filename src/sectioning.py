"""The sectioning module contains methods for identifying and cropping the
individual sections for other modules to be able to use."""

from ultralytics import YOLO
from PIL import Image
import pandas as pd

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
    preds = SECTIONING_MODEL(image, verbose=False)
    preds_df = make_predictions_into_dataframe(preds)
    preds_df = filter_section_predictions(preds)
    for index, box in preds_df.iterrows():
        pass


def make_predictions_into_dataframe(preds):
    """Converts YOLOv8 predictions into a dataframe.

    Parameters:
        preds - a YOLOv8 predictions object.

    Returns : the predictions in a pandas dataframe.
    """
    cols = ["left", "top", "right", "bottom", "confidence", "class"]
    names_dict = preds[0].names
    index_of_class_in_box = -1
    preds_df = pd.DataFrame(columns=cols)
    for box in preds[0].boxes.data:
        data = box.tolist()
        data[index_of_class_in_box] = names_dict[data[index_of_class_in_box]]
        new_row = pd.DataFrame(data=[data], columns=cols)
        preds_df = pd.concat([preds_df, new_row])
    preds_df.reset_index(inplace=True, drop=True)
    return preds_df


def filter_section_predictions(preds_df):
    """Chooses the prediction with higher confidence if there are duplicates.

    Parameters:
        preds_df - a dataframe with the section predictions.

    Returns : a filtered dataframe with a unique row per predicted section.
    """
    pass
