"""A module for extracting the checkbox data from the checkbox section of a
Rwandan intraoperative Flowsheet."""

from typing import List
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
import tiles

CHECKBOX_TILE_DATA = {"ROWS": 1, "COLUMNS": 8, "STRIDE": 1 / 2}
BLUE = (35, 45, 75, 100)
ORANGE = (229, 114, 0, 100)
CHECKBOX_MODEL = YOLO("../models/checkbox_yolov8m.pt")


def read_checkbox_data(image):
    """Reads the checkbox data from an image of the checkbox section.

    Parameters :
        image - a PIL image of the checkbox section.

    Returns :
    """
    detections = make_detections(image)
    values = read_checkbox_values(detections)
    return values


def remove_overlapping_detections(
    detections: List[List[float]], tolerance: float = 0.3
):
    """Removes bounding boxes that overlap more than the tolerance level.

    Parameters :
        detections - A list of detection boxes.
        tolerance - the IOU score that triggers a match between boxes.
    """
    remove = []
    for this_box_index, this_box in enumerate(detections):
        for that_box_index, that_box in enumerate(detections[this_box_index + 1 :]):
            iou = intersection_over_union(this_box, that_box)
            if iou > tolerance:
                remove.append(this_box_index + that_box_index + 1)
    remove = sorted(list(set(remove)), reverse=True)
    for index in remove:
        detections.pop(index)
    return detections


def intersection_over_union(box_a: List[float], box_b: List[float]):
    """Computes the bounding box intersection over union.

    Parameters:
        box_a - the first box (order doesn't matter.)
        box_b - the second box (order doesn't matter.)

    Returns : the intersection over union for the two bounding boxes.
    """
    left_side = max(box_a[0], box_b[0])
    top_side = max(box_a[3], box_b[3])
    right_side = min(box_a[2], box_b[2])
    bottom_side = min(box_a[1], box_b[1])
    intersection_area = max(0, right_side - left_side + 1) * max(
        0, bottom_side - top_side + 1
    )
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[1] - box_a[3] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[1] - box_b[3] + 1)
    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
    return iou


def get_x_coords(width: float):
    """Gets the x coordinates for where to crop a tile.

    Parameters:
        width - the width of the image to tile.

    Returns : The x coordinates for all the tiles.
    """
    columns = CHECKBOX_TILE_DATA["COLUMNS"]
    return [int((width * i / columns)) for i in range(0, columns)] + [width]


def get_y_coords(height: float):
    """Gets the y coordinates for where to crop a tile.

    Parameters:
        height - the height of the image to tile.

    Returns : The y coordinates for all the tiles."""
    rows = CHECKBOX_TILE_DATA["ROWS"]
    return [int((height * i / rows)) for i in range(0, rows)] + [height]


def tile_image(image):
    """Segments the image into tiles.

    Parameters:
        image - the image to tile.

    Returns: A 2d array of tiles.
    """
    stride = CHECKBOX_TILE_DATA["STRIDE"]
    tiles = []
    width, height = image.size
    x_coords = get_x_coords(width)
    y_coords = get_y_coords(height)
    for index_x in range(len(x_coords[0 : -int(1 / stride)])):
        row = []
        for index_y in range(len(y_coords[0 : -int(1 / stride)])):
            temp = image.crop(
                (
                    x_coords[index_x],
                    y_coords[index_y],
                    x_coords[index_x + int(1 / stride)],
                    y_coords[index_y + int(1 / stride)],
                )
            )
            row.append(temp)
        tiles.append(row)
    return tiles


def make_detections(image) -> dict:
    """Makes and cleans the detections.

    Returns:
        A dictionary where the keys are 'bp_type' and 'box', indicating
        whether the blood pressure detection is systolic or diastolic,
        and what the normalized x, y, width, height of the box are.
    """
    detections = run_model(image)
    detections = clean_raw_detections(image, detections)

    return detections


def run_model(image):
    """Runs the BP yolo model on all tiles.

    Parameters:
        image - the image to run the model on.

    Returns: The predictions from the YOLOv8 model.
    """
    predictions = []
    for row in tile_image(image):
        new_preds = []
        for tile in row:
            new_preds.append(list(CHECKBOX_MODEL.predict(tile, verbose=False))[0])
        predictions.append(new_preds)
    return predictions


def clean_raw_detections(image, detections):
    """Cleans the raw detections from the model.

    Parameters:
        detections -
    """
    detections = map_raw_detections_to_full_image(image, detections)
    detections = remove_non_square_detections(detections)
    detections = remove_overlapping_detections(detections)
    return detections


def map_raw_detections_to_full_image(image, detections):
    """Maps the coordinates of the raw detections to where they are on the full image."""
    rows = CHECKBOX_TILE_DATA["ROWS"]
    columns = CHECKBOX_TILE_DATA["COLUMNS"]
    mapped_boxes = []
    im_width, im_height = image.size
    for ix, col in enumerate(detections):
        for iy, preds in enumerate(col):
            shift_x = ix / columns
            shift_y = iy / rows
            for ix, detection in enumerate(preds.boxes.xywhn):
                confidence = float(preds.boxes[ix].conf)
                checked = bool(preds.boxes[ix].cls)
                x = (float(detection[0]) / (columns / 2)) + shift_x
                y = (float(detection[1]) / (rows / 2)) + shift_y
                w = float(detection[2]) / (columns / 2)
                h = float(detection[3]) / (rows / 2)
                mapped_boxes.append(
                    (
                        im_width * (x - (w / 2)),
                        im_height * (y + (h / 2)),
                        im_width * (x + (w / 2)),
                        im_height * (y - (h / 2)),
                        checked,
                        confidence,
                    )
                )
    return mapped_boxes


def remove_non_square_detections(detections, threshold: float = 0.25):
    """Removes detections whose percent difference between height and
    width is greater than the threshold.

    Parameters :
        detections - a list of detection boxes.
        threshold - a float to determine how non-square a detection must
                    be to be removed.
    """
    remove = []
    for index, box in enumerate(detections):
        left, right = min(box[0], box[2]), max(box[0], box[2])
        top, bottom = min(box[1], box[3]), max(box[1], box[3])
        width = left - right
        height = top - bottom
        if abs((width - height) / ((height + width) / 2)) > threshold:
            remove.append(index)
    for index in sorted(remove, reverse=True):
        detections.pop(index)
    return detections


def show_detections(image, detections):
    """Shows an image of the detections."""
    new = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(new)
    for _, (_, val) in enumerate(detections.items()):
        detection = val
        bp_type = detection["type"]
        box = detection["box"]
        color = BLUE if bp_type == "systolic" else ORANGE
        draw.rectangle(box, outline="black", width=2, fill=color)
    out = Image.alpha_composite(image.convert("RGBA"), new)
    return out


def read_checkbox_values(detections):
    """Creates values from the raw detections.

    Parameters:
        detections - the model detections.

    Returns: A dictionary with (checkbox name(str):True/False(bool)).
    """
    dataframe = make_detections_df(detections)
    dataframe = add_section_column_to_df(dataframe)
    dataframe = add_box_name_column_to_df(dataframe)
    keys = list(dataframe["name"])
    values = list(dataframe["checked"])
    return {keys[ix]: values[ix] for ix in range(0, len(keys))}


def make_detections_df(detections):
    """Makes a dataframe from the detections.

    Parameters:
        detections - the list of the detections from make_detections()

    Returns: A sorted dataframe with the detections.
    """
    columns = ["x1", "y1", "x2", "y2", "checked", "confidence"]
    dataframe = pd.DataFrame(detections, columns=columns)
    dataframe = dataframe.sort_values(["x1"])
    return dataframe


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
