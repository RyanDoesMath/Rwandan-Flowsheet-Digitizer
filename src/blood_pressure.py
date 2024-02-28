"""The blood_pressure module extracts the data from the blood
pressure section of the Rwandan flowsheet using YOLOv8."""

import warnings
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from ultralytics import YOLO
import tiles
import deshadow
from bounding_box import BoundingBox

BLOOD_PRESSURE_MODEL = YOLO("../models/bp_model_yolov8l.pt")
TWOHUNDRED_THIRTY_MODEL = YOLO("../models/30_200_detector_yolov8l.pt")
BP_TILE_DATA = {"ROWS": 2, "COLUMNS": 8, "STRIDE": 1 / 2}


@dataclass
class BloodPressure:
    """Data class that is a struct for blood pressure.

    Attributes :
        box - The bounding box for the detections
        systolic - The systolic blood pressure.
        diastolic - The diastolic blood pressure.
        timestamp - The timestamp.
    """

    systolic_box: BoundingBox = None
    diastolic_box: BoundingBox = None
    systolic: int = None
    diastolic: int = None
    timestamp: int = None

    def get_box_x_center(self, box_type: str) -> float:
        """Computes the x center of the systolic box."""

        if box_type == "systolic":
            if self.systolic_box is not None:
                return self.systolic_box.get_x_center()
            else:
                return None
        if box_type == "diastolic" and self.diastolic_box is not None:
            if self.diastolic_box is not None:
                return self.diastolic_box.get_x_center()
            else:
                return None
        raise ValueError(f"BloodPressure doesn't have a box called {box_type}")


def extract_blood_pressure(image, do_preprocessing:bool=True) -> dict:
    """Runs methods in order to extract the blood pressure.

    Args :
        image - a PIL image that has been deshadowed and normalized.

    Returns : a dictionary of detections where the keys are timestamps,
              and the values are tuples with (systolic, diastolic).
    """
    cropped_width = crop_legend_out(image).size[0]
    systolic_pred, diastolic_pred = make_detections(image, do_preprocessing=do_preprocessing)
    bp_pred = {"systolic": systolic_pred, "diastolic": diastolic_pred}
    bp_pred = find_timestamp_for_bboxes(bp_pred, cropped_width)
    bp_pred = find_bp_value_for_bbox(image, bp_pred)
    return bp_pred


def make_detections(image, do_preprocessing:bool=True) -> Tuple[List[List[float]], List[List[float]]]:
    """Makes detections using the tile_predict method.

    Args :
        image - a PIL image that has been deshadowed and normalized.

    Returns : A tuple with systolic_boxes, distolic_boxes
    """
    if do_preprocessing:
        img = preprocess_image(image).copy()
    else:
        img = image.copy()
    systolic_pred = tiles.tile_predict(
        BLOOD_PRESSURE_MODEL,
        img,
        rows=BP_TILE_DATA["ROWS"],
        columns=BP_TILE_DATA["COLUMNS"],
        stride=BP_TILE_DATA["STRIDE"],
        overlap_tolerance=0.3,
        remove_non_square=True,
    )
    diastolic_pred = tiles.tile_predict(
        BLOOD_PRESSURE_MODEL,
        img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        rows=BP_TILE_DATA["ROWS"],
        columns=BP_TILE_DATA["COLUMNS"],
        stride=BP_TILE_DATA["STRIDE"],
        overlap_tolerance=0.3,
        remove_non_square=True,
    )
    im_height = img.size[1]
    diastolic_pred = adjust_diastolic_preds(diastolic_pred, im_height)
    systolic_pred, diastolic_pred = remove_legend_predictions(
        image, systolic_pred, diastolic_pred
    )
    return systolic_pred, diastolic_pred


def remove_legend_predictions(
    image, sys_pred: List[List[float]], dia_pred: List[List[float]]
) -> Tuple[List[List[float]], List[List[float]]]:
    """Removes detections made on the legend of the image.

    Args :
        image - a PIL image.
        sys_pred - a list of bounding boxes for the systolic predictions.
        dia_pred - a list of bounding boxes for the diastolic predictions.
    """
    try:
        box_and_class = make_legend_predictions(image)
        two_hundred_box, _ = get_twohundred_and_thirty_box(box_and_class)
    except ValueError:
        box_and_class = make_legend_predictions(preprocess_image(image))
        two_hundred_box, _ = get_twohundred_and_thirty_box(box_and_class)
    sys_pred = list(filter(lambda box: box.left > two_hundred_box.right, sys_pred))
    dia_pred = list(filter(lambda box: box.left > two_hundred_box.right, dia_pred))
    return sys_pred, dia_pred


def preprocess_image(image):
    """Deshadows, normalizes, and denoises a PIL image.

    Args:
        image - a PIL image.

    Returns : a deshadowed, normalized, denoised PIL image.
    """
    img = image.copy()
    img = deshadow.deshadow_and_normalize_image(img)
    return img


###############################################################################
# Legend Cropping
###############################################################################


def crop_legend_out(image):
    """Crops out everything left of the legend.

    Args :
        image - A PIL image of the BP section.

    Returns : a cropped version of the image with only the BP graph.
    """
    width, _ = image.size
    try:
        box_and_class = make_legend_predictions(image)
        two_hundred_box, thirty_box = get_twohundred_and_thirty_box(box_and_class)
    except ValueError:
        box_and_class = make_legend_predictions(preprocess_image(image))
        two_hundred_box, thirty_box = get_twohundred_and_thirty_box(box_and_class)
    top = two_hundred_box.top
    bottom = thirty_box.bottom
    right = max(two_hundred_box.right, thirty_box.right)

    small_offset = 3
    crop = image.crop((right, top, width, bottom + small_offset))
    return crop


def make_legend_predictions(image) -> List[BoundingBox]:
    """Predicts where 200 and 30 are on the image.

    This function first crops out the rightmost four fifths of the image.
    This allows the YOLOv8 model to predict on relatively larger objects
    and slightly speeds up prediction.

    Args :
        image - A PIL image of the BP section.

    Returns : A list of boxes with the prediction data.
    """
    width, height = image.size
    crop = image.crop([0, 0, width // 5, height])

    preds = TWOHUNDRED_THIRTY_MODEL(crop, verbose=False)
    box_and_class = preds[0].boxes.data.tolist()
    box_and_class = [
        BoundingBox(b[0], b[1], b[2], b[3], b[5], b[4]) for b in box_and_class
    ]

    return box_and_class


def get_twohundred_and_thirty_box(
    box_and_class: List[BoundingBox],
) -> Tuple[List[float], List[float]]:
    """From the predictions, returns the twohundred box and thirty box.

    Since there could in theory be erronous predictions, or a missed prediction,
    this function will attempt to filter the predictions to the highest
    confidence predictions, and in the case where there is a missing prediction,
    it will raise an exception.

    Args :
        box_and_class - A list of box predictions.

    Returns : A tuple containing (two hundred box, thirty box)
    """
    two_hundred = 0.0
    thirty = 1.0

    two_hundred_boxes = list(
        filter(lambda bnc: bnc.predicted_class == two_hundred, box_and_class)
    )
    thirty_boxes = list(
        filter(lambda bnc: bnc.predicted_class == thirty, box_and_class)
    )
    if len(two_hundred_boxes) == 0:
        raise ValueError("No detection for 200 on the legend.")
    if len(thirty_boxes) == 0:
        raise ValueError("No detection for 30 on the legend.")

    two_hundred_boxes.sort(key=lambda box: box.confidence)
    thirty_boxes.sort(key=lambda box: box.confidence)

    two_hundred_box = two_hundred_boxes[0]
    thirty_box = thirty_boxes[0]

    return two_hundred_box, thirty_box


def adjust_diastolic_preds(preds: List[BoundingBox], image_height: int):
    """Flips the diastolic predictions back around."""
    for box in preds:
        box.bottom = image_height - box.bottom
        box.top = image_height - box.top
        box.bottom, box.top = box.top, box.bottom
    return preds


###############################################################################
# BP Values
###############################################################################


def find_bp_value_for_bbox(
    image, blood_pressure_predictions: List[BloodPressure]
) -> List[BloodPressure]:
    """Finds the blood pressure value for each bounding box in preds.

    Args :
        image - A PIL image of the legend cropped BP section.
        preds -

    Returns:
        A list of predicted values to put into a column of the dataframe.
    """
    bp_values_for_y_pixel = get_bp_values_for_all_y_pixels(image)
    cropped_image = crop_legend_out(image)
    cropped_image = preprocess_image(cropped_image)
    cropped_image = cropped_image.crop(
        [0, 0, cropped_image.width // 6, cropped_image.height]
    )
    cropped_image = ImageOps.invert(cropped_image)
    blood_pressure_predictions = adjust_boxes_for_margins(
        image, blood_pressure_predictions
    )
    for blood_pressure in blood_pressure_predictions:
        has_systolic = blood_pressure.systolic_box is not None
        has_diastolic = blood_pressure.diastolic_box is not None
        if has_systolic:
            blood_pressure_sys_center = int(
                round(blood_pressure.systolic_box.get_y_center(), 0)
            )
            blood_pressure.systolic = bp_values_for_y_pixel[blood_pressure_sys_center]
        if has_diastolic:
            blood_pressure_dia_center = int(
                round(blood_pressure.diastolic_box.get_y_center(), 0)
            )
            blood_pressure.diastolic = bp_values_for_y_pixel[blood_pressure_dia_center]
        if not has_systolic and not has_diastolic:
            warnings.warn("Box has no systolic or distolic prediction.")

    return blood_pressure_predictions


def get_bp_values_for_all_y_pixels(image):
    """Finds the BP values associated with all y-pixels in the image.

    Args :
        image - A PIL image of the BP section with the legend cropped out.

    Returns : a list with a BP value for every y-pixel.
    """
    proposed_bp_lines = propose_array_of_bp_lines(image)
    bp_lines = correct_array_of_bp_lines(proposed_bp_lines)
    bp_array = apply_bp_values_to_lines(bp_lines)
    return bp_array


def get_y_axis_histogram(image):
    """Generates a normalized pixel histogram for all y values.

    EX:
    |-----|      |-----|
    |  *  |  ->  |*    |                 ->
    | * * |  ->  |**   |  =   [1, 2, 2]  ->  [0.5, 1, 1]
    |*   *|  ->  |**   |                 ->
    |-----|      |-----|

    Parameters:
        image:
            A pil image.

    Returns:
        A normalized pixel histogram of x axis values cast to the y axis.
    """
    image = deshadow.cv2_to_pil(image)
    image = image.convert("L")
    y_axis_hist = np.sum(np.array(image) / 255, axis=1)
    return y_axis_hist


def propose_array_of_bp_lines(image: Image.Image) -> np.array:
    """

    Returns : An array of 0s and 1s with proposed locations for the lines.
    """
    horizontal_line_intervals = [
        (0.0, 0.0),
        (0.056533251056695474, 0.06159073783571166),
        (0.11542740402874478, 0.12028660897067465),
        (0.17484168058745772, 0.17901180589936436),
        (0.23311291446062804, 0.23889253107716374),
        (0.29400319231533045, 0.2978177066758504),
        (0.35232945757779566, 0.3559548680799348),
        (0.4102788394469044, 0.4155862656077412),
        (0.4686683781735512, 0.4725458358776743),
        (0.5263428784299743, 0.5318996991666031),
        (0.5871310838784735, 0.5914795362558752),
        (0.6458323134992556, 0.6503497744670448),
        (0.7049369688901181, 0.708276448501646),
        (0.7625337320460511, 0.7665968331490604),
        (0.8210819364239584, 0.8267498674511095),
        (0.8803272536695689, 0.8839648383371167),
        (0.9389473275414888, 0.9440461403363014),
        (1.0, 1.0),
    ]
    width, height = image.size
    cropped_img = image.crop([0, 0, width // 5, height])
    dets = TWOHUNDRED_THIRTY_MODEL(cropped_img, verbose=False)[0].boxes.data.tolist()
    loc_200 = sorted(list(filter(lambda x: x[5] == 1.0, dets)), key=lambda x: x[4])[0]
    loc_200 = loc_200[1] + (loc_200[3] - loc_200[1]) / 1.5

    loc_30 = sorted(list(filter(lambda x: x[5] == 0.0, dets)), key=lambda x: x[4])[0]
    adjust = int(round(loc_30[1], 0))
    loc_30 = loc_30[1] + (loc_30[3] - loc_30[1]) / 2

    val = loc_200 - loc_30
    readj_intervals = [
        (loc_30 + (x * val), loc_30 + (y * val)) for (x, y) in horizontal_line_intervals
    ]
    readj_intervals = [(math.floor(x), math.floor(y)) for (x, y) in readj_intervals]
    lines = [[x for x in range(a, b + 1)] for (a, b) in readj_intervals]
    lines = [x for y in lines for x in y]
    vals = [1 if x in lines else 0 for x in range(0, height + 1)]
    vals = vals[
        adjust:
    ]  # adjusts for the fact that the boxes are in the cropped image not the full image.
    return vals


def get_contiguous_array_sections(array: np.array) -> List[Tuple[Tuple[int], np.array]]:
    """Gets the contigious sections of an array where 0 is considered a break.
    Args :
        array - the numpy array to get the contiguous sections of.

    Returns : A list of tuples containg ((start index, end index), values).
    """
    if len(array) == 0:
        return []

    contiguous_array_sections = []
    accumulated_section = []
    prev_val = array[0]
    if prev_val != 0:
        accumulated_section.append(array[0])
        section_start_index = 0

    for index, val in enumerate(array):
        if index == 0:
            continue
        if val == 0 and prev_val != 0:
            contiguous_array_sections.append(
                [(section_start_index, index), accumulated_section]
            )
            accumulated_section = []
        elif val != 0 and prev_val == 0:
            section_start_index = index
            accumulated_section.append(val)
        elif val != 0 and prev_val != 0:
            accumulated_section.append(val)
        prev_val = val
    if len(accumulated_section) > 0:
        contiguous_array_sections.append(
            [(section_start_index, len(array)), accumulated_section]
        )

    return [(tup, np.array(x)) for (tup, x) in contiguous_array_sections]


def correct_array_of_bp_lines(bp_lines: np.array) -> np.array:
    """Removes erroneous proposed lines, and inserts lines that track with the
    structure of the sheet as well as a-priori analysis of where the lines
    typically are.

    Args :
        bp_lines - a numpy array of 0s and 1s that propose locations for the bp lines.

    Returns : A corrected array of bp_lines.
    """
    return bp_lines


def apply_bp_values_to_lines(bp_lines: np.array) -> np.array:
    """Applies values to the bp lines array of 1s and 0s.

    Args :
        bp_lines - the array that contains the locations of the horizontal
                   lines on the image that denote 10 bp changes.

    Returns : An array where a BP mmHg value is associated with each item.
    """
    skeleton_bp_array = assign_bp_to_array_vals(bp_lines)
    full_bp_array = fill_gaps_in_bp_array(skeleton_bp_array)
    return full_bp_array


def assign_bp_to_array_vals(thresholded_array):
    """Assigns a blood pressure to the 1 values in the thresholded array.

    EX:
    Input:
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, ..., 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    Output:
    [0, 0, 0, 200, 200, 200, 0, 0, 0, 190, 190, 0, 0, ..., 0, 40, 40, 40, 0, 0, 0, 30, 0, 0]

    Parameters:
        thresholded_array:
            numpy array of one hot coded pixels that correspond to the places where bp lines
            are in the image of the blood pressure section.

    Returns:
        A version of the array with the correct blood pressures in place of the ones.
    """
    bp_at_pixel = thresholded_array.copy()
    blood_pressure = 200  # highest BP on sheet
    prev = 0  # tracks previous pixel

    for i, value in enumerate(bp_at_pixel):
        pix_val = value
        if pix_val == 1 and prev == 0:
            # begins swapping section of ones with bp value.
            prev = 1
            bp_at_pixel[i] = blood_pressure
        elif pix_val == 1 and prev == 1:
            # continues swapping section of ones with bp value.
            bp_at_pixel[i] = blood_pressure
        elif pix_val == 0 and prev == 1:
            # ends swapping and reduces BP for the next section of ones.
            prev = 0
            blood_pressure -= 10

    return bp_at_pixel


def fill_gaps_in_bp_array(array_with_gaps):
    """Interpolates the gaps in the blood pressure array.

    EX:
    input:
    [0, 0, 0, 200, 200, 200, 0, 0, 0, 190, 190, 0, 0, ..., ]
    [0, 0, 0, 200, 200, 200, 197.5, 195, 192.5, 190, 190, 189, 188, ...,]

    Parameters:
        array_with_gaps:
            The numpy array with gaps to fill.
    Returns:
        A numpy array with interpolated gaps.
    """

    input_array = array_with_gaps.copy()
    output_array = array_with_gaps.copy()
    found_first_non_zero_value = False

    for index, value in enumerate(input_array):
        if value != 0:
            found_first_non_zero_value = True
            continue
        if not found_first_non_zero_value:
            continue
        try:
            output_array[index] = interpolate_value(input_array, index)
        except ZeroDivisionError as _:
            output_array[index] = input_array[index]
        except IndexError as _:
            output_array[index] = input_array[index]

    output_array = [int(round(x, 0)) for x in output_array]
    return output_array


def interpolate_value(array, target_index: int) -> float:
    """Interpolates with the closest non-zero values given an array.

    This function does not handle errors. Always use in a try catch block.

    Returns : The interpolated value.
    """
    left_ix = target_index
    right_ix = target_index

    while left_ix != 0 and array[left_ix] == 0:
        left_ix -= 1
    while right_ix != len(array) and array[right_ix] == 0:
        right_ix += 1

    left = array[left_ix]
    right = array[right_ix]
    dist = right_ix - left_ix
    return left - abs(left_ix - target_index) * ((left - right) / dist)


def adjust_boxes_for_margins(
    image, detections: List[BloodPressure]
) -> List[BloodPressure]:
    """Adjusts the boxes for the margins created by the 200 30 crop.

    Args :
        image - a PIL image.
        detections - the final detections with BP value and timestamps.
    Returns :
    """
    box_and_class = make_legend_predictions(image)
    two_hundred_box, _ = get_twohundred_and_thirty_box(box_and_class)
    for det in detections:
        if det.systolic_box is not None:
            det.systolic_box = BoundingBox(
                det.systolic_box.left,
                det.systolic_box.top - two_hundred_box.top,
                det.systolic_box.right,
                det.systolic_box.bottom - two_hundred_box.top,
                det.systolic_box.predicted_class,
                det.systolic_box.confidence,
            )
        if det.diastolic_box is not None:
            det.diastolic_box = BoundingBox(
                det.diastolic_box.left,
                det.diastolic_box.top - two_hundred_box.top,
                det.diastolic_box.right,
                det.diastolic_box.bottom - two_hundred_box.top,
                det.diastolic_box.predicted_class,
                det.diastolic_box.confidence,
            )
    return detections


###############################################################################
# Timestamps
###############################################################################


def find_timestamp_for_bboxes(
    bp_bounding_boxes: Dict[str, List[float]], cropped_image_width: int
) -> List[BloodPressure]:
    """Finds the timestamp for all bounding boxes detected.

    This function goes through a series of steps to impute a timestamp based on
    x distances.

    Args :
        bp_bounding_boxes - the bounding boxes detected as a tuple (sys, dia)

    Returns : A list with BloodPressures sorted by timestamp.
    """
    threshold_dist_for_match = 0.01 * cropped_image_width
    blood_pressures = get_matches(bp_bounding_boxes, threshold_dist_for_match)
    timestamped_blood_pressures = timestamp_blood_pressures(blood_pressures)
    return timestamped_blood_pressures


def get_matches(
    bp_bounding_boxes: Dict[str, List[float]], threshold_dist_for_match: float
) -> List[BloodPressure]:
    """Gets the matches where a match is computed by two bounding boxes that are
    reasonably within each other's
    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.
        threshold_dist_for_match - all bp boxes within the threshold are considered
                                   possible matches. The minimum is selected.

    Returns : A list of blood pressures.
    """
    no_systolic_detections = len(bp_bounding_boxes["systolic"]) == 0
    no_diastolic_detections = len(bp_bounding_boxes["diastolic"]) == 0
    if no_systolic_detections and no_diastolic_detections:
        return ([], [])
    if no_systolic_detections:
        return (
            [BloodPressure(diastolic_box=db) for db in bp_bounding_boxes["diastolic"]],
        )
    if no_diastolic_detections:
        return (
            [BloodPressure(systolic_box=sb) for sb in bp_bounding_boxes["systolic"]],
        )

    matches = []
    for sys_index, sys_box in enumerate(bp_bounding_boxes["systolic"]):
        distance_to_diastolics = [
            (dia_index, abs(sys_box.get_x_center() - dia_box.get_x_center()))
            for dia_index, dia_box in enumerate(bp_bounding_boxes["diastolic"])
        ]
        distance_to_diastolics = list(
            filter(
                lambda tup: tup[0] not in [mat[1] for mat in matches],
                distance_to_diastolics,
            )
        )
        distance_to_diastolics = list(
            filter(
                lambda tup: tup[1] < threshold_dist_for_match, distance_to_diastolics
            )
        )
        if len(distance_to_diastolics) > 0:
            distance_to_diastolics.sort(key=lambda tup: tup[1])
            dia_index = distance_to_diastolics[0][0]
            matches.append((sys_index, dia_index))
    sys_non_matches = list(
        {x for x in range(0, len(bp_bounding_boxes["systolic"]))}
        - {tup[0] for tup in matches}
    )
    sys_non_matches = [
        BloodPressure(systolic_box=bp_bounding_boxes["systolic"][index])
        for index in sys_non_matches
    ]
    dia_non_matches = list(
        {x for x in range(0, len(bp_bounding_boxes["diastolic"]))}
        - {tup[1] for tup in matches}
    )
    dia_non_matches = [
        BloodPressure(diastolic_box=bp_bounding_boxes["diastolic"][index])
        for index in dia_non_matches
    ]
    non_matches = sys_non_matches + dia_non_matches
    matches = [
        BloodPressure(
            systolic_box=bp_bounding_boxes["systolic"][tup[0]],
            diastolic_box=bp_bounding_boxes["diastolic"][tup[1]],
        )
        for tup in matches
    ]

    return matches + non_matches


def timestamp_blood_pressures(
    blood_pressures: List[BloodPressure],
) -> List[BloodPressure]:
    """Applies a timestamp to all the blood pressure structs.

    Args :
        blood_pressures - the blood pressure structs without timestamps.

    Returns : the blood pressure structs with timestamps.
    """

    def average_x_coord(blood_pressure: BloodPressure) -> float:
        no_systolic_box = blood_pressure.systolic_box is None
        no_diastolic_box = blood_pressure.diastolic_box is None
        if no_systolic_box:
            return blood_pressure.diastolic_box.get_x_center()
        if no_diastolic_box:
            return blood_pressure.systolic_box.get_x_center()
        sys_x_center = blood_pressure.systolic_box.get_x_center()
        dia_x_center = blood_pressure.diastolic_box.get_x_center()
        return (sys_x_center + dia_x_center) / 2

    stamped_bps = sorted(blood_pressures, key=average_x_coord)
    for index, blood_pressure in enumerate(stamped_bps):
        blood_pressure.timestamp = index * 5
    return stamped_bps


def show_detections(image, do_preprocessing:bool=True):
    """Draws the bp detections on the image."""
    if do_preprocessing:
        img = preprocess_image(image).copy()
    else:
        img = image.copy()
    systolic_pred, diastolic_pred = make_detections(image, do_preprocessing=do_preprocessing)
    draw = ImageDraw.Draw(img)
    for box in systolic_pred:
        draw.rectangle(box.get_box(), outline="#fbb584")
    for box in diastolic_pred:
        draw.rectangle(box.get_box(), outline="#6c799c")
    return img


def show_detections_30_200(image: Image.Image):
    """Draws the 30 200 detections on the image."""
    img = preprocess_image(image).copy()
    width, height = img.size
    img = img.crop([0, 0, width // 5, height])
    bboxes = TWOHUNDRED_THIRTY_MODEL(img, verbose=False)[0].boxes.data.tolist()
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], width=2, outline="black")
    return img
