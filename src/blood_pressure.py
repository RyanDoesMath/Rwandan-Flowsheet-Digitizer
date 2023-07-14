"""The blood_pressure module extracts the data from the blood
pressure section of the Rwandan flowsheet using YOLOv8."""

import warnings
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import tiles
import deshadow

BLOOD_PRESSURE_MODEL = YOLO("../models/bp_model_yolov8m_retrain.pt")
TWOHUNDRED_THIRTY_MODEL = YOLO("../models/30_200_detector_yolov8s.pt")
BP_TILE_DATA = {"ROWS": 6, "COLUMNS": 17, "STRIDE": 1 / 2}


@dataclass
class BloodPressure:
    """Data class that is a struct for blood pressure.

    Attributes :
        box - The bounding box for the detections
        systolic - The systolic blood pressure.
        diastolic - The diastolic blood pressure.
        timestamp - The timestamp.
    """

    systolic_box: List[float] = None
    diastolic_box: List[float] = None
    systolic: int = None
    diastolic: int = None
    timestamp: int = None


def extract_blood_pressure(image) -> dict:
    """Runs methods in order to extract the blood pressure.

    Args :
        image - a PIL image that has been deshadowed and normalized.

    Returns : a dictionary of detections where the keys are timestamps,
              and the values are tuples with (systolic, diastolic).
    """
    image = preprocess_image(image)
    image = crop_legend_out(image)
    systolic_pred = tiles.tile_predict(
        BLOOD_PRESSURE_MODEL,
        image,
        rows=BP_TILE_DATA["ROWS"],
        columns=BP_TILE_DATA["COLUMNS"],
        stride=BP_TILE_DATA["STRIDE"],
        overlap_tolerance=0.5,
    )
    diastolic_pred = tiles.tile_predict(
        BLOOD_PRESSURE_MODEL,
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        rows=BP_TILE_DATA["ROWS"],
        columns=BP_TILE_DATA["COLUMNS"],
        stride=BP_TILE_DATA["STRIDE"],
        overlap_tolerance=0.5,
    )
    print("sys_pred", systolic_pred)
    print("dia_pred", diastolic_pred)
    diastolic_pred = adjust_diastolic_preds(diastolic_pred, image.size[1])
    bp_pred = {"systolic": systolic_pred, "diastolic": diastolic_pred}
    bp_pred = find_timestamp_for_bboxes(bp_pred)
    bp_pred = find_bp_value_for_bbox(image, bp_pred)
    return bp_pred


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
    box_and_class = make_legend_predictions(image)

    two_hundred_box, thirty_box = get_twohundred_and_thirty_box(box_and_class)
    top = two_hundred_box[1]
    bottom = thirty_box[3]
    right = max(two_hundred_box[2], thirty_box[2])

    small_offset = 3
    crop = image.crop((right, top, width, bottom + small_offset))
    return crop


def make_legend_predictions(image) -> List[List[float]]:
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

    return box_and_class


def get_twohundred_and_thirty_box(
    box_and_class: List[List[float]],
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
    index_of_confidence = 4
    index_of_class = 5

    two_hundred_boxes = list(
        filter(lambda bnc: bnc[index_of_class] == two_hundred, box_and_class)
    )
    thirty_boxes = list(
        filter(lambda bnc: bnc[index_of_class] == thirty, box_and_class)
    )
    if len(two_hundred_boxes) == 0:
        raise ValueError("No detection for 200 on the legend.")
    if len(thirty_boxes) == 0:
        raise ValueError("No detection for 30 on the legend.")

    two_hundred_boxes.sort(key=lambda box: box[index_of_confidence])
    thirty_boxes.sort(key=lambda box: box[index_of_confidence])

    two_hundred_box = two_hundred_boxes[0]
    thirty_box = thirty_boxes[0]

    return two_hundred_box, thirty_box


def bb_intersection(box_a, box_b):
    """Finds the bounding box intersection for two rectangles.

    Parameters:
        boxA - A tuple containing the box's data (xmin, ymax, xmax, ymin)
        boxB - A tuple containing the box's data (xmin, ymax, xmax, ymin)

    Returns: The area of intersection for the two bounding boxes.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    left = max(box_a[0], box_b[0])
    bottom = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    top = min(box_a[3], box_b[3])

    if left >= right and bottom >= top:
        return 0

    # compute the area of intersection rectangle
    area_of_intersection = (right - left) * (top - bottom)
    return area_of_intersection


def adjust_diastolic_preds(preds, image_height):
    """Flips the diastolic predictions back around."""
    for box in preds:
        box[3] = image_height - box[3]
        box[1] = image_height - box[1]
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

    def compute_box_y_center(box: List[float]):
        return int(round(box[3] + (box[3] - box[1]), 0))

    horizontal_lines = extract_horizontal_lines(image)
    bp_values_for_y_pixel = get_bp_values_for_all_y_pixels(horizontal_lines)
    for blood_pressure in blood_pressure_predictions:
        has_systolic = blood_pressure.systolic_box is not None
        has_diastolic = blood_pressure.diastolic_box is not None
        if has_systolic:
            blood_pressure_sys_center = compute_box_y_center(
                blood_pressure.systolic_box
            )
            blood_pressure.systolic = bp_values_for_y_pixel[blood_pressure_sys_center]
        if has_diastolic:
            blood_pressure_dia_center = compute_box_y_center(
                blood_pressure.diastolic_box
            )
            blood_pressure.diastolic = bp_values_for_y_pixel[blood_pressure_dia_center]
        if not has_systolic and not has_diastolic:
            warnings.warn("Box has no systolic or distolic prediction.")
    return blood_pressure_predictions


def extract_horizontal_lines(image):
    """Binarizes an image and removes everything except horizontal lines.

    Args :
        image - The PIL image to extract the horizontal lines from

    Returns: A PIL image that is binarized and has only horizontal lines.
    """
    cv2_img = deshadow.pil_to_cv2(image)
    grey = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(grey)
    black_and_white = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    horizontal = np.copy(black_and_white)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 50
    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1)
    )
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure, iterations=3)
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure, iterations=3)
    horizontal = cv2.bitwise_not(horizontal)
    return horizontal


def get_bp_values_for_all_y_pixels(image):
    """Finds the BP values associated with all y-pixels in the image.

    Args :
        image - A PIL image of the BP section with the legend cropped out.

    Returns : a list with a BP value for every y-pixel.
    """
    y_hist = get_y_axis_histogram(image)
    proposed_bp_lines = propose_array_of_bp_lines(y_hist)
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


def propose_array_of_bp_lines(bp_hist: np.array) -> np.array:
    """Proposes an array where 0 indicates space between the BP demarkations,
    and 1 indicates a line that demarkates where 10 mmHg have changed, that is
    the location of the horizontal lines that encode blood pressure.

    Uses a binary search for thresholds for performance.

    Args :
        bp_hist - the binarized histogram of the BP image with horizontal lines
        extracted.

    Returns : An array of 0s and 1s with proposed locations for the lines.
    """
    num_of_lines_on_sheet = 18
    high_threshold = max(bp_hist)
    low_threshold = 0
    threshold = (high_threshold + low_threshold) // 2
    threshed_hist = [0 if x < threshold else 1 for x in bp_hist]
    number_of_contiguous_array_sections = len(
        get_contiguous_array_sections(threshed_hist)
    )
    iters = 0
    while number_of_contiguous_array_sections != num_of_lines_on_sheet:
        if number_of_contiguous_array_sections < num_of_lines_on_sheet:
            high_threshold = threshold
        if number_of_contiguous_array_sections > num_of_lines_on_sheet:
            low_threshold = threshold
        threshold = (high_threshold + low_threshold) // 2
        threshed_hist = [0 if x < threshold else 1 for x in bp_hist]
        number_of_contiguous_array_sections = len(
            get_contiguous_array_sections(threshed_hist)
        )
        if iters == 50:
            warnings.warn("Could not find value of threshold for 18 bp lines.")
            break
        iters += 1
    return threshed_hist


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

    bp_at_pixel = array_with_gaps.copy()

    idxs_to_change = []
    begin_count = False
    bottom = 200
    prev = 0
    for i, value in enumerate(bp_at_pixel):
        pix_val = value
        if pix_val == 0 and prev != 0:
            begin_count = True
        if pix_val != 0 and prev == 0:
            begin_count = False
            top = pix_val
            delta = (top - bottom) / (len(idxs_to_change) + 3)
            val = bottom
            for j in idxs_to_change:
                val += delta
                bp_at_pixel[j] = val
            idxs_to_change = []
            bottom = top
        if begin_count:
            idxs_to_change.append(i)
        prev = pix_val
    return bp_at_pixel


###############################################################################
# Timestamps
###############################################################################


def find_timestamp_for_bboxes(
    bp_bounding_boxes: Dict[str, List[float]]
) -> List[BloodPressure]:
    """Finds the timestamp for all bounding boxes detected.

    This function goes through a series of steps to impute a timestamp based on
    x distances.

    Args :
        bp_bounding_boxes - the bounding boxes detected as a tuple (sys, dia)

    Returns : A list with BloodPressures sorted by timestamp.
    """
    dists = generate_x_dists_matrix(bp_bounding_boxes)
    dists, non_matches = filter_non_matches(dists, bp_bounding_boxes)
    matches = generate_matches(dists, bp_bounding_boxes)
    timestamped_blood_pressures = timestamp_blood_pressures(matches + non_matches)
    return timestamped_blood_pressures


def generate_x_dists_matrix(
    bp_bounding_boxes: Dict[str, List[float]]
) -> List[List[float]]:
    """Returns a matrix where each row corresponds to a systolic blood pressure,
    each column corresponds to a diastolic blood pressure, and the entries are
    the x distances between the center of the two bounding boxes.

    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.

    Returns : A matrix systolic rows, diastolic columns, and distances as entries.
    """
    dists = []
    systolic_centers = [
        box[2] - (box[2] - box[0]) / 2 for box in bp_bounding_boxes["systolic"]
    ]
    diastolic_centers = [
        box[2] - (box[2] - box[0]) / 2 for box in bp_bounding_boxes["diastolic"]
    ]
    for sys_center in systolic_centers:
        sys_row = []
        for dia_center in diastolic_centers:
            sys_row.append(abs(sys_center - dia_center))
        dists.append(sys_row)
    return dists


def filter_non_matches(
    dists: List[List[float]], bp_bounding_boxes: Dict[str, List[float]]
) -> Tuple[List[List[float]], List[BloodPressure]]:
    """If there are more systolic than diastolic boxes (or vice versa), this
    method will find the systolic box which is the furthest x-distance from a
    matching diastolic box, then remove it and create a BloodPressure struct
    with None for the diastolic box.

    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.
        dists - the matrix of distances where the rows correspond to systolic
                boxes and the columns correspond to diastolic boxes.

    Returns : a tuple with the distances matrix sans the non-matches and the
              non-matches as BloodPressure structs.
    """
    no_systolic_detections = len(bp_bounding_boxes["systolic"]) == 0
    no_diastolic_detections = len(bp_bounding_boxes["diastolic"]) == 0
    if no_systolic_detections and no_diastolic_detections:
        return ([], [])
    if no_systolic_detections:
        return (
            dists,
            [BloodPressure(diastolic_box=db) for db in bp_bounding_boxes["diastolic"]],
        )
    if no_diastolic_detections:
        return (
            dists,
            [BloodPressure(systolic_box=sb) for sb in bp_bounding_boxes["systolic"]],
        )
    dists_was_tranposed = False
    num_rows = len(dists)
    num_columns = len(dists[0])
    if num_columns > num_rows:
        dists = transpose_dists(dists)
        num_rows = len(dists)
        num_columns = len(dists[0])
        dists_was_tranposed = True

    non_matches = []
    while num_rows > num_columns:
        non_match_index = get_index_of_list_with_largest_min_val(dists)
        non_matches.append(non_match_index)
        del dists[non_match_index]
        num_rows = len(dists)
        num_columns = len(dists[0])

    if dists_was_tranposed:
        dists = transpose_dists(dists)
        non_matches = [
            BloodPressure(diastolic_box=bp_bounding_boxes["diastolic"][x])
            for x in non_matches
        ]
    else:
        non_matches = [
            BloodPressure(systolic_box=bp_bounding_boxes["systolic"][x])
            for x in non_matches
        ]

    return dists, non_matches


def transpose_dists(dists: List[List[float]]) -> List[List[float]]:
    """Transposes the dists matrix."""
    return list(map(list, zip(*dists)))


def get_index_of_list_with_largest_min_val(dists: List[List[float]]) -> int:
    """Gets the index of the list in dists with the largest minimum value."""
    list_with_largest_minimum = sorted(dists, key=min)[-1]
    return dists.index(list_with_largest_minimum)


def generate_matches(
    dists: List[List[float]], bp_bounding_boxes: Dict[str, List[float]]
) -> List[BloodPressure]:
    """Generates a list of matched blood pressures.

    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.
        dists - the matrix of distances where the rows correspond to systolic
                boxes and the columns correspond to diastolic boxes. This matrix
                has already had non-matches removed so it is square.

    Returns : A list of BloodPressure structs.
    """
    no_systolic_detections = len(bp_bounding_boxes["systolic"]) == 0
    no_diastolic_detections = len(bp_bounding_boxes["diastolic"]) == 0
    if no_systolic_detections or no_diastolic_detections:
        return []

    matches = []
    while len(dists) > 0:
        smallest_sys = get_index_of_list_with_smallest_min_val(dists)
        smallest_dia = get_index_of_smallest_val(dists[smallest_sys])
        matches.append(
            BloodPressure(
                systolic_box=bp_bounding_boxes["systolic"][smallest_sys],
                diastolic_box=bp_bounding_boxes["diastolic"][smallest_dia],
            )
        )
        del bp_bounding_boxes["systolic"][smallest_sys]
        del bp_bounding_boxes["diastolic"][smallest_dia]
        for row in dists:
            del row[smallest_dia]
        del dists[smallest_sys]
    return matches


def get_index_of_list_with_smallest_min_val(dists: List[List[float]]) -> int:
    """Gets the index of the list in dists with the smallest minimum value."""
    try:
        list_with_smallest_minimum = sorted(dists, key=min)[0]
        return dists.index(list_with_smallest_minimum)
    except IndexError as _:
        warnings.warn(
            "Empty list passed into get_index_of_list_with_smallest_min_val()."
        )
        return None


def get_index_of_smallest_val(row: List[float]) -> int:
    """Gets the index of the smallest value in a list."""
    try:
        return row.index(min(row))
    except ValueError as _:
        warnings.warn("Empty list passed into get_index_of_smallest_val().")
        return None


def timestamp_blood_pressures(
    blood_pressures: List[BloodPressure],
) -> List[BloodPressure]:
    """Applies a timestamp to all the blood pressure structs.

    Args :
        blood_pressures - the blood pressure structs without timestamps.

    Returns : the blood pressure structs with timestamps.
    """

    def compute_box_x_center(box: List[float]):
        return box[2] + (box[2] - box[0])

    def average_x_coord(blood_pressure: BloodPressure) -> float:
        no_systolic_box = blood_pressure.systolic_box is None
        no_diastolic_box = blood_pressure.diastolic_box is None
        if no_systolic_box:
            return compute_box_x_center(blood_pressure.diastolic_box)
        if no_diastolic_box:
            return compute_box_x_center(blood_pressure.systolic_box)
        sys_x_center = compute_box_x_center(blood_pressure.systolic_box)
        dia_x_center = compute_box_x_center(blood_pressure.diastolic_box)
        return (sys_x_center + dia_x_center) / 2

    stamped_bps = sorted(blood_pressures, key=average_x_coord)
    for index, blood_pressure in enumerate(stamped_bps):
        blood_pressure.timestamp = index * 5
    # stamped_bps = [x.set_timestamp(ix * 5) for ix, x in enumerate(stamped_bps)]
    return stamped_bps


def show_detections(image):
    """Draws the bp detections on the image."""
    extractions = extract_blood_pressure(image)
    img = crop_legend_out(image)
    draw = ImageDraw.Draw(img)
    for _, det in extractions.iterrows():
        box = (det["xmin"], det["ymin"], det["xmax"], det["ymax"])
        draw.rectangle(box, outline="red")
    return img
