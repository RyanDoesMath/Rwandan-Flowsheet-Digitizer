"""The blood_pressure module extracts the data from the blood
pressure section of the Rwandan flowsheet using YOLOv8."""

import warnings
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import tiles

BLOOD_PRESSURE_MODEL = YOLO("../models/bp_model_yolov8s.pt")
TWOHUNDRED_THIRTY_MODEL = YOLO("../models/30_200_detector_yolov8s.pt")


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
        rows=4,
        columns=10,
        stride=1 / 2,
        overlap_tolerance=0.5,
    )
    diastolic_pred = tiles.tile_predict(
        BLOOD_PRESSURE_MODEL,
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        rows=4,
        columns=10,
        stride=1 / 2,
        overlap_tolerance=0.5,
    )
    print(systolic_pred, diastolic_pred)
    diastolic_pred = adjust_diastolic_preds(diastolic_pred, image.size[1])
    bp_pred = {"systolic": systolic_pred, "diastolic": diastolic_pred}
    bp_pred["predicted_timestamp_mins"] = find_timestamp_for_bboxes(bp_pred)
    bp_pred["predicted_values_mmhg"] = find_bp_value_for_bbox(image, bp_pred)
    bp_pred = filter_duplicate_detections(bp_pred)
    return bp_pred


def preprocess_image(image):
    """Deshadows, normalizes, and denoises a PIL image.

    Args:
        image - a PIL image.

    Returns : a deshadowed, normalized, denoised PIL image.
    """


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
    temp = preds.copy()
    temp["ymin"] = image_height - temp["ymin"]
    temp["ymax"] = image_height - temp["ymax"]
    return temp


def find_bp_value_for_bbox(image, preds):
    """Finds the blood pressure value for each bounding box in preds.

    Parameters:
        image - the image we get bounding boxes for.
        preds - the predicted bounding boxes.

    Returns:
        A list of predicted values to put into a column of the dataframe.
    """
    predicted_bps = []
    bp_matrix = get_bp_matrix(image)
    for _, row in preds.iterrows():
        cntr = compute_center(row, len(bp_matrix[0]))
        predicted_bps.append(bp_matrix[cntr[0]][cntr[1]])
    return predicted_bps


def get_bp_matrix(image):
    """Calls methods to get the blood pressure matrix from an image."""
    bp_matrix = get_blood_pressure_matrix(binarized_horizontal_lines(image))
    bp_matrix = bp_matrix_to_np_arrays(bp_matrix)
    bp_matrix = np.fliplr(bp_matrix)
    return bp_matrix


def get_y_axis_histogram(image):
    """Generates a normalized pixel histogram for all y values.

    EX:
    |-----|      |-----|
    |  *  |  ->  |*    |  ->             ->
    | * * |  ->  |**   |  ->  [1, 2, 2]  ->  [0.5, 1, 1]
    |*   *|  ->  |**   |  ->             ->
    |-----|      |-----|

    Parameters:
        image:
            A pil image.

    Returns:
        A normalized pixel histogram of x axis values cast to the y axis.
    """
    grayscale_image = image.copy().convert("L")
    y_axis_hist = np.sum(np.array(grayscale_image) / 255, axis=1)
    return y_axis_hist


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


def get_bps_at_y_pixel_values(image, thresh: float = 0.7):
    """Gets the bp at the particular y pixel value.

    Parameters:
        image:
            A sliding window PIL image.
    """
    temp = image.copy()
    y_axis_hist = get_y_axis_histogram(temp)
    thresholded_array = np.array([1 if x > thresh else 0 for x in y_axis_hist])

    bp_at_pixel = assign_bp_to_array_vals(thresholded_array)
    interpolated_bps_at_pixel = fill_gaps_in_bp_array(bp_at_pixel)

    return interpolated_bps_at_pixel


def predict_blood_pressure_from_coordinates(coordinates: tuple):
    """Predicts the blood pressure from the coordinates of a bounding box.

    Parameters:
        coordinates - the pixel coordinates of the middle of the bounding box.

    Returns: The predicted blood pressure of the coordinates.
    """
    bps = get_bps_at_y_pixel_values(temp)
    bps = [0] * int((height - thirty.ymax)) + list(
        bps
    )  # offset for pixels above the "200" marker.
    y_value = int(coordinates[3])
    return bps[y_value]


def get_blood_pressure_matrix(image, window_size: float = 0.6, stride: float = 0.1):
    """Predicts the blood pressure values from an image.

    Parameters:
        image:
            A PIL image with the blood pressure section.

    Returns:
        An array with the blood pressure values from left to right.
    """
    width, height = image.size
    window = [0, 0, int(width * window_size), height]
    prediction_matrix = []
    while window[2] < width:
        bps = get_bps_at_y_pixel_values(image.crop(window).convert("L"))
        new_values = [bps] * int(width * stride)
        prediction_matrix.append(new_values)
        window[2] += width * stride
        window[0] += width * stride

    prediction_matrix = [i for j in prediction_matrix for i in j]
    # fill the remaining space
    prediction_matrix = prediction_matrix + [prediction_matrix[-1]] * int(
        width - len(prediction_matrix)
    )
    return prediction_matrix


def binarized_horizontal_lines(img):
    """Binarizes an image and removes everything except horizontal lines.

    Parameters:
        img - the image to binarize and remove all non-horizontal lines from.

    Returns: A version of the image that is binarized and has only horizontal lines.
    """
    cv2_img = np.array(img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    # convert to greyscale
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    # greyscale to binary
    gray = cv2.bitwise_not(gray)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    horizontal = np.copy(binarized)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1)
    )
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure, iterations=4)
    color_converted = cv2.cvtColor(horizontal, cv2.COLOR_BGR2RGB)
    horizontal = Image.fromarray(color_converted)
    return horizontal


def bp_matrix_to_np_arrays(bp_matrix):
    """Converts the bp matrix to a 2d numpy array."""
    ret_arr = []
    for i in bp_matrix:
        next_line = []
        for j in i:
            next_line.append(j)
        ret_arr.append(np.array(next_line))
    return np.array(ret_arr)


def compute_center(row, height):
    """Computes the center of a row given the height of the image."""
    ymin = height - row.ymin
    ymax = height - row.ymax
    xmin, xmax = row.xmin, row.xmax
    return (int(xmax - ((xmax - xmin) / 2)), int(ymin - ((ymin - ymax) / 2)))


def filter_duplicate_detections(detections):
    """Makes sure there are only one detection per type for each timestamp.

    This method finds whether there are two or more detections for one timestamp of a
    particular type, then removes all but the highest confidence detection.

    Parameters:
        detections - The detections dataframe.

    Returns: A reduced dataframe with likely erroneous detections removed.
    """
    systolics = detections[detections["name"] == "systolic"].copy()
    diastolics = detections[detections["name"] == "diastolic"].copy()

    systolics = filter_duplicate_detections_for_one_bp_type(systolics)
    diastolics = filter_duplicate_detections_for_one_bp_type(diastolics)

    return pd.concat([systolics, diastolics])


def filter_duplicate_detections_for_one_bp_type(detections):
    """Helper function for filter_duplicate_detections()."""
    bps = detections.copy()
    ix_to_remove = []
    for timestamp in list(bps["predicted_timestamp_mins"].unique()):
        temp = bps[bps["predicted_timestamp_mins"] == timestamp]
        if len(temp) == 1:
            continue
        temp = temp.sort_values("confidence", ascending=False)
        for index in temp.index[1:]:
            ix_to_remove.append(index)
    return bps[~(bps.index.isin(ix_to_remove))]


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
