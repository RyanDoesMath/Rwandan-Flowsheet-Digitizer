"""The blood_pressure module extracts the data from the blood
pressure section of the Rwandan flowsheet using YOLOv8."""

from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

    box: List[Tuple[float]]
    systolic: int
    diastolic: int
    timestamp: int


def extract_blood_pressure(image) -> dict:
    """Runs methods in order to extract the blood pressure.

    Args :
        image - a PIL image that has been deshadowed and normalized.

    Returns : a dictionary of detections where the keys are timestamps,
              and the values are tuples with (systolic, diastolic).
    """
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
    dists = get_x_dists_matrix(bp_bounding_boxes)
    matches, non_matches = filter_non_matches(bp_bounding_boxes, dists)


def get_x_dists_matrix(bp_bounding_boxes: Dict[str, List[float]]) -> List[List[float]]:
    """Returns a matrix where each row corresponds to a systolic blood pressure,
    each column corresponds to a diastolic blood pressure, and the entries are
    the x distances between the center of the two bounding boxes.

    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.

    Returns : A matrix systolic rows, diastolic columns, and distances as entries.
    """


def filter_non_matches(
    bp_bounding_boxes: Dict[str, List[float]], dists: List[List[float]]
) -> Tuple(Dict[str, List[float]], Dict[str, List[float]]):
    """Removes bp detections which don't have a pair until there is an equal amount
    of systolic and diastolic predictions.

    Args :
        bp_bounding_boxes - the bounding boxes for the systolic and diastolic bps.
        dists - the matrix of distances between the systolic and diastolic boxes.

    Returns : The predictions separated into two different dictionaries as
              a tuple (matches, non-matches)
    """


def show_detections(image):
    """Draws the bp detections on the image."""
    extractions = extract_blood_pressure(image)
    img = crop_legend_out(image)
    draw = ImageDraw.Draw(img)
    for _, det in extractions.iterrows():
        box = (det["xmin"], det["ymin"], det["xmax"], det["ymax"])
        draw.rectangle(box, outline="red")
    return img
