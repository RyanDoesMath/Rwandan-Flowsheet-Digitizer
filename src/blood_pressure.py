"""The blood_pressure module extracts the data from the blood
pressure section of the Rwandan flowsheet using YOLOv8."""

from typing import List, Tuple
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


BLOOD_PRESSURE_MODEL = YOLO("../models/bp_model_yolov8s.pt")
TWOHUNDRED_THIRTY_MODEL = YOLO("../models/30_200_detector_yolov8s.pt")


def extract_blood_pressure(image) -> dict:
    """Runs methods in order to extract the blood pressure.

    Args :
        image - a PIL image that has been deshadowed and normalized.

    Returns : a dictionary of detections where the keys are timestamps,
              and the values are tuples with (systolic, diastolic).
    """
    image = normalize(image)
    image = crop_legend_out(image)
    systolic_pred = BLOOD_PRESSURE_MODEL(image).pandas().xyxy[0]
    diastolic_pred = (
        BLOOD_PRESSURE_MODEL(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        .pandas()
        .xyxy[0]
    )
    systolic_pred, diastolic_pred = filter_and_adjust_bp_predictions(
        systolic_pred, diastolic_pred, image
    )
    bp_pred = combine_predictions(systolic_pred, diastolic_pred)
    bp_pred["predicted_values_mmhg"] = find_bp_value_for_bbox(image, bp_pred)
    bp_pred["predicted_timestamp_mins"] = find_timestamp_for_bbox(image, bp_pred)
    bp_pred = filter_duplicate_detections(bp_pred)
    return bp_pred


def filter_and_adjust_bp_predictions(
    systolic_predictions, diastolic_predictions, image
):
    """Filters the blood pressure predictions and unflips the diastolic predictions."""
    systolic_predictions = filter_bp_predictions(systolic_predictions)
    diastolic_predictions = filter_bp_predictions(diastolic_predictions)
    diastolic_predictions = adjust_diastolic_preds(diastolic_predictions, image.size[1])
    return systolic_predictions, diastolic_predictions


def combine_predictions(systolic_predictions, diastolic_predictions):
    """Combines the predictions together into one dataframe."""
    systolic_predictions["name"] = "systolic"
    diastolic_predictions["name"] = "diastolic"
    bp_predictions = pd.concat([systolic_predictions, diastolic_predictions])
    bp_predictions = bp_predictions.reset_index(drop=True)
    bp_predictions = bp_predictions.drop(["flagged_for_removal", "class"], axis=1)
    return bp_predictions


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


def normalize(image):
    """Normalizes the image for better prediction."""
    img = pil_to_cv2(image)
    img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    readjusted = img_normalized * 255
    readjusted = readjusted.astype(np.uint8)
    im_normalized = Image.fromarray(readjusted)

    return im_normalized


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


def box_area(box):
    """Computes the area of a box."""
    return (box[2] - box[0]) * (box[3] - box[1])


def filter_bp_predictions(preds):
    """Filters overlapping bounding boxes out of the predictions.

    Parameters:
        preds - the pandas prediction dataframe from the yolo model.
    Returns: A pandas dataframe with less or no erroneous predictions.
    """
    threshold = 0.5
    temp = preds.copy()
    temp["flagged_for_removal"] = False

    def get_box(row):
        return (row.xmin, row.ymin, row.xmax, row.ymax)

    for _, this_row in preds.iterrows():
        this_box = get_box(this_row)
        for that_ix, that_row in preds.iterrows():
            that_box = get_box(that_row)
            percent_overlap = bb_intersection(this_box, that_box) / box_area(this_box)
            if percent_overlap > threshold and percent_overlap != 1:
                temp.at[that_ix, "flagged_for_removal"] = True

    detections_filtered = remove_filtered_detections(temp)

    return detections_filtered


def remove_filtered_detections(detections_flagged):
    """Removes the filtered detections from the dataframe."""
    mask = ~detections_flagged["flagged_for_removal"]
    detections_filtered = detections_flagged[mask].copy()
    detections_filtered.reset_index(drop=True, inplace=True)
    return detections_filtered


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
        cntr = compute_center(row, len(H[0]))
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


def binarized_vertical_lines(img):
    """Binarizes an image and removes everything except vertical lines.

    Parameters:
        img - the image to binarize and remove all non-vertical lines from.

    Returns: A version of the image that is binarized and has only vertical lines."""
    cv2_img = np.array(img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    # convert to greyscale
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    # greyscale to binary
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    vertical = np.copy(bw)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, vertical_structure, iterations=2)
    vertical = cv2.dilate(vertical, vertical_structure, iterations=2)
    return vertical


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


def find_timestamp_for_bbox(image, preds):
    """Gets the timestamp for each bounding box in the predictions.

    Parameters:
        image - The image to find the timestamp for.
        preds - A dataframe with the bounding boxes.

    Returns: A list of timestamps to put into a column of the dataframe.
    """
    predicted_timestamps = []
    timestamp_at_pixel = get_minutes_array(image)

    for _, row in preds.iterrows():
        value = timestamp_at_pixel[int(((row.xmax - row.xmin) / 2) + row.xmin)]
        predicted_timestamps.append(value)

    return predicted_timestamps


def get_minutes_array(image):
    """Gets a 1d array with the minute for each pixel value on the x axis of the image.

    This method is trash :(

    Parameters:
        image - the image to predict on.

    Returns - a 1d array with the minute for each pixel value on the x axis of the image.
    """
    vertical = binarized_vertical_lines(image)
    y_axis_hist = np.sum(vertical / 255, axis=0)
    y_axis_hist = [x / max(y_axis_hist) for x in y_axis_hist]

    five_minute_markers = np.array([1 if y >= 0.3 else 0 for y in y_axis_hist])
    minute = 0
    temp = five_minute_markers.copy()
    previous = 0
    counter = 0
    for index, current in enumerate(five_minute_markers):
        if current == 1 and previous == 0:
            temp[index] = minute
        elif current == 1 and previous == 1:
            temp[index] = minute
        elif current == 0 and previous == 1:
            minute += 5
        previous = current
    temp[0] = 0

    closest_right = 0
    closest_left = 0
    counter_right = 0
    counter_left = 0
    final_array = temp.copy()
    for index, i in enumerate(temp):
        if i != 0:
            continue
        while closest_right == 0 and index + counter_right < len(temp):
            if index + counter > len(temp):
                break
            closest_right = temp[index + counter_right]
            counter_right += 1

        while closest_left == 0 and index - counter_left > 0:
            if index - counter < 0:
                break
            closest_left = temp[index - counter_left]
            counter_left += 1

        if counter_right < counter_left:
            final_array[index] = closest_right
        elif counter_left <= counter_right:
            final_array[index] = closest_left
        closest_right = 0
        closest_left = 0
        counter_right = 0
        counter_left = 0

    final_array = [0 if i == 1 else i for i in final_array]
    return final_array


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


def show_detections(image):
    """Draws the bp detections on the image."""
    extractions = extract(image)
    img = crop_legend_out(image)
    draw = ImageDraw.Draw(img)
    for _, det in extractions.iterrows():
        box = (det["xmin"], det["ymin"], det["xmax"], det["ymax"])
        draw.rectangle(box, outline="red")
    return img


def cv2_to_pil(cv2_image):
    """Converts a cv2 image to a PIL image."""
    color_converted = cv2.cvtColor(cv2.bitwise_not(cv2_image), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    return pil_image


def pil_to_cv2(pil_img):
    """Converts a PIL image to a cv2 image."""
    open_cv_image = np.array(pil_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image
