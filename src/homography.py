"""The homography module predicts document landmarks, computes a homography matrix,
and returns a rectified image based on the landmarks identified."""

from ultralytics import YOLO
from cv2 import warpPerspective, findHomography
import pandas as pd
import numpy as np
import deshadow

CORNER_MODEL = YOLO("../models/four_corners_detector_yolov8s.pt")


def correct_image(image: np.ndarray):
    """Uses a yolov8 model and a"""
    if not isinstance(image, np.ndarray):
        image = deshadow.pil_to_cv2(image)
    corner_predictions = CORNER_MODEL(image, verbose=False)
    target_df = preds_to_df(corner_predictions)
    if len(target_df) < 4:
        raise ValueError("Could not find all four image landmarks.")
    base_df = pd.read_csv("../data/Rwandan_Four_Corner_Perfect_Labels.csv")

    homography_matrix = compute_homography(base_df, target_df)
    warped_image = warp_via_homography(image, homography_matrix)

    return deshadow.cv2_to_pil(warped_image)


def get_homography_list(base_df, target_df):
    """Gets a set of source and destination points for cv2.findHomography.

    Parameters :
        target_df - the target image.

    Returns : a tuple with the source points and the destination points in that order.
    """
    merged_df = base_df.merge(
        target_df, on="label_name", how="inner", suffixes=["_base", "_target"]
    )
    source_points = []
    destination_points = []
    for _, row in merged_df.iterrows():
        destination_points.append([row.bbox_x_base, row.bbox_y_base])
        source_points.append([row.bbox_x_target, row.bbox_y_target])

    return np.array(source_points), np.array(destination_points)


def compute_homography(base_df, target_df):
    """Computes the homography matrix given two dataframes.

    Parameters :
        base_df - the dataframe to use as a destination.
        target_df - the dataframe to use as a source/target.

    Returns : a homography matrix that transforms a target image into the base.
    """
    src_pts, dst_pts = get_homography_list(base_df, target_df)
    homography, _ = findHomography(src_pts, dst_pts)
    return homography


def warp_via_homography(im_target, homography_matrix):
    """Warps an image by using the homography matrix.

    Parameters :
        im_target - the target/warped image to correct.

    Returns : an image which has linear distortions corrected via homography transformation.
    """
    new_height, new_width = (1168, 1606)
    im_warped = warpPerspective(im_target, homography_matrix, (new_width, new_height))
    return im_warped


def preds_to_df(preds):
    """Makes a dataframe from yolov8 predictions.

    Parameters :
        preds - the predictions from a yolov8 model.

    Returns : a dataframe version of the predictions.
    """
    bbox_classes = [preds[0].names[float(x)] for x in preds[0].boxes.cls]
    boxes = [[float(y) for y in x] for x in preds[0].boxes.xyxy]
    confs = [float(x) for x in preds[0].boxes.conf]

    boxes = pd.DataFrame(
        boxes, columns=["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
    )
    boxes["label_name"] = bbox_classes
    boxes["conf"] = confs

    return boxes
