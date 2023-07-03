"""The homography module predicts document landmarks, computes a homography matrix,
and returns a rectified image based on the landmarks identified."""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from glob import glob


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
    homography, _ = cv2.findHomography(src_pts, dst_pts)
    return homography


def warp_via_homography(im_base, im_target, homography_matrix):
    """Warps an image by using the homography matrix.

    Parameters :
        im_base - the control image which is not warped.
        im_target - the target/warped image to correct.

    Returns : an image which has linear distortions corrected via homography transformation.
    """
    size = im_base.shape
    im_warped = cv2.warpPerspective(im_target, homography_matrix, (size[1], size[0]))
    return im_warped
