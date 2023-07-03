"""The deshadow module provides a function for deshadowing and normalizing and image."""

import cv2
import numpy as np


def deshadow_and_normalize_image(image: np.ndarray):
    """Removes shadows from an image and normalizes it.

    Args:
        filepath:str - the filepath to the image.

    Returns: A deshadowed, normalized image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image is not a cv2 image.")
    dilated_img = cv2.dilate(im, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(im, bg_img)
    norm_img = diff_img.copy()  # Needed for 3.x compatibility
    norm_img = cv2.normalize(
        diff_img,
        norm_img,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )
    return norm_img
