"""The deshadow module provides a function for deshadowing and normalizing and image."""

from cv2 import dilate, medianBlur, absdiff, normalize, NORM_MINMAX, CV_8UC1
import numpy as np


def deshadow_and_normalize_image(image: np.ndarray):
    """Removes shadows from an image and normalizes it.

    Args:
        image:np.ndarray - a cv2 image.

    Returns: A deshadowed, normalized image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image is not a cv2 image.")
    dilated_img = dilate(image, np.ones((7, 7), np.uint8))
    medblur_img = medianBlur(dilated_img, 21)
    diff_img = 255 - absdiff(image, medblur_img)
    norm_img = diff_img.copy()  # Needed for 3.x compatibility
    norm_img = normalize(
        diff_img,
        norm_img,
        alpha=0,
        beta=255,
        norm_type=NORM_MINMAX,
        dtype=CV_8UC1,
    )
    return norm_img
