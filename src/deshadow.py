"""The deshadow module provides a function for deshadowing and normalizing and image."""

from cv2 import (
    dilate,
    medianBlur,
    absdiff,
    normalize,
    cvtColor,
    bitwise_not,
    fastNlMeansDenoising,
    NORM_MINMAX,
    CV_8UC1,
    COLOR_BGR2RGB,
)
from PIL import Image
import numpy as np


def deshadow_and_normalize_image(image):
    """Removes shadows from an image and normalizes it.

    Args:
        image - a pil image.

    Returns: A deshadowed, normalized pil image.
    """
    image = pil_to_cv2(image)
    dilated_img = dilate(image, np.ones((7, 7), np.uint8))
    medblur_img = medianBlur(dilated_img, 21)
    diff_img = absdiff(image, medblur_img)
    norm_img = diff_img.copy()  # Needed for 3.x compatibility
    norm_img = normalize(
        diff_img,
        norm_img,
        alpha=0,
        beta=255,
        norm_type=NORM_MINMAX,
        dtype=CV_8UC1,
    )
    norm_img = cv2_to_pil(norm_img)
    return norm_img


def denoise_image(image):
    """Denoises a PIL image using the cv2 function fastNLMeansDenoising.

    Args :
        image - a PIL image.

    Returns : a Denoised PIL image.
    """
    img = image.copy()
    img = pil_to_cv2(img)
    img = fastNlMeansDenoising(img, h=75, templateWindowSize=7, searchWindowSize=21)
    return img


def cv2_to_pil(cv2_image):
    """Converts a cv2 image to a PIL image."""
    color_converted = cvtColor(bitwise_not(cv2_image), COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    return pil_image


def pil_to_cv2(pil_img):
    """Converts a PIL image to a cv2 image."""
    open_cv_image = np.array(pil_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image
