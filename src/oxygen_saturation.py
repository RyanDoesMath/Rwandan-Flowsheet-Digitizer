"""The oxygen_saturation module provides an implementation for
get_values_for_boxes that the physiological_indicators module can
use to get values for the bounding box detections it made."""

from typing import List
from PIL import Image
from bounding_box import BoundingBox


def get_values_for_boxes(boxes: List[BoundingBox], image: Image.Image) -> list:
    """Implements a strategy for getting the values for the spo2
    boxes from the physiological indicator section.

    Args :
        boxes - the boxes in that section.
        image - the image that the boxes come from.

    Returns : The actual values for the spo2 section in a list of objects.
    """
