"""Provides strategies for pi_gasses to use that relate specifically to how the SpO2 section of
needs to have erroneous values dealt with, as well as which values are plausible."""

from dataclasses import dataclass
from typing import Tuple

LOWEST_PLAUSIBLE_VALUE = 75
HIGHEST_PLAUSIBLE_VALUE = 100


@dataclass
class OxygenSaturation:
    """Dataclass for spo2."""

    chars: list
    boxes: list
    percent: int
    timestamp: int
    implausible: bool = False


def get_limits_for_number_of_clusters(number_of_boxes: int) -> Tuple(int):
    """Gets the upper and lower limit for the number of clusters that could be made with the
    number of boxes present.

    Args :
        number_of_boxes - the number of boxes detected.

    Returns - a tuple containing the upper and lower limits of the number of clusters.
    """
