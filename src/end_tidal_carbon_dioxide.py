"""Provides strategies for pi_gasses to use that relate specifically to how the EtCO2 section
needs to have erroneous values dealt with, as well as which values are plausible."""

from dataclasses import dataclass
from typing import Tuple

LOWEST_PLAUSIBLE_VALUE = 10
HIGHEST_PLAUSIBLE_VALUE = 80
JUMP_THRESHOLD = 20


@dataclass
class EndTidalCarbonDioxide:
    """Dataclass for etco2."""

    chars: list
    boxes: list
    percent: int
    timestamp: int
    implausible: bool = False


def get_dataclass():
    """Returns the dataclass object."""
    return EndTidalCarbonDioxide


def get_limits_for_number_of_clusters(number_of_boxes: int) -> Tuple[int]:
    """Gets the upper and lower limit for the number of clusters that could be made with the
    number of boxes present.

    For end tidal co2, there can only really be two digit number plausible, so we just give a
    20% allowance for error on either side.

    Args :
        number_of_boxes - the number of boxes detected.

    Returns - a tuple containing the upper and lower limits of the number of clusters.
    """
    lower_lim = number_of_boxes // 2 - int(0.2 * number_of_boxes // 2)
    upper_lim = number_of_boxes // 2 + int(0.2 * number_of_boxes // 2)
    return lower_lim, upper_lim
