"""Provides strategies for the respiratory rate half of the tidal volume x respiratory rate
section of the physiological indicators which relate specifically to how the volume half of that
section needs to have erroneous values dealt with, as well as which values are plausible."""

from dataclasses import dataclass

LOWEST_PLAUSIBLE_VALUE = 7
HIGHEST_PLAUSIBLE_VALUE = 40
JUMP_THRESHOLD = 8


@dataclass
class RespiratoryRate:
    """Dataclass for the respiratory rate portion of the tidal volume."""

    chars: list
    boxes: list
    respiratory_rate: int = None
    implausible: bool = False


def get_dataclass():
    """Returns the dataclass object."""
    return RespiratoryRate
