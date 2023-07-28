"""Provides strategies for the volume half of the tidal volume x respiratory rate section of the
physiological indicators which relate specifically to how the volume half of that section needs to
have erroneous values dealt with, as well as which values are plausible."""

from dataclasses import dataclass


@dataclass
class Volume:
    """Dataclass for the volume portion of the tidal volume."""

    chars: list
    boxes: list
    volume: int
    implausible: bool = False
