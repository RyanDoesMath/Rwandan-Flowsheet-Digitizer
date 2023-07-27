"""Provides strategies for pi_gasses to use that relate specifically to how the SpO2 section of
needs to have erroneous values dealt with, as well as which values are plausible."""

from dataclasses import dataclass


@dataclass
class OxygenSaturation:
    """Dataclass for spo2."""

    chars: list
    boxes: list
    percent: int
    timestamp: int
    implausible: bool = False
