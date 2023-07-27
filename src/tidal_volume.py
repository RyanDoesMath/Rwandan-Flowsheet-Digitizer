"""Provides the strategy for get_values_for_boxes to physiological_indicators for the
tidal volume row of the physiological indicators."""

from dataclasses import dataclass


@dataclass
class TidalVolume:
    """Dataclass for tidal volume."""

    chars: list
    boxes: list
    respiratory_rate: int
    tidal_volume: int
    timestamp: int
    implausible: bool = False
