"""Provides the strategy for get_values_for_boxes to physiological_indicators for the
tidal volume row of the physiological indicators."""

from dataclasses import dataclass
import volume


@dataclass
class RespiratoryRate:
    """Dataclass for the respiratory rate portion of the tidal volume."""

    chars: list
    boxes: list
    respiratory_rate: int
    implausible: bool = False


@dataclass
class TidalVolumeXRespiratoryRate:
    """Dataclass for tidal volume."""

    volume: volume.Volume
    respiratory_rate: RespiratoryRate
    timestamp: int
    implausible_tidal_volume: bool = False
    implausible_respiratory_rate: bool = False
