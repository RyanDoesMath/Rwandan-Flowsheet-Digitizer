"""Provides the strategy for get_values_for_boxes to physiological_indicators for the
tidal volume row of the physiological indicators."""

from dataclasses import dataclass
import volume
import respiratory_rate


@dataclass
class TidalVolumeXRespiratoryRate:
    """Dataclass for tidal volume."""

    volume: volume.Volume
    respiratory_rate: respiratory_rate.RespiratoryRate
    timestamp: int
    implausible_tidal_volume: bool = False
    implausible_respiratory_rate: bool = False
