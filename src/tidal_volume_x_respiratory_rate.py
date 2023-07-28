"""Provides the strategy for get_values_for_boxes to physiological_indicators for the
tidal volume row of the physiological indicators."""

from dataclasses import dataclass
from typing import Tuple
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


def get_limits_for_number_of_clusters(number_of_boxes: int) -> Tuple[int]:
    """Gets the upper and lower limit for the number of clusters that could be made with the
    number of boxes present.

    The tidal_volumexF section has clusters that are nearly always exactly 6, although it is
    conceivable that a respiratory rate could be set up to 35, and a volume of as low as 16ml
    for a premature baby. Thus, the lowest value is if all clusters have 5, and the highest
    is if all clusters have 6. Because this range can throw things off, the allowance here
    on either side is only 10%.

    Args :
        number_of_boxes - the number of boxes detected.

    Returns - a tuple containing the upper and lower limits of the number of clusters.
    """
    lower_lim = number_of_boxes // 6 - int(0.1 * number_of_boxes // 6)
    upper_lim = number_of_boxes // 5 + int(0.1 * number_of_boxes // 5)
    return lower_lim, upper_lim
