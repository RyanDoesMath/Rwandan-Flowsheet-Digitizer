"""The physiological_indicators module extracts the data from the
physiological indicator section of the Rwandan flowsheet using YOLOv8."""

from dataclasses import dataclass


@dataclass
class OxygenSaturation:
    """Dataclass for spo2."""

    percent: int
    timestamp: int


@dataclass
class EndTidalCarbonDioxide:
    """Dataclass for etco2."""

    percent: int
    timestamp: int


@dataclass
class FractionOfInspiredOxygen:
    """Dataclass for fio2."""

    percent: int
    timestamp: int


@dataclass
class TidalVolume:
    """Dataclass for Tidal Vol x F."""

    tidal_volume_ml: int
    respiratory_rate: int
    timestamp: int


@dataclass
class Temperature:
    """Dataclass for temp in celcius."""

    temp_c: int
    timestamp: int


@dataclass
class Diuresis:
    """Dataclass for diuresis (urine output)."""

    diuresis_ml: int
    timestamp: int


@dataclass
class BloodLoss:
    """Dataclass for estimated blood loss."""

    blood_loss_ml: int
    timestamp: int
