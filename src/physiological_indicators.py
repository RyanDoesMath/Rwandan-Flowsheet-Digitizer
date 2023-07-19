"""The physiological_indicators module extracts the data from the
physiological indicator section of the Rwandan flowsheet using YOLOv8."""

from dataclasses import dataclass
from typing import Dict
from PIL import Image
from ultralytics import YOLO
import deshadow
import tiles

SINGLE_CHAR_MODEL = YOLO("../models/single_char_pi_detector_yolov8l.pt")
PHYSIOLOGICAL_INDICATOR_TILE_DATA = {"ROWS": 4, "COLUMNS": 17, "STRIDE": 1 / 2}


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


def extract_physiological_indicators(image: Image.Image) -> Dict[str, list]:
    """Extracts the physiological indicators from a cropped image
    of the physiological indicators section of a Rwandan Flowsheet.

    Args :
        image - a PIL image of the physiological indicators section.

    Returns : A dictionary with keys for each row of the PI section,
              and a list of timestamped values for that section.
    """
    img = deshadow.deshadow_and_normalize_image(image)
    predictions = tiles.tile_predict(img)
