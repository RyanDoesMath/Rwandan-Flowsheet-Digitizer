"""Defines the BoundingBox class."""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Defines and gives methods for a bounding box.

    Uses the PIL convention of having y=0 be the top of the image.
    """

    left: int
    top: int
    right: int
    bottom: int
    predicted_class: str
    confidence: float

    def __post_init__(self):
        """Inits variables that depend on the four constructor variables."""
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.x_center = self.left + (self.width / 2)
        self.y_center = self.top + (self.height / 2)
        self.area = (self.right - self.left) * (self.bottom - self.top)

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Computes the intersection over union of the two bounding boxes."""
        smaller, _ = (self, other) if self.area < other.area else (other, self)
        intersection_left, intersection_right = max((self.left, other.left)), min(
            (self.right, other.right)
        )
        intersection_top, intersection_bottom = max((self.top, other.top)), min(
            (self.bottom, other.bottom)
        )
        width, height = (
            intersection_right - intersection_left,
            intersection_bottom - intersection_top,
        )
        intersection_area = width * height
        if width > 0 and height > 0:
            return intersection_area / smaller.area
        return 0
