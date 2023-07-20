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
        self.box = [self.left, self.top, self.right, self.bottom]

    def intersection_over_union(self, other) -> float:
        """Computes the intersection over union of the two bounding boxes."""
        intersection_left, intersection_right = max((self.left, other.left)), min(
            (self.right, other.right)
        )
        intersection_top, intersection_bottom = max((self.top, other.top)), min(
            (self.bottom, other.bottom)
        )
        intersection = BoundingBox(
            intersection_left,
            intersection_top,
            intersection_right,
            intersection_bottom,
            -1,
            -1,
        )
        if intersection.width > 0 and intersection.height > 0:
            return intersection.area / (self.area + other.area - intersection.area)
        return 0
