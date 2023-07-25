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
    predicted_class: float
    confidence: float

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
        width = intersection.get_width()
        height = intersection.get_height()
        area = intersection.get_area()
        if width > 0 and height > 0:
            return area / (self.get_area() + other.get_area() - area)
        return 0

    def intersection_over_smaller_box(self, other) -> float:
        """Computes the intersection over the area of the smaller box of the two
        bounding boxes."""
        smaller = self if self.get_area() < other.get_area() else other
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
        width = intersection.get_width()
        height = intersection.get_height()
        area = intersection.get_area()
        if width > 0 and height > 0:
            return area / smaller.get_area()
        return 0

    def get_box(self):
        """Gets the 4 coordinate box for this BoundingBox."""
        return [self.left, self.top, self.right, self.bottom]

    def get_x_center(self):
        """Gets the x center for the box."""
        return self.left + (self.get_width() / 2)

    def get_y_center(self):
        """Gets the y center for the box."""
        return self.top + (self.get_height() / 2)

    def get_width(self):
        """Gets the width for the box."""
        return self.right - self.left

    def get_height(self):
        """Gets the height for the box."""
        return self.bottom - self.top

    def get_area(self):
        """Gets the area for the box."""
        return (self.get_width()) * (self.get_height())
