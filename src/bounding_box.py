"""Defines the BoundingBox class."""


class BoundingBox:
    """Defines and gives methods for a bounding box.

    Uses the PIL convention of having y=0 be the top of the image.
    """

    def __init__(self, left: int, top: int, right: int, bottom: int):
        """Inits the BoundingBox."""
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = right - left
        self.height = bottom - top
        self.x_center = self.compute_x_center()
        self.y_center = self.compute_y_center()

    def compute_x_center(self) -> float:
        """Computes the x center of the box."""
        return self.left + self.width / 2

    def compute_y_center(self) -> float:
        """Computes the y center of the box."""
        return self.top + self.height / 2
