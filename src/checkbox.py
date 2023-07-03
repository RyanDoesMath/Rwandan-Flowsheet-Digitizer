"""A module for extracting the checkbox data from the checkbox section of a
Rwandan intraoperative Flowsheet."""

from typing import List

CHECKBOX_TILE_DATA = {"ROWS": 2, "COLUMNS": 7, "STRIDE": 1 / 2}


def remove_overlapping_detections(
    detections: List[List[float]], tolerance: float = 0.3
):
    """Removes bounding boxes that overlap more than the tolerance level.

    Parameters :
        detections - A list of detection boxes.
        tolerance - the IOU score that triggers a match between boxes.
    """
    remove = []
    for this_box_index, this_box in enumerate(detections):
        for that_box_index, that_box in enumerate(detections[this_box_index + 1 :]):
            iou = bb_intersection_over_union(this_box, that_box)
            if iou > tolerance:
                remove.append(this_box_index + that_box_index + 1)
    remove = sorted(list(set(remove)), reverse=True)
    for index in remove:
        detections.pop(index)
    return detections


def bb_intersection_over_union(box_a: List[float], box_b: List[float]):
    """Computes the bounding box intersection over union.

    Parameters:
        box_a - the first box (order doesn't matter.)
        box_b - the second box (order doesn't matter.)

    Returns : the intersection over union for the two bounding boxes.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[3], box_b[3])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[1], box_b[1])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[1] - box_a[3] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[1] - box_b[3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou


def get_x_coords(width: float):
    """Gets the x coordinates for where to crop a tile.

    Parameters:
        width - the width of the image to tile.

    Returns : The x coordinates for all the tiles.
    """
    columns = CHECKBOX_TILE_DATA["COLUMNS"]
    return [int((width * i / columns)) for i in range(0, columns)] + [width]


def get_y_coords(height: float):
    """Gets the y coordinates for where to crop a tile.

    Parameters:
        height - the height of the image to tile.

    Returns : The y coordinates for all the tiles."""
    rows = CHECKBOX_TILE_DATA["ROWS"]
    return [int((height * i / rows)) for i in range(0, rows)] + [height]


def tile_image(image):
    """Segments the image into tiles.

    Parameters:
        image - the image to tile.

    Returns: A 2d array of tiles.
    """
    stride = CHECKBOX_TILE_DATA["STRIDE"]
    tiles = []
    width, height = image.size
    x_coords = get_x_coords(width)
    y_coords = get_y_coords(height)
    for index_x in range(len(x_coords[0 : -int(1 / stride)])):
        row = []
        for index_y in range(len(y_coords[0 : -int(1 / stride)])):
            temp = image.crop(
                (
                    x_coords[index_x],
                    y_coords[index_y],
                    x_coords[index_x + int(1 / stride)],
                    y_coords[index_y + int(1 / stride)],
                )
            )
            row.append(temp)
        tiles.append(row)
    return tiles
