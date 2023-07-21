"""Tiles is a module that splits images up into tiles, uses
YOLOv8 models to predict on sets of tiles, and defines
functions that automatically filter and combine tiles with
predictions, allowing other code which uses tiles to not
worry about the implementation of image tiling."""

from typing import List
from PIL import ImageDraw
from bounding_box import BoundingBox


def tile_predict(
    model,
    image,
    rows: int,
    columns: int,
    stride: float,
    overlap_tolerance: float,
    remove_non_square: bool = False,
) -> List[BoundingBox]:
    """Uses a YOLOv8 model to predict on an image using image tiling.

    Args :
        model - the YOLOv8 model to use.
        image - the PIL image to predict on.
        rows - the number of rows that the image will be cut into.
        columns - the number of columns that the image will be cut into.
        stride - the amount that the window should slide when making cuts.
        overlap_tolerance - the percentage of combined area that two boxes
                            can overlap before one is removed.
        remove_non_square - Whether or not to remove non-square predictions.

    Returns : Predictions on the whole image identical in form to what you
              would get if you just called "model(image)".
    """
    tiles = tile_image(image, rows, columns, stride)
    tiled_predictions = predict_on_tiles(model, tiles)
    width, height = image.size
    predictions = reassemble_predictions(
        tiled_predictions,
        overlap_tolerance,
        remove_non_square,
        rows,
        columns,
        width,
        height,
        stride,
    )
    return predictions


def tile_image(image, rows: int, columns: int, stride: float = 1 / 2) -> List[List]:
    """Tiles an image.

    Args :
        image - the PIL image to be tiled.
        rows - the number of rows that the image will be cut into.
        columns - the number of columns that the image will be cut into.
        stride - the amount that the window should slide when making cuts.

    Returns : A tiled image, where you can find the tile you wish by tiles[row][col].
    """
    tiles = []
    width, height = image.size
    x_coords = get_x_coords(width, columns, stride)
    y_coords = get_y_coords(height, rows, stride)
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


def get_x_coords(width: float, columns: int, stride: float):
    """Gets the x coordinates for where to crop a tile.

    Parameters:
        width - the width of the image to tile.
        columns - the number of oclumns that the image is cut into.

    Returns : The x coordinates for all the tiles.
    """
    new_number_of_columns = columns * int(1 / stride)
    return [
        int((width * i / new_number_of_columns))
        for i in range(0, new_number_of_columns)
    ] + [width]


def get_y_coords(height: float, rows: int, stride: float):
    """Gets the y coordinates for where to crop a tile.

    Parameters:
        height - the height of the image to tile.
        rows - the number of rows that the image will be cut into.

    Returns : The y coordinates for all the tiles."""
    new_number_of_rows = rows * int(1 / stride)
    return [
        int((height * i / new_number_of_rows)) for i in range(0, new_number_of_rows)
    ] + [height]


def predict_on_tiles(model, tiles) -> List[List[List[BoundingBox]]]:
    """Uses a YOLOv8 model to predict on image tiles.

    Args :
        model - the YOLOv8 model to predict on the image with.
        tiles - the tiled PIL image to predict on.

    Returns : The model's predictions on each tile.
    """
    predictions = []
    for row in tiles:
        new_preds = []
        for tile in row:
            new_pred = model(tile, verbose=False)[0].boxes.data.tolist()
            new_pred = [
                BoundingBox(
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[5],
                    b[4],
                )
                for b in new_pred
            ]
            new_preds.append(new_pred)
        predictions.append(new_preds)
    return predictions


def reassemble_predictions(
    tiled_predictions: List[List[List[BoundingBox]]],
    overlap_tolerance: float,
    remove_non_square: bool,
    rows: int,
    columns: int,
    width: int,
    height: int,
    stride: float,
) -> List[BoundingBox]:
    """Reassembles the tiled predictions into predictions on the full image.

    Args :
        tiled_predictions - the tiled predictions.
        overlap_tolerance - the percentage of combined area that two boxes
                            can overlap before one is removed.
        remove_non_square - whether or not to remove non-square predictions.
        rows - the number of rows that the image will be cut into.
        columns - the number of columns that the image will be cut into.
        width - the width of the image in pixels.
        height - the height of the image in pixels.

    Returns : Predictions whose xy coords are on the full image, and has overlapping
              predictions removed.
    """
    predictions = map_raw_detections_to_full_image(
        tiled_predictions, rows, columns, width, height, stride
    )
    if remove_non_square:
        predictions = remove_non_square_detections(predictions)
    predictions = remove_overlapping_detections(predictions, overlap_tolerance)
    return predictions


def map_raw_detections_to_full_image(
    predictions: List[List[List[BoundingBox]]],
    rows: int,
    columns: int,
    width: int,
    height: int,
    stride: float,
) -> List[BoundingBox]:
    """Maps the coordinates of the raw detections to where they are on the full image.

    Args :
        predictions - the
        rows - the number of rows that the image will be cut into.
        columns - the number of columns that the image will be cut into.
        width - the width of the image in pixels.
        height - the height of the image in pixels.
    """

    x_coords = get_x_coords(width, columns, stride)
    y_coords = get_y_coords(height, rows, stride)

    mapped_boxes = []

    for col_ix, col in enumerate(predictions):
        for row_ix, row in enumerate(col):
            boxes = row.copy()
            for box in boxes:
                mapped_boxes.append(
                    [
                        box.left + x_coords[col_ix],
                        box.top + y_coords[row_ix],
                        box.right + x_coords[col_ix],
                        box.bottom + y_coords[row_ix],
                        box.predicted_class,
                        box.confidence,
                    ]
                )

    mapped_boxes = [
        BoundingBox(mb[0], mb[1], mb[2], mb[3], mb[4], mb[5]) for mb in mapped_boxes
    ]
    return mapped_boxes


def remove_non_square_detections(
    predictions: List[BoundingBox], threshold: float = 0.5
) -> List[BoundingBox]:
    """Removes detections that aren't square enough.

    Args :
        predictions - The bounding box predictions.
        threshold - a float to determine how non-square a detection must
                    be to be removed. Lower is more strict.

    Returns : A list of predictions with non-square detections removed.
    """
    remove = []
    for index, box in enumerate(predictions):
        left, right = min(box.left, box.right), max(box.left, box.right)
        top, bottom = min(box.top, box.bottom), max(box.top, box.bottom)
        width = left - right
        height = top - bottom
        if abs((width - height) / ((height + width) / 2)) > threshold:
            remove.append(index)
    for index in sorted(remove, reverse=True):
        predictions.pop(index)
    return predictions


def remove_overlapping_detections(
    predictions: List[BoundingBox], overlap_tolerance: float, strategy: str = "iou"
) -> List[BoundingBox]:
    """Removes detections that overlap too much.

    Args :
        predictions - The bounding box predictions.
        overlap_tolerance - The IOU over which two boxes are said to be matching.

    Returns : A list of predictions where the remaining boxes do not overlap significantly.
    """
    remove = []
    rects = sorted(predictions, key=lambda x: x[4], reverse=True)  # sort by confidence.
    for this_ix, this_rect in enumerate(rects):
        for that_ix, that_rect in enumerate(rects[this_ix + 1 :]):
            if strategy == "iou":
                criteria = this_rect.intersection_over_union(that_rect)
            else:
                criteria = this_rect.intersection_over_smaller_box(that_rect)
            if criteria > overlap_tolerance:
                index_to_remove = (
                    this_ix if this_rect[4] < that_rect[4] else this_ix + that_ix + 1
                )
                remove.append(index_to_remove)
    for index in sorted(list(set(remove)), reverse=True):
        del rects[index]
    return rects


def show_detections(
    model,
    image,
    rows: int,
    columns: int,
    stride: float,
    overlap_tolerance: float,
    remove_non_square: bool = False,
):
    """Draws the detections on a PIL image and returns it."""
    preds = tile_predict(
        model,
        image,
        rows=rows,
        columns=columns,
        stride=stride,
        overlap_tolerance=overlap_tolerance,
        remove_non_square=remove_non_square,
    )
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in preds:
        draw.rectangle(box.box)
    return img
