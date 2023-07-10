"""Tiles is a module that splits images up into tiles, uses
YOLOv8 models to predict on sets of tiles, and defines
functions that automatically filter and combine tiles with
predictions, allowing other code which uses tiles to not
worry about the implementation of image tiling."""

from typing import List


def tile_predict(
    model, image, rows: int, columns: int, stride: float, overlap_tolerance: float
) -> List[List[float]]:
    """Uses a YOLOv8 model to predict on an image using image tiling.

    Args :
        model - the YOLOv8 model to use.
        image - the PIL image to predict on.
        rows - the number of rows that the image will be cut into.
        columns - the number of columns that the image will be cut into.
        stride - the amount that the window should slide when making cuts.
        overlap_tolerance - the percentage of combined area that two boxes
                            can overlap before one is removed.

    Returns : Predictions on the whole image identical in form to what you
              would get if you just called "model(image)".
    """
    tiles = tile_image(image, rows, columns, stride)
    tiled_predictions = predict_on_tiles(model, tiles)
    predictions = reassemble_predictions(tiled_predictions, overlap_tolerance)
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
    x_coords = get_x_coords(width, columns)
    y_coords = get_y_coords(height, rows)
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


def get_x_coords(width: float, columns: int):
    """Gets the x coordinates for where to crop a tile.

    Parameters:
        width - the width of the image to tile.
        columns - the number of oclumns that the image is cut into.

    Returns : The x coordinates for all the tiles.
    """
    return [int((width * i / columns)) for i in range(0, columns)] + [width]


def get_y_coords(height: float, rows: int):
    """Gets the y coordinates for where to crop a tile.

    Parameters:
        height - the height of the image to tile.
        rows - the number of rows that the image will be cut into.

    Returns : The y coordinates for all the tiles."""
    return [int((height * i / rows)) for i in range(0, rows)] + [height]


def predict_on_tiles(model, tiles) -> List[List[float]]:
    """Uses a YOLOv8 model to predict on image tiles.

    Args :
        model - the YOLOv8 model to predict on the image with.
        tiles - the tiled PIL image to predict on.

    Returns : The model's predictions on each tile.
    """


def reassemble_predictions(
    tiled_predictions: List[List[float]], overlap_tolerance: float
) -> List[List[float]]:
    """Reassembles the tiled predictions into predictions on the full image.

    Args :
        tiled_predictions - the tiled predictions.

    Returns : Predictions whose xy coords are on the full image, and has overlapping
              predictions removed.
    """
