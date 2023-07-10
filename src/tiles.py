"""Tiles is a module that splits images up into tiles, uses
YOLOv8 models to predict on sets of tiles, and defines
functions that automatically filter and combine tiles with
predictions, allowing other code which uses tiles to not
worry about the implementation of image tiling."""

from typing import List


def tile_predict(
    model, image, rows: int, columns: int, stride: float = 1 / 2
) -> List[List[float]]:
    """Uses a YOLOv8 model to predict on an image using image tiling."""
    tiles = tile_image(image, rows, columns, stride)
    tiled_predictions = predict_on_tiles(model, tiles)
    predictions = reassemble_predictions(tiled_predictions)
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


def predict_on_tiles(model, tiles) -> List[List[float]]:
    """Uses a YOLOv8 model to predict on image tiles.

    Args :
        model - the YOLOv8 model to predict on the image with.
        tiles - the tiled PIL image to predict on.

    Returns : The model's predictions on each tile.
    """


def reassemble_predictions(tiled_predictions: List[List[float]]) -> List[List[float]]:
    """Reassembles the tiled predictions into predictions on the full image.

    Args :
        tiled_predictions - the tiled predictions.

    Returns : Predictions whose xy coords are on the full image, and has overlapping
              predictions removed.
    """
