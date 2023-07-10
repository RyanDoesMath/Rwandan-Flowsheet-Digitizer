"""Tiles is a module that splits images up into tiles, uses
YOLOv8 models to predict on sets of tiles, and defines
functions that automatically filter and combine tiles with
predictions, allowing other code which uses tiles to not
worry about the implementation of image tiling."""
