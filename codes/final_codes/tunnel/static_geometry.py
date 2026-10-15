"""Use surveyed wall surfaces only when they reproduce the loaded wall raster."""

import hashlib
import json
from pathlib import Path

import numpy as np


def refined_static_cells(static_map, resolution, static_data, occupied_threshold,
                         crop=None):
    """Clip surveyed walls to occupied cells; reject stale or incomplete surveys.

    The coarse mask remains the spatial index for mapping and search. Only its
    collision rectangles become precise. Maps without a survey stay coarse.
    """
    if not static_map or not static_map.get('geometry'):
        return {}
    with Path(static_map['geometry']).open(encoding='utf-8') as stream:
        survey = json.load(stream)
    raster = survey['raster']
    values = np.asarray(static_data)
    height, width = values.shape
    origin = np.asarray(static_map['origin'], dtype=float)
    if (raster['width'] != width or raster['height'] != height
            or raster['quarter_turns'] != static_map['quarter_turns']
            or not np.isclose(raster['resolution'], resolution, rtol=0., atol=1e-10)
            or np.asarray(raster['origin']).shape != (2,)
            or not np.allclose(raster['origin'], origin, rtol=0., atol=1e-10)):
        raise ValueError('static wall geometry raster metadata does not match the loaded map')
    digest = hashlib.sha256(Path(static_map['image']).read_bytes()).hexdigest()
    if raster['sha256'] != digest:
        raise ValueError('static wall geometry image hash does not match the loaded map')
    walls = survey['wall_rectangles']
    if not isinstance(walls, dict) or not walls:
        raise ValueError('static wall geometry needs named nonempty wall rectangles')
    boxes = np.asarray(list(walls.values()), dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4
            or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('static wall rectangles must have finite positive dimensions')
    upper = origin + np.asarray((width, height)) * resolution
    if np.any(boxes[:, :2] < origin - 1e-10) or np.any(boxes[:, 2:] > upper + 1e-10):
        raise ValueError('static wall rectangles must lie inside the loaded map')

    # Match build_practice_map's positive-area overlap rule. Do not turn
    # floating point boundary slivers into tiny collision sampling steps.
    cell_x = origin[0] + np.arange(width) * resolution
    cell_y = origin[1] + np.arange(height) * resolution
    refined = {}
    covered = np.zeros(values.shape, dtype=bool)
    for xmin, ymin, xmax, ymax in boxes:
        columns = np.flatnonzero((cell_x < xmax - 1e-10)
                                 & (cell_x + resolution > xmin + 1e-10))
        rows = np.flatnonzero((cell_y < ymax - 1e-10)
                              & (cell_y + resolution > ymin + 1e-10))
        for row in rows:
            for column in columns:
                rectangle = (max(xmin, cell_x[column]), max(ymin, cell_y[row]),
                             min(xmax, cell_x[column] + resolution),
                             min(ymax, cell_y[row] + resolution))
                refined.setdefault(int(row * width + column), []).append(rectangle)
                covered[row, column] = True
    occupied = values if values.dtype == np.bool_ else values >= occupied_threshold
    if np.any(values < 0) or not np.array_equal(covered, occupied):
        raise ValueError('static wall geometry does not reproduce all loaded occupied cells')
    if crop is None:
        return refined
    # TunnelCostmap can crop the source raster to the planning window. Keep
    # physical coordinates, but translate flat indices into that cropped grid.
    column_start, row_start, crop_width, crop_height = crop
    result = {}
    for index, rectangles in refined.items():
        row, column = divmod(index, width)
        row, column = row - row_start, column - column_start
        if 0 <= row < crop_height and 0 <= column < crop_width:
            result[row * crop_width + column] = rectangles
    return result
