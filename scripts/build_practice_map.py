#!/usr/bin/env python3
"""Build the surveyed tunnel raster and exact wall geometry without ROS."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / 'codes' / 'stepbystep' / 'maps'
RESOLUTION = 0.02
BOUNDS = [-0.5, -0.4, 2.4, 2.2]
MAP_NAME = 'tunnel_1800x1820'
OUTER_WIDTH_MM = 1800  # Left/right when looking into the entrance: map y.
OUTER_DEPTH_MM = 1820  # Forward from the entrance: map x.
WALL_MM = 10
ENTRY_WIDTH_MM = 300
EXIT_WIDTH_MM = 300


def surveyed_geometry():
    # The origin stays on the entrance centreline, 140 mm outside its wall.
    x0, y0 = 140, -ENTRY_WIDTH_MM / 2 - WALL_MM
    x1, y1 = x0 + OUTER_DEPTH_MM, y0 + OUTER_WIDTH_MM
    ix0, iy0, ix1, iy1 = x0 + WALL_MM, y0 + WALL_MM, x1 - WALL_MM, y1 - WALL_MM
    exit_left = ix1 - EXIT_WIDTH_MM

    def metres(values):
        return [value / 1000 for value in values]

    return {
        'source': 'User-confirmed outer dimensions, 2026-11-13 Asia/Seoul',
        'user_dimensions_m': {
            'outer_width': OUTER_WIDTH_MM / 1000,
            'outer_depth': OUTER_DEPTH_MM / 1000,
            'wall_thickness': WALL_MM / 1000,
            'entry_clear_width': ENTRY_WIDTH_MM / 1000,
            'exit_clear_width': EXIT_WIDTH_MM / 1000,
        },
        'coordinate_frame': 'Origin 140mm outside entrance centre; +x inward, +y left.',
        'outer_bounds': metres([x0, y0, x1, y1]),
        'inner_clear_bounds': metres([ix0, iy0, ix1, iy1]),
        'inner_clear_size_m': {
            'width': (iy1 - iy0) / 1000,
            'depth': (ix1 - ix0) / 1000,
        },
        'wall_rectangles': {
            'entry_wall': metres([x0, ENTRY_WIDTH_MM / 2, ix0, y1]),
            'far_wall': metres([ix1, y0, x1, y1]),
            'right_side_wall': metres([x0, y0, x1, iy0]),
            'exit_wall': metres([x0, iy1, exit_left, y1]),
        },
        'entry_clear_segment': [metres([x0, iy0]), metres([x0, ENTRY_WIDTH_MM / 2])],
        'exit_clear_segment': [metres([exit_left, y1]), metres([ix1, y1])],
        'goal': {'pose': metres([(exit_left + ix1) / 2, y1 + 200]) + [90.0]},
        'keep_in_rectangles': [
            metres([x0 - 560, iy0, x0 + 200, ENTRY_WIDTH_MM / 2]),
            metres([ix0, iy0, ix1, iy1]),
            metres([exit_left, y1 - 180, ix1, y1 + 400]),
        ],
    }


def build(output_dir=MAPS):
    geometry = surveyed_geometry()
    walls = geometry['wall_rectangles']
    width = round((BOUNDS[2] - BOUNDS[0]) / RESOLUTION)
    height = round((BOUNDS[3] - BOUNDS[1]) / RESOLUTION)
    pixels = bytearray([254] * (width * height))
    occupied = 0
    for row in range(height):
        # PGM starts at the top; runtime np.flipud restores Cartesian y.
        ymin = BOUNDS[1] + (height - 1 - row) * RESOLUTION
        ymax = ymin + RESOLUTION
        for column in range(width):
            xmin = BOUNDS[0] + column * RESOLUTION
            xmax = xmin + RESOLUTION
            if any(min(xmax, wall[2]) - max(xmin, wall[0]) > 1e-10
                   and min(ymax, wall[3]) - max(ymin, wall[1]) > 1e-10
                   for wall in walls.values()):
                pixels[row * width + column] = 0
                occupied += 1
    image = ('P5\n# 1800mm width x 1820mm depth OUTER; 10mm walls; 300mm openings\n'
             '{} {}\n255\n'.format(width, height)).encode('ascii') + pixels
    image_path = Path(output_dir) / (MAP_NAME + '.pgm')
    image_path.write_bytes(image)
    geometry['raster'] = {'image': image_path.name, 'resolution': RESOLUTION,
                   'origin': BOUNDS[:2], 'quarter_turns': 0,
                   'grid_bounds': BOUNDS, 'width': width, 'height': height,
                   'occupied_cells': occupied,
                   'sha256': hashlib.sha256(image).hexdigest(),
                   'method': 'Conservative positive-area cell/wall overlap; black=occupied, 254=free. No cylinders or full-course obstacles are baked in.',
                   'limit': 'Raster cells index walls; exact wall rectangles provide physical boundaries.',
    }
    output = Path(output_dir) / (MAP_NAME + '.geometry.json')
    output.write_text(json.dumps(geometry, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(str(image_path))
    print(str(output))
    return geometry


if __name__ == '__main__':
    build()
