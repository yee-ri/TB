#!/usr/bin/env python3
"""Rebuild the user's 1820 mm OUTER-square, 2 mm wall practice tunnel.

Pure geometry only: does not import ROS, run a controller, or publish anything.
Keep the 20 mm runtime raster and mark every cell intersecting a wall. The
physical walls remain 2 mm; occupied cells conservatively overstate thickness.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / 'codes' / 'stepbystep' / 'maps'
RESOLUTION = 0.02
BOUNDS = [-0.5, -0.4, 2.4, 2.2]
# Local x is the initial robot forward direction; local y is its left side.
# Wheel-axis midpoint is 140 mm before the entry wall and on the portal axis.
WALLS = {
    'entry_wall': [0.140, 0.150, 0.142, 1.668],
    'far_wall': [1.958, -0.152, 1.960, 1.668],
    'right_side_wall': [0.140, -0.152, 1.960, -0.150],
    'exit_wall': [0.140, 1.666, 1.658, 1.668],
}


def build():
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
                   for wall in WALLS.values()):
                pixels[row * width + column] = 0
                occupied += 1
    image = ('P5\n# 1820mm OUTER square; 2mm walls; 300mm openings\n'
             '{} {}\n255\n'.format(width, height)).encode('ascii') + pixels
    image_path = MAPS / 'tunnel_practice_1820.pgm'
    image_path.write_bytes(image)
    geometry = {
        'source': 'User confirmation in this chat, 2026-10-09 Asia/Seoul',
        'status': 'Dimensions user-confirmed; nominal start defines the surveyed map frame. Actual activation pose is measured from LiDAR, not assumed equal to this reference.',
        'user_dimensions_m': {'outer_square': 1.820, 'wall_thickness': 0.002,
                              'entry_clear_width': 0.300, 'exit_clear_width': 0.300},
        'topology': 'Entry near the right corner; exit on the perpendicular left side near the far corner, matching the existing tunnel topology.',
        'coordinate_frame': 'Fixed survey frame: origin at nominal axle 140mm outside entry; +x perpendicular inward, +y left, yaw degrees counterclockwise. Independent of actual activation pose.',
        'outer_square_bounds': [0.140, -0.152, 1.960, 1.668],
        'inner_clear_bounds': [0.142, -0.150, 1.958, 1.666],
        'inner_clear_square_m': 1.816,
        'wall_rectangles': WALLS,
        'entry_clear_segment': [[0.140, -0.150], [0.140, 0.150]],
        'exit_clear_segment': [[1.658, 1.668], [1.958, 1.668]],
        'start_pose_local': [0.0, 0.0, 0.0],
        'placement': {'wheel_axis_before_outer_entry_wall_m': 0.140,
                      'wheel_axis_left_of_right_inner_side_wall_m': 0.150,
                      'heading': 'Perpendicular inward through the entry, aligned with its centre.'},
        'goal': {'planning_pose': [1.808, 1.488, 90.0],
                 'pose': [1.808, 1.868, 90.0]},
        'entry': {'distance': 0.920, 'obstacle_clearance': 0.300,
                  'source': 'Preserved original normal straight-entry distance and body-front obstacle gap.'},
        'keep_in_rectangles': [[-0.420, -0.150, 0.340, 0.150],
                               [0.142, -0.150, 1.958, 1.666],
                               [1.658, 1.488, 1.958, 2.068]],
        'raster': {'image': image_path.name, 'resolution': RESOLUTION,
                   'origin': BOUNDS[:2], 'quarter_turns': 0,
                   'grid_bounds': BOUNDS, 'width': width, 'height': height,
                   'occupied_cells': occupied,
                   'sha256': hashlib.sha256(image).hexdigest(),
                   'method': 'Conservative positive-area cell/wall overlap; black=occupied, 254=free. No cylinders or full-course obstacles are baked in.',
                   'limit': '20 mm cells enlarge 2 mm walls and may narrow each opening; exact dimensions are the continuous rectangles above.'},
        'not_verified': ['Actual robot placement', 'On-site competition dimensions',
                         'Dynamic obstacles', 'Physical odometry scale and braking',
                         'Hardware stop switch operation'],
    }
    output = MAPS / 'tunnel_practice_1820.geometry.json'
    output.write_text(json.dumps(geometry, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
    print(str(image_path))
    print(str(output))
    return geometry


if __name__ == '__main__':
    build()
