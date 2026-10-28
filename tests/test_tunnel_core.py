"""Grid observations, swept geometry and velocity-limit regressions."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))

from tunnel.costmap import TunnelCostmap
from tunnel.geometry import CollisionGeometry, OccupancyGrid, RectangularFootprint
from tunnel.tracking import limit_tracking_command


def grid_with(rectangles=()):
    values = np.zeros((200, 200), dtype=np.int8)
    grid = OccupancyGrid(values, .01, -1., -1.)
    for x0, y0, x1, y1 in rectangles:
        c0, r0 = grid.world_to_grid(x0, y0)
        c1, r1 = grid.world_to_grid(x1, y1)
        values[max(r0, 0):min(r1 + 1, 200), max(c0, 0):min(c1 + 1, 200)] = 100
    return OccupancyGrid(values, .01, -1., -1.)


class CollisionGeometryTest(unittest.TestCase):
    def setUp(self):
        self.planner = CollisionGeometry(
            RectangularFootprint(.08, .12, .04, .01),
            minimum_turning_radius=.20, collision_check_step=.01,
        )


    def test_between_sample_collision_blocks_straight_sweep(self):
        grid = grid_with(((.25, -.01, .26, .01),))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (.5, 0., 0.)))
        self.assertFalse(self.planner.primitive_is_collision_free(grid, (0., 0., 0.), 0., .5))

    def test_asymmetric_rear_changes_collision_when_rotated(self):
        grid = grid_with(((-.12, -.005, -.11, .005),))
        self.assertFalse(self.planner.pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (0., 0., math.pi)))


class CostmapTest(unittest.TestCase):
    def make_grid(self, **kwargs):
        return TunnelCostmap([0] * 21, 7, 3, 1., 0., 0., **kwargs)

    def scan(self, grid, ranges):
        return grid.update_scan(ranges, 0., .1, .1, 6., (.5, 1.5, 0.))

    def test_hit_marks_endpoint_and_ray_clears_it(self):
        grid = self.make_grid()
        self.scan(grid, [3.])
        self.assertTrue(grid.to_occupancy_data()[10] == 100)
        self.assertFalse(grid.to_occupancy_data()[9] == 100)
        self.scan(grid, [math.inf])
        self.assertFalse(grid.to_occupancy_data()[10] == 100)

    def test_clearing_does_not_remove_static_wall(self):
        values = [0] * 21
        values[10] = 100
        grid = TunnelCostmap(values, 7, 3, 1., 0., 0.)
        self.scan(grid, [math.inf])
        self.assertEqual(grid.to_occupancy_data()[10], 100)


    def test_invalid_beam_does_not_clear_existing_obstacle(self):
        grid = self.make_grid()
        self.scan(grid, [3.])
        before = grid.to_occupancy_data()
        self.scan(grid, [math.nan, -math.inf, 0.])
        self.assertEqual(grid.to_occupancy_data(), before)
        self.assertEqual(before[10], 100)


class CommandLimitsTest(unittest.TestCase):


    def test_limiter_keeps_curvature_during_acceleration(self):
        command = limit_tracking_command(.3, .3, .6, 0., 0., .1, .4, .5, 1., 1., .5)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .04 + 1e-9)
        self.assertLessEqual(command.angular_velocity, .1 + 1e-9)
        self.assertAlmostEqual(command.angular_velocity / command.linear_velocity, 2.)


if __name__ == "__main__":
    unittest.main()
