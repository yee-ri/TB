"""ROS-independent port regressions: python3 -m unittest discover -s tests.

Geometry fixtures adapt the original project's tunnel planner/costmap tests;
they validate the ported algorithms, not physical dimensions or course driving.
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes/stepbystep"))

from tunnel.costmap import TunnelCostmap
from tunnel.planner import HybridAStarPlanner, OccupancyGrid, Pose2D, RectangularFootprint
from tunnel.tracking import (
    TrackingPath, build_speed_profile, calculate_tracking,
    limit_tracking_command, nearest_path_index,
)


def grid_with(rectangles=()):
    values = np.zeros((200, 200), dtype=np.int8)
    grid = OccupancyGrid(values, .01, -1., -1.)
    for x0, y0, x1, y1 in rectangles:
        c0, r0 = grid.world_to_grid(x0, y0)
        c1, r1 = grid.world_to_grid(x1, y1)
        values[max(r0, 0):min(r1 + 1, 200), max(c0, 0):min(c1 + 1, 200)] = 100
    return OccupancyGrid(values, .01, -1., -1.)


class PlannerPortTest(unittest.TestCase):
    def setUp(self):
        self.planner = HybridAStarPlanner(
            RectangularFootprint(.08, .12, .04, .01),
            minimum_turning_radius=.20, collision_check_step=.01,
        )

    def test_straight_plan_reaches_goal_with_zero_curvature(self):
        path = self.planner.plan(grid_with(), (0., 0., 0.), (.6, 0., 0.))
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path.y, 0., atol=1e-12)
        np.testing.assert_allclose(path.curvature, 0., atol=1e-12)
        self.assertAlmostEqual(path.length, .6)

    def test_between_sample_collision_blocks_straight_sweep(self):
        grid = grid_with(((.25, -.01, .26, .01),))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (.5, 0., 0.)))
        self.assertFalse(self.planner.primitive_is_collision_free(grid, (0., 0., 0.), 0., .5))

    def test_asymmetric_rear_changes_collision_when_rotated(self):
        grid = grid_with(((-.12, -.005, -.11, .005),))
        self.assertFalse(self.planner.pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertTrue(self.planner.pose_is_collision_free(grid, (0., 0., math.pi)))

    def test_entry_selects_farthest_observed_stop_with_turning_room(self):
        distance = self.planner.select_entry_stop(
            grid_with(), Pose2D(0., 0., 0.), .20, .53, .10, math.pi / 2.,
        )
        self.assertEqual(distance, .53)

    def test_entry_cannot_skip_blocked_interval(self):
        distance = self.planner.select_entry_stop(
            grid_with(((.32, -.3, .33, .3),)), Pose2D(0., 0., 0.),
            .20, .70, .10, math.pi / 2.,
        )
        self.assertIsNone(distance)

    def test_unknown_region_blocks_entry(self):
        values = np.zeros((200, 200), dtype=np.int8)
        values[:, 124:] = -1
        distance = self.planner.select_entry_stop(
            OccupancyGrid(values, .01, -1., -1.), Pose2D(0., 0., 0.),
            .20, .60, .10, math.pi / 2.,
        )
        self.assertIsNone(distance)

    def test_central_obstacle_is_avoided_and_exit_reached(self):
        values = np.zeros((50, 80), dtype=np.int8)
        values[21:30, 32:40] = 100
        grid = OccupancyGrid(values, .05, -.5, -1.25)
        planner = HybridAStarPlanner(
            RectangularFootprint(.16, .10, .09, .01), heading_bins=48,
            minimum_turning_radius=.4, primitive_step=.1, steering_samples=3,
            goal_position_tolerance=.12, goal_heading_tolerance=math.radians(9),
            goal_curvature_tolerance=1e-9, heading_heuristic_weight=.30,
            collision_check_step=.04, path_sample_step=.04,
            state_xy_resolution=.1, maximum_iterations=50000,
        )
        path = planner.plan(grid, (0., 0., 0.), (2.6, 0., 0.))
        self.assertIsNotNone(path)
        self.assertGreater(np.max(np.abs(path.y)), .30)
        self.assertTrue(all(planner.pose_is_collision_free(grid, pose) for pose in path.poses))
        self.assertLess(math.hypot(path.x[-1] - 2.6, path.y[-1]), .12)


class CostmapPortTest(unittest.TestCase):
    def make_grid(self, **kwargs):
        return TunnelCostmap([0] * 21, 7, 3, 1., 0., 0., **kwargs)

    def scan(self, grid, ranges):
        return grid.update_scan(ranges, 0., .1, .1, 6., (.5, 1.5, 0.))

    def test_hit_marks_endpoint_and_ray_clears_it(self):
        grid = self.make_grid()
        self.scan(grid, [3.])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))
        self.assertFalse(grid.is_dynamic_occupied(2, 1))
        self.scan(grid, [math.inf])
        self.assertFalse(grid.is_dynamic_occupied(3, 1))

    def test_clearing_does_not_remove_static_wall(self):
        values = [0] * 21
        values[10] = 100
        grid = TunnelCostmap(values, 7, 3, 1., 0., 0.)
        self.scan(grid, [math.inf])
        self.assertEqual(grid.to_occupancy_data()[10], 100)

    def test_static_free_is_not_observed_free(self):
        grid = self.make_grid()
        self.assertEqual(grid.to_observed_collision_occupancy_data()[3], [-1] * 21)
        self.scan(grid, [3.])
        observed = grid.to_observed_collision_occupancy_data()[3]
        self.assertEqual(observed[8], 0)
        self.assertEqual(observed[10], 100)
        self.assertEqual(observed[20], -1)

    def test_invalid_beam_does_not_clear_unknown_space(self):
        grid = self.make_grid()
        self.scan(grid, [math.nan, -math.inf, 0.])
        self.assertEqual(grid.observed_free_mask, bytes(21))

    def test_update_copy_does_not_mutate_tracking_snapshot(self):
        grid = self.make_grid()
        self.scan(grid, [3.])
        updated = grid.copy_for_update()
        self.scan(updated, [math.inf])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))
        self.assertFalse(updated.is_dynamic_occupied(3, 1))


class TrackingPortTest(unittest.TestCase):
    def path(self):
        station = np.linspace(0., 1., 51)
        return TrackingPath(station, station * 0., station * 0., station * 0.,
                            station.copy(), np.full(station.size, .2))

    def test_nearest_index_never_rewinds_and_limits_search(self):
        path = self.path()
        self.assertEqual(nearest_path_index(path, .1, 0., 10), 10)
        self.assertLessEqual(nearest_path_index(path, 1., 0., 0), 15)

    def test_straight_tracking_and_lateral_correction(self):
        path = self.path()
        centred = calculate_tracking(path, .2, 0., 0., 0, .1, 1., .5, .2)
        offset = calculate_tracking(path, .2, .05, 0., 0, .1, 1., .5, .2)
        self.assertAlmostEqual(centred.angular_velocity, 0.)
        self.assertLess(offset.angular_velocity, 0.)
        self.assertGreater(centred.target_index, centred.path_index)

    def test_profile_slows_before_high_curvature(self):
        station = np.linspace(0., 2., 101)
        curve = station * 0.
        curve[50:70] = 4.
        speed = build_speed_profile(station, curve, .5, .01, .5, .5, 1., .3, .3, .3)
        self.assertLessEqual(speed[60] * 4., 1. + 1e-9)
        self.assertLess(speed[49], .5)
        self.assertTrue(np.all(np.diff(speed ** 2) <= 2. * .3 * np.diff(station) + 1e-9))
        self.assertTrue(np.all(-np.diff(speed ** 2) <= 2. * .3 * np.diff(station) + 1e-9))

    def test_limiter_keeps_curvature_during_acceleration(self):
        command = limit_tracking_command(.3, .3, .6, 0., 0., .1, .4, .5, 1., 1., .5)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .04 + 1e-9)
        self.assertLessEqual(command.angular_velocity, .1 + 1e-9)
        self.assertAlmostEqual(command.angular_velocity / command.linear_velocity, 2.)


if __name__ == "__main__":
    unittest.main()
