"""2-D route guidance plus present-pose, full-body checked local commands."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))
from tunnel.planner import HybridAStarPlanner, OccupancyGrid, Pose2D, RectangularFootprint
from tunnel.planner_v2 import GlobalLocalPlanner, GlobalRoute
from tunnel.tracking import LimitedCommand


def grid_with(rectangles=(), unknown=False):
    values = np.zeros((100, 160), dtype=np.int8)
    for left, bottom, right, top in rectangles:
        c0, c1 = math.floor((left+.5)/.025), math.ceil((right+.5)/.025)
        r0, r1 = math.floor((bottom+1.25)/.025), math.ceil((top+1.25)/.025)
        values[max(0, r0):min(100, r1), max(0, c0):min(160, c1)] = -1 if unknown else 100
    return OccupancyGrid(values, .025, -.5, -1.25)


CONTROL = dict(period=.05, cruise_velocity=.32, maximum_angular_velocity=1.4,
               maximum_lateral_acceleration=.30, linear_acceleration=.35,
               linear_deceleration=.40, angular_acceleration=8.,
               safety_linear_deceleration=.40, safety_angular_deceleration=8.,
               safety_reaction_time=.15, safety_distance_margin=.01)


class GlobalLocalPlannerTest(unittest.TestCase):
    def setUp(self):
        self.base = HybridAStarPlanner(
            RectangularFootprint(.08, .10, .045, .005),
            minimum_turning_radius=.18, collision_check_step=.01)
        self.planner = GlobalLocalPlanner(self.base)

    def test_complete_route_around_wall_without_hybrid_search(self):
        grid = grid_with(((.45, -.20, .55, .20),))
        with patch.object(self.base, 'plan', side_effect=AssertionError('SE2 search forbidden')):
            route = self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points[0], (0., 0.))
        np.testing.assert_array_equal(route.points[-1], (2., 0.))
        self.assertGreater(route.length, 2.)
        self.assertGreater(np.max(np.abs(route.points[:, 1])), .25)
        self.assertFalse(route.points.flags.writeable)
        blocked, _ = self.planner._route_costs(grid)
        for first, last in zip(route.points[:-1], route.points[1:]):
            self.assertTrue(self.planner._line_clear(grid, blocked, first, last))

    def test_disconnected_unknown_wall_returns_no_partial_route(self):
        grid = grid_with(((.3, -1.25, .35, 1.25),), unknown=True)
        self.assertIsNone(self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.)))

    def test_simplification_preserves_soft_cost_detour(self):
        grid = grid_with()
        soft = np.zeros(grid.data.shape)
        c0, r0 = grid.world_to_grid(.6, -.20)
        c1, r1 = grid.world_to_grid(1.3, .20)
        soft[r0:r1+1, c0:c1+1] = 30.
        grid = OccupancyGrid(grid.data, grid.resolution, grid.origin_x, grid.origin_y,
                             soft_cost_data=soft)
        planner = GlobalLocalPlanner(self.base, {'global_soft_cost_weight': 10.})
        route = planner.global_route(grid, (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        self.assertGreater(np.max(np.abs(route.points[:, 1])), .20)
        _, traversal = planner._route_costs(grid)
        retained_cost = sum(planner._line_cost(grid, traversal, a, b)
                            for a, b in zip(route.points[:-1], route.points[1:]))
        direct_cost = planner._line_cost(grid, traversal, route.points[0], route.points[-1])
        self.assertLess(retained_cost, direct_cost)

    def test_current_pose_can_approach_end_of_older_guidance(self):
        # Projection onto an older route may reach its end before the robot
        # reaches it. Use actual target distance, not a zero station remainder.
        route = GlobalRoute(np.array(((0., 0.), (1., 0.))))
        command = self.planner.local_command(grid_with(), (1., .1, -math.pi/2.),
                                             route, CONTROL, 0., 0., .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)

    def test_iteration_bound_returns_no_partial_route(self):
        planner = GlobalLocalPlanner(self.base, {'global_maximum_iterations': 1})
        self.assertIsNone(planner.global_route(grid_with(), (0., 0., 0.), (2., 0., 0.)))

    def test_keep_in_gap_is_not_bridged_by_a_shortcut(self):
        self.base.keep_in_rectangles = ((-.1, -.3, .3, .3), (.5, -.3, 2.2, .3))
        self.assertIsNone(self.planner.global_route(grid_with(), (0., 0., 0.), (2., 0., 0.)))

    def test_target_moves_with_current_pose_without_fixed_prefix_endpoint(self):
        route = GlobalRoute(np.array(((0., 0.), (1., 0.), (1., 1.))))
        first = self.planner.select_target(route, (0., 0., 0.), .30)
        later = self.planner.select_target(route, (.9, .04, .2), .30)
        np.testing.assert_allclose(first[:2], (.3, 0.))
        np.testing.assert_allclose(later[:2], (1., .2))
        self.assertAlmostEqual(later[2], 1.1)
        self.assertAlmostEqual(later[3], math.pi/2.)

    def test_local_command_uses_current_pose_and_actual_rate_limits(self):
        grid = grid_with()
        route = GlobalRoute(np.array(((0., 0.), (2., .3))))
        pose = Pose2D(.6, .08, .1)
        with patch.object(self.base, 'primitive_is_collision_free',
                          wraps=self.base.primitive_is_collision_free) as check:
            command = self.planner.local_command(grid, pose, route, CONTROL, .1, .2, .05,
                                                 measured_linear=.09, measured_angular=.18)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .1+CONTROL['linear_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity-.2), CONTROL['angular_acceleration']*.05+1e-9)
        self.assertTrue(all(call.args[1] == pose for call in check.call_args_list))
        self.assertTrue(self.planner.command_is_safe(grid, pose, command, CONTROL, .09, .18))

    def test_latest_wall_cannot_be_ignored_by_old_route(self):
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        grid = grid_with(((.18, -1.25, .23, 1.25),))
        command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, .25, 0., .05,
                                             measured_linear=.25)
        self.assertIsNone(command)

    def test_target_behind_uses_checked_rotation_instead_of_fixed_tail_retry(self):
        grid = grid_with()
        route = GlobalRoute(np.array(((0., 0.), (-.25, 0.))))
        command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, 0., 0., .05)
        self.assertIsNotNone(command)
        self.assertEqual(command.linear_velocity, 0.)
        self.assertNotEqual(command.angular_velocity, 0.)
        self.assertEqual(command.reason, 'turning toward route')
        self.assertTrue(self.planner.command_is_safe(grid, (0., 0., 0.), command, CONTROL))

    def test_braking_sweep_rejects_obstacle_beyond_one_command_tick(self):
        grid = grid_with(((.21, -.1, .24, .1),))
        pose = Pose2D(0., 0., 0.)
        command = LimitedCommand(.32, 0.)
        self.assertTrue(self.base.primitive_is_collision_free(grid, pose, 0., .32*.05))
        self.assertFalse(self.planner.command_is_safe(grid, pose, command, CONTROL))

    def test_driving_to_rotation_transition_retains_lateral_limit(self):
        route = GlobalRoute(np.array(((0., 0.), (-.25, .01))))
        command = self.planner.local_command(grid_with(), (0., 0., 0.), route,
                                             CONTROL, .32, .8, .05)
        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.linear_velocity, .30)
        self.assertLessEqual(command.linear_velocity*abs(command.angular_velocity),
                             CONTROL['maximum_lateral_acceleration']+1e-12)
        self.assertLessEqual(abs(command.angular_velocity-.8),
                             CONTROL['angular_acceleration']*.05+1e-12)

    def test_negative_measured_motion_checks_rear_sweep_without_clipping(self):
        pose = Pose2D(0., 0., 0.)
        command = LimitedCommand(.01, 0.)
        self.assertTrue(self.planner.command_is_safe(grid_with(), pose, command, CONTROL,
                                                     measured_linear=-.001))
        grid = grid_with(((-.25, -.15, -.20, .15),))
        self.assertTrue(self.base.pose_is_collision_free(grid, pose))
        self.assertFalse(self.planner.command_is_safe(grid, pose, command, CONTROL,
                                                      measured_linear=-.3))
        self.assertFalse(self.planner.command_is_safe(grid_with(), pose, LimitedCommand(-.01, 0.), CONTROL))

    def test_route_has_no_terminal_heading_constraint(self):
        grid = grid_with()
        a = self.planner.global_route(grid, (0., 0., 0.), (1., 0., 0.))
        b = self.planner.global_route(grid, (0., 0., 0.), (1., 0., math.pi/2.))
        np.testing.assert_allclose(a.points, b.points)

    def test_invalid_inputs_cannot_produce_command(self):
        grid = grid_with()
        self.assertIsNone(self.planner.global_route(grid, (math.nan, 0., 0.), (2., 0., 0.)))
        with self.assertRaises(ValueError):
            GlobalLocalPlanner(self.base, {'local_steering_samples': 4})
        with self.assertRaises(ValueError):
            GlobalRoute(np.array(((0., math.inf),)))
        route = GlobalRoute(np.array(((0., 0.), (1., 0.))))
        self.assertIsNone(self.planner.local_command(grid, (0., 0., 0.), route, CONTROL,
                                                     math.nan, 0., .05))

    def test_surveyed_map_supplies_complete_outside_route_and_checked_first_command(self):
        # Static geometry only. This does not validate ROS, localization,
        # vehicle dynamics, Gazebo completion, or a real robot.
        from tunnel.mission import TunnelMission, load_config
        root = Path(__file__).resolve().parents[1]
        mission = TunnelMission(load_config(root/'codes/stepbystep/tunnel.real.yaml'))
        self.addCleanup(mission.close)
        mission._initialize(anchor=(0., 0., 0.))
        grid = mission._grid()
        planner = GlobalLocalPlanner(mission.planner)
        pose = Pose2D(.92, 0., 0.)
        route = planner.global_route(grid, pose, mission.goal)
        self.assertIsNotNone(route)
        np.testing.assert_allclose(route.points[-1], (mission.goal.x, mission.goal.y))
        command = planner.local_command(grid, pose, route, mission.control, 0., 0., .05)
        self.assertIsNotNone(command)
        self.assertTrue(planner.command_is_safe(grid, pose, command, mission.control))


if __name__ == '__main__':
    unittest.main()
