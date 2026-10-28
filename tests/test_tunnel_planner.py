"""2-D route guidance plus present-pose, full-body checked local commands."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import CollisionGeometry, OccupancyGrid, Pose2D, RectangularFootprint
from tunnel.planner import GlobalLocalPlanner, GlobalRoute
from tunnel.tracking import LimitedCommand, turn_command


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
        self.base = CollisionGeometry(
            RectangularFootprint(.08, .10, .045, .005),
            minimum_turning_radius=.18, collision_check_step=.01)
        self.planner = GlobalLocalPlanner(self.base)

    def test_complete_route_around_wall_without_hybrid_search(self):
        grid = grid_with(((.45, -.20, .55, .20),))
        self.assertFalse(hasattr(self.base, 'plan'))
        route = self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points[0], (0., 0.))
        np.testing.assert_array_equal(route.points[-1], (2., 0.))
        self.assertGreater(route.station[-1], 2.)
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
            command = self.planner.local_command(grid, pose, route, CONTROL, .1, .2, .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .1+CONTROL['linear_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity-.2), CONTROL['angular_acceleration']*.05+1e-9)
        self.assertTrue(all(call.args[1] == pose for call in check.call_args_list))
        self.assertTrue(self.planner.command_is_collision_free(grid, pose, command, CONTROL))

    def test_latest_wall_cannot_be_ignored_by_old_route(self):
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        grid = grid_with(((.18, -1.25, .23, 1.25),))
        command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, .25, 0., .05)
        self.assertIsNone(command)

    def test_target_behind_uses_checked_rotation_instead_of_fixed_tail_retry(self):
        grid = grid_with()
        route = GlobalRoute(np.array(((0., 0.), (-.25, 0.))))
        command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, 0., 0., .05)
        self.assertIsNotNone(command)
        self.assertEqual(command.linear_velocity, 0.)
        self.assertNotEqual(command.angular_velocity, 0.)
        self.assertEqual(command.reason, 'turning toward route')
        self.assertTrue(self.planner.command_is_collision_free(grid, (0., 0., 0.), command, CONTROL))

    def test_next_command_tick_and_fixed_candidate_distance_are_separate_checks(self):
        grid = grid_with(((.21, -.1, .24, .1),))
        pose = Pose2D(0., 0., 0.)
        command = LimitedCommand(.32, 0.)
        self.assertTrue(self.base.primitive_is_collision_free(grid, pose, 0., .32*.05))
        self.assertTrue(self.planner.command_is_collision_free(grid, pose, command, CONTROL))
        self.assertFalse(self.base.primitive_is_collision_free(
            grid, pose, 0., self.planner.local_horizon))

    def test_command_check_uses_only_the_limited_twist_and_one_period(self):
        grid, pose = grid_with(), Pose2D(.1, .2, .3)
        command = LimitedCommand(.2, .4)
        # This contract needs no reaction time, braking, or measured-motion
        # settings. The actual limited twist defines the checked curve.
        with patch.object(self.base, 'primitive_is_collision_free',
                          return_value=True) as check:
            self.assertTrue(self.planner.command_is_collision_free(
                grid, pose, command, {'period': .05}))
        check.assert_called_once_with(grid, pose, 2., .2*.05)

    def test_command_check_rejects_invalid_twists_and_periods(self):
        grid, pose = grid_with(), Pose2D(0., 0., 0.)
        for command, control in (
                (LimitedCommand(-.1, 0.), {'period': .05}),
                (LimitedCommand(math.nan, 0.), {'period': .05}),
                (LimitedCommand(0., math.inf), {'period': .05}),
                (LimitedCommand(.1, 0.), {'period': 0.}),
                (LimitedCommand(.1, 0.), {'period': math.inf}),
                (LimitedCommand(.1, 0.), {}),
                (object(), {'period': .05})):
            with self.subTest(command=command, control=control):
                self.assertFalse(self.planner.command_is_collision_free(
                    grid, pose, command, control))

    def test_turn_command_has_gain_and_rate_limits_without_a_stopping_angle_cap(self):
        control = dict(maximum_angular_velocity=1.4,
                       maximum_lateral_acceleration=.30, angular_acceleration=8.)
        command = turn_command(.2, 0., 0., .05, control, 2.)
        self.assertEqual(command, LimitedCommand(0., .4))

    def test_clear_straight_reaches_configured_cruise_with_actual_acceleration_limit(self):
        control = dict(CONTROL, cruise_velocity=.4, linear_acceleration=.25,
                       safety_linear_deceleration=.15, safety_reaction_time=.10)
        grid, pose = grid_with(), Pose2D(0., 0., 0.)
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        velocity = .27
        for _ in range(12):
            command = self.planner.local_command(grid, pose, route, control,
                                                 velocity, 0., control['period'])
            self.assertIsNotNone(command)
            self.assertGreaterEqual(command.linear_velocity, velocity)
            self.assertLessEqual(command.linear_velocity-velocity,
                                 control['linear_acceleration']*control['period']+1e-12)
            self.assertAlmostEqual(command.angular_velocity, 0.)
            pose = Pose2D(pose.x+command.linear_velocity*control['period'], 0., 0.)
            velocity = command.linear_velocity
        self.assertAlmostEqual(velocity, control['cruise_velocity'])

    def test_wall_beyond_candidate_does_not_add_predictive_braking_speed_cap(self):
        control = dict(CONTROL, cruise_velocity=.4, linear_acceleration=.25,
                       safety_linear_deceleration=.15, safety_reaction_time=.10)
        # The full body fits along the 30 cm candidate and the actual next
        # command tick; the earlier velocity-based stopping sweep went farther.
        grid, pose = grid_with(((.40, -1.25, .425, 1.25),)), Pose2D(0., 0., 0.)
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        command = self.planner.local_command(grid, pose, route, control, .4, 0., .05)
        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.linear_velocity, .4)
        self.assertTrue(self.planner.command_is_collision_free(grid, pose, command, control))

    def test_geometry_horizon_and_endpoint_distance_do_not_add_speed_caps(self):
        control = dict(CONTROL, cruise_velocity=.4, safety_linear_deceleration=.15)
        route = GlobalRoute(np.array(((0., 0.), (.20, 0.))))
        for horizon in (.1, .3, .42):
            with self.subTest(horizon=horizon):
                planner = GlobalLocalPlanner(self.base, {'local_horizon': horizon})
                with patch.object(self.base, 'primitive_is_collision_free',
                                  wraps=self.base.primitive_is_collision_free) as check:
                    command = planner.local_command(grid_with(), (0., 0., 0.), route,
                                                     control, .4, 0., .05)
                self.assertIsNotNone(command)
                self.assertAlmostEqual(command.linear_velocity, .4)
                self.assertAlmostEqual(check.call_args_list[0].args[3], min(horizon, .20))
        endpoint = self.planner.local_command(grid_with(), (.2, 0., 0.), route,
                                              control, 0., 0., .05)
        self.assertEqual(endpoint.linear_velocity, 0.)

    def test_actual_deceleration_remains_limited_when_cruise_is_lowered(self):
        control = dict(CONTROL, cruise_velocity=.1, safety_linear_deceleration=.15)
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        command = self.planner.local_command(grid_with(), (0., 0., 0.), route,
                                             control, .4, 0., .05)
        self.assertIsNotNone(command)
        self.assertAlmostEqual(command.linear_velocity, .4-.15*.05)

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

    def test_forward_command_does_not_predict_a_reverse_sweep(self):
        pose = Pose2D(0., 0., 0.)
        command = LimitedCommand(.01, 0.)
        grid = grid_with(((-.25, -.15, -.20, .15),))
        self.assertTrue(self.base.pose_is_collision_free(grid, pose))
        self.assertTrue(self.planner.command_is_collision_free(grid, pose, command, CONTROL))
        self.assertFalse(self.planner.command_is_collision_free(
            grid_with(), pose, LimitedCommand(-.01, 0.), CONTROL))

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
        from tunnel.mission import load_config
        from tunnel.mission import TunnelMission
        root = Path(__file__).resolve().parents[1]
        mission = TunnelMission(load_config(root/'codes/stepbystep/tunnel.real.yaml'))
        self.addCleanup(mission.close)
        mission._initialize(anchor=(0., 0., 0.))
        grid = mission._grid()
        planner = mission.local_planner
        pose = Pose2D(.92, 0., 0.)
        route = planner.global_route(grid, pose, mission.goal)
        self.assertIsNotNone(route)
        np.testing.assert_allclose(route.points[-1], (mission.goal.x, mission.goal.y))
        command = planner.local_command(grid, pose, route, mission.control, 0., 0., .05)
        self.assertIsNotNone(command)
        self.assertTrue(planner.command_is_collision_free(grid, pose, command, mission.control))


if __name__ == '__main__':
    unittest.main()
