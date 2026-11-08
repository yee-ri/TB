"""2-D obstacle guidance and full-body clearance-scored local commands."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import CollisionGeometry, OccupancyGrid, Pose2D, RectangularFootprint
from collision_inspection import inspect_geometry
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
               safety_linear_deceleration=.40,
               safety_reaction_time=.15, safety_distance_margin=.01)


class GlobalLocalPlannerTest(unittest.TestCase):
    def setUp(self):
        self.base = CollisionGeometry(
            RectangularFootprint(.08, .10, .045, .005),
            minimum_turning_radius=.18)
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

    def test_global_route_allows_body_overlap_when_start_grid_cell_is_open(self):
        grid = grid_with(((-.1, -.04, -.08, .04),))
        start = Pose2D(0., 0., 0.)
        self.assertFalse(inspect_geometry(self.base).pose_is_collision_free(grid, start))
        route = self.planner.global_route(grid, start, (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points[0], (0., 0.))

    def test_global_route_leaves_multiple_inflated_cells_from_exact_start(self):
        grid = grid_with(((-.10, -1.25, -.075, 1.25),))
        planner = GlobalLocalPlanner(self.base, {'global_clearance': .14})
        start = (.003, .007, 0.)
        blocked, _ = planner._route_costs(grid)
        cell = grid.world_to_grid(*start[:2])
        self.assertTrue(blocked[cell[::-1]])
        self.assertFalse(grid._occupied[cell[::-1]])
        route = planner.global_route(grid, start, (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points[0], start[:2])
        np.testing.assert_array_equal(route.points[-1], (2., 0.))
        exit_index = next(i for i, point in enumerate(route.points)
                          if not blocked[grid.world_to_grid(*point)[::-1]])
        self.assertGreaterEqual(exit_index, 3)
        for first, last in zip(route.points[:exit_index], route.points[1:exit_index+1]):
            self.assertTrue(planner._line_clear(grid, grid._occupied, first, last))
        for first, last in zip(route.points[exit_index:-1], route.points[exit_index+1:]):
            self.assertTrue(planner._line_clear(grid, blocked, first, last))

    def test_occupied_start_exception_does_not_erase_obstacle_map(self):
        empty = grid_with()
        values = empty.data.copy()
        column, row = empty.world_to_grid(0., 0.)
        values[row, column] = 100
        grid = OccupancyGrid(values, empty.resolution, empty.origin_x, empty.origin_y)
        occupied_before = grid._occupied.copy()
        route = self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(grid.data, values)
        np.testing.assert_array_equal(grid._occupied, occupied_before)
        only_start_allowed = occupied_before.copy()
        only_start_allowed[row, column] = False
        for first, last in zip(route.points[:-1], route.points[1:]):
            self.assertTrue(self.planner._line_clear(grid, only_start_allowed, first, last))

    def test_global_route_still_requires_unblocked_goal_cell(self):
        for offset in (0, -2):
            with self.subTest(obstacle_offset=offset):
                empty = grid_with()
                values = empty.data.copy()
                column, row = empty.world_to_grid(2., 0.)
                values[row, column+offset] = 100
                grid = OccupancyGrid(values, empty.resolution, empty.origin_x, empty.origin_y)
                blocked, _ = self.planner._route_costs(grid)
                self.assertTrue(blocked[row, column])
                self.assertEqual(grid._occupied[row, column], offset == 0)
                self.assertIsNone(self.planner.global_route(
                    grid, (0., 0., 0.), (2., 0., 0.)))

    def test_blocked_start_does_not_open_a_real_wall_to_the_goal(self):
        grid = grid_with(((-.05, -1.25, -.025, 1.25),
                          (.3, -1.25, .35, 1.25)))
        blocked, _ = self.planner._route_costs(grid)
        self.assertTrue(blocked[grid.world_to_grid(0., 0.)[::-1]])
        self.assertIsNone(self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.)))

    def test_start_escape_cannot_cut_diagonally_between_occupied_cells(self):
        values = np.zeros((12, 12), dtype=np.int8)
        values[4:7, 4:7] = 100
        values[5, 5] = values[6, 6] = 0
        grid = OccupancyGrid(values, .1)
        start, goal = grid.grid_to_world(5, 5), grid.grid_to_world(9, 9)
        blocked, _ = self.planner._route_costs(grid)
        self.assertTrue(blocked[5, 5])
        self.assertFalse(blocked[9, 9])
        self.assertIsNone(self.planner.global_route(grid, (*start, 0.), (*goal, 0.)))

    def test_escaped_route_does_not_reenter_later_obstacle_inflation(self):
        grid = grid_with(((-.05, -1.25, -.025, 1.25),
                          (.7, -.15, .8, .15)))
        blocked, _ = self.planner._route_costs(grid)
        self.assertTrue(blocked[grid.world_to_grid(0., 0.)[::-1]])
        route = self.planner.global_route(grid, (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        exit_index = next(i for i, point in enumerate(route.points)
                          if not blocked[grid.world_to_grid(*point)[::-1]])
        self.assertGreater(np.max(np.abs(route.points[:, 1])), .20)
        for first, last in zip(route.points[exit_index:-1], route.points[exit_index+1:]):
            self.assertTrue(self.planner._line_clear(grid, blocked, first, last))

    def test_start_exception_does_not_bridge_cells_outside_keep_in(self):
        self.base.keep_in_rectangles = ((.1, -.3, 2.2, .3),)
        self.assertIsNone(self.planner.global_route(grid_with(), (0., 0., 0.), (2., 0., 0.)))

    def test_open_start_keeps_the_same_direct_route(self):
        route = self.planner.global_route(grid_with(), (0., 0., 0.), (2., 0., 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points, ((0., 0.), (2., 0.)))

    def test_goal_in_the_current_cell_needs_no_escape_route(self):
        grid = grid_with(((-.025, -.025, .025, .025),))
        route = self.planner.global_route(grid, (0., 0., 0.), (.005, .005, 0.))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route.points, ((0., 0.), (.005, .005)))

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

    def test_local_command_scores_current_pose_and_preserves_actual_rate_limits(self):
        grid = grid_with()
        route = GlobalRoute(np.array(((0., 0.), (2., .3))))
        pose = Pose2D(.6, .08, .1)
        with patch.object(self.base, 'primitive_clearances',
                          wraps=self.base.primitive_clearances) as score:
            command = self.planner.local_command(grid, pose, route, CONTROL, .1, .2, .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .1+CONTROL['linear_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity-.2), CONTROL['angular_acceleration']*.05+1e-9)
        score.assert_called()
        self.assertTrue(all(call.args[1] == pose for call in score.call_args_list))

    def test_latest_wall_changes_scores_without_rejecting_all_local_arcs(self):
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        grid = grid_with(((.18, -1.25, .23, 1.25),))
        self.assertFalse(inspect_geometry(self.base).primitive_is_collision_free(
            grid, Pose2D(0., 0., 0.), 0., self.planner.local_horizon))
        command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, .25, 0., .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)

    def test_target_behind_reaches_candidate_scoring_and_returns_a_forward_arc(self):
        grid = grid_with()
        route = GlobalRoute(np.array(((0., 0.), (-.25, 0.))))
        with patch.object(self.base, 'primitive_clearances',
                          wraps=self.base.primitive_clearances) as score:
            command = self.planner.local_command(grid, (0., 0., 0.), route, CONTROL, .32, .2, .05)
        score.assert_called()
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(command.linear_velocity, .32+CONTROL['linear_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity-.2), CONTROL['angular_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity), CONTROL['maximum_angular_velocity'])
        self.assertLessEqual(abs(command.linear_velocity*command.angular_velocity),
                             CONTROL['maximum_lateral_acceleration']+1e-9)

    def test_targets_on_both_sides_of_ninety_degrees_return_forward_motion_with_limits(self):
        for degrees in (8., 30., 60., 89.999, 90., 90.001, 100.):
            for sign in (1., -1.):
                with self.subTest(degrees=sign*degrees):
                    heading = math.radians(sign*degrees)
                    route = GlobalRoute(np.array(((0., 0.),
                        (math.cos(heading), math.sin(heading)))))
                    command = self.planner.local_command(
                        grid_with(), (0., 0., 0.), route, CONTROL, .32, 0., .05)
                    self.assertIsNotNone(command)
                    self.assertGreater(command.linear_velocity, 0.)
                    self.assertLessEqual(abs(command.angular_velocity), CONTROL['maximum_angular_velocity'])
                    self.assertLessEqual(abs(command.angular_velocity),
                                         CONTROL['angular_acceleration']*.05+1e-9)
                    self.assertLessEqual(command.linear_velocity,
                                         .32+CONTROL['linear_acceleration']*.05+1e-9)
                    self.assertLessEqual(abs(command.linear_velocity*command.angular_velocity),
                                         CONTROL['maximum_lateral_acceleration']+1e-9)

    def test_current_body_overlap_does_not_discard_the_local_candidates(self):
        grid = grid_with(((-.1, -.02, .06, .02),))
        pose = Pose2D(0., 0., 0.)
        route = GlobalRoute(np.array(((0., 0.), (2., 0.))))
        self.assertFalse(inspect_geometry(self.base).pose_is_collision_free(grid, pose))
        command = self.planner.local_command(grid, pose, route, CONTROL, .1, 0., .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)

    def test_behind_targets_keep_forward_arc_generation_across_pi_wrap(self):
        for start in (0., math.radians(170.), math.radians(-170.)):
            for sign in (1., -1.):
                with self.subTest(start=start, direction=sign):
                    heading = start+sign*math.radians(120.)
                    route = GlobalRoute(np.array(((0., 0.),
                        (math.cos(heading), math.sin(heading)))))
                    command = self.planner.local_command(
                        grid_with(), (0., 0., start), route, CONTROL, .32, 0., .05)
                    self.assertIsNotNone(command)
                    self.assertGreater(command.linear_velocity, 0.)
                    self.assertLessEqual(abs(command.angular_velocity),
                                         CONTROL['angular_acceleration']*.05+1e-9)

    def test_short_grid_connectors_do_not_reverse_the_same_corridor_heading(self):
        # Consecutive photo1 routes differed only near their grid-cell entry.
        # Their 17 mm west and 29 mm northwest connectors are not new corridors.
        pose = Pose2D(.4880335961, .3704046247, 2.1879018073)
        routes = [GlobalRoute(np.array(points)) for points in (
            ((.4870708150, .3707153368), (.47, .37), (.47, .53), (.69, .75), (1.61, .75)),
            ((.4887617473, .3680729385), (.47, .39), (.47, .53), (.69, .75), (1.61, .75)),
        )]
        headings = [self.planner.route_heading(route, pose) for route in routes]
        for heading in headings:
            self.assertGreater(heading, 0.)
            self.assertLess(heading, math.pi/2.)
        self.assertLess(abs(headings[0]-headings[1]), math.radians(8.))

    def test_heading_at_route_end_points_toward_actual_target_not_old_tangent(self):
        route = GlobalRoute(np.array(((0., 0.), (.6, 0.))))
        pose = Pose2D(.606, -.035, 0.)
        self.assertAlmostEqual(self.planner.route_heading(route, pose),
                               math.atan2(.035, -.006))




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

    def test_geometry_horizon_and_endpoint_distance_do_not_add_speed_caps(self):
        control = dict(CONTROL, cruise_velocity=.4, safety_linear_deceleration=.15)
        route = GlobalRoute(np.array(((0., 0.), (.20, 0.))))
        for horizon in (.1, .3, .42):
            with self.subTest(horizon=horizon):
                planner = GlobalLocalPlanner(self.base, {'local_horizon': horizon})
                with patch.object(self.base, 'primitive_clearances',
                                  wraps=self.base.primitive_clearances) as score:
                    command = planner.local_command(grid_with(), (0., 0., 0.), route,
                                                     control, .4, 0., .05)
                self.assertIsNotNone(command)
                self.assertAlmostEqual(command.linear_velocity, .4)
                self.assertAlmostEqual(score.call_args_list[0].args[3], min(horizon, .20))
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

    def test_moving_toward_a_behind_target_keeps_angular_ramp_and_lateral_limits(self):
        route = GlobalRoute(np.array(((0., 0.), (-.25, .01))))
        command = self.planner.local_command(grid_with(), (0., 0., 0.), route,
                                             CONTROL, .32, .8, .05)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLessEqual(abs(command.angular_velocity-.8), CONTROL['angular_acceleration']*.05+1e-9)
        self.assertLessEqual(abs(command.angular_velocity), CONTROL['maximum_angular_velocity'])
        self.assertLessEqual(abs(command.linear_velocity*command.angular_velocity),
                             CONTROL['maximum_lateral_acceleration']+1e-9)


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

    def test_surveyed_map_supplies_complete_outside_route_and_first_command(self):
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
        self.assertGreater(command.linear_velocity, 0.)


if __name__ == '__main__':
    unittest.main()
