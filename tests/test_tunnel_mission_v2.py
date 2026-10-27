"""ROS-free global-guidance/local-command integration, not driving evidence.

Worker timing is controlled; geometry and local command generation stay real.
"""
from pathlib import Path
import importlib
import math
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))
from tunnel.mission_v2 import TunnelMissionV2
from tunnel.planner import OccupancyGrid, Pose2D
from tunnel.planner_v2 import GlobalRoute
from test_tunnel_mission import configuration, integrate, wall_returns


class GlobalLocalMissionTest(unittest.TestCase):
    def make_mission(self, goal=(2.5, 0., 0.), velocity=.12, exit_crossing=None):
        config = configuration(goal=goal)
        if exit_crossing is not None:
            config['goal']['exit_crossing'] = exit_crossing
            config['footprint']['padding'] = 0.  # Surveyed physical outline.
        config['grid']['bounds'] = [-.5, -.75, 3.5, 1.25]
        config['control'].update(cruise_velocity=.2, entry_velocity=.08,
                                 linear_acceleration=.4, linear_deceleration=.4,
                                 safety_linear_deceleration=.4)
        mission = TunnelMissionV2(config)
        self.addCleanup(mission.close)
        self.feed(mission, 10., (0., 0., 0.), 0.)
        mission._initialize((0., 0., 0.))
        mission.global_path = self.route((0., 0.), goal[:2])
        mission.state = 'FOLLOWING'
        mission.last_linear, mission.last_angular = velocity, 0.
        mission.last_time = mission.last_plan_attempt = 10.
        # Real worker lifetime is checked with Events in dedicated tests.
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        return mission

    @staticmethod
    def route(start=(0., 0.), goal=(2.5, 0.)):
        return GlobalRoute(np.array([start, goal], dtype=float))

    @staticmethod
    def feed(mission, now, pose=(.2, 0., 0.), velocity=.12, angular=0.):
        mission.update_odometry(pose, velocity, angular, now)
        mission.update_cloud(wall_returns(pose, bounds=(-.4, -.65, 3.4, 1.1)),
                             (-.033073, 0., 0.), pose, now)

    @staticmethod
    def wall_grid(x):
        values = np.zeros((200, 400), dtype=np.int8)
        values[:, int((x+.5)/.01)] = 100
        return OccupancyGrid(values, .01, -.5, -.75)

    def test_replanning_keeps_safe_local_commands_without_measured_rest(self):
        mission = self.make_mission()
        route = mission.global_path
        pose, velocity = (.2, 0., 0.), .12
        for tick in range(1, 5):
            now = 10.+tick*.05
            self.feed(mission, now, pose, velocity)
            linear, angular, complete = mission.step(now)
            self.assertGreater(linear, 0., mission.reason)
            self.assertFalse(complete)
            self.assertIs(mission.global_path, route)
            self.assertLessEqual(abs(linear-velocity), .4*.05+1e-9)
            pose = integrate(pose, linear, angular, .05)
            velocity = linear

    def test_failed_global_update_retains_guidance_and_runs_local_controller(self):
        mission = self.make_mission()
        route = mission.global_path
        mission.pending_route = (None, 'no global route', Pose2D(.1, 0., 0.), .01)
        self.feed(mission, 10.05)
        with patch.object(mission.local_planner, 'local_command',
                          wraps=mission.local_planner.local_command) as local:
            linear, _, complete = mission.step(10.05)
        self.assertIs(mission.global_path, route)
        self.assertGreater(linear, 0., mission.reason)
        self.assertFalse(complete)
        self.assertEqual(local.call_count, 1)
        self.assertIn('no global route', mission.global_report)

    def test_moving_pose_and_new_scan_do_not_discard_global_route_update(self):
        mission = self.make_mission()
        replacement = self.route((.05, 0.), (2.5, 0.))
        mission.pending_route = (replacement, '', Pose2D(.05, 0., 0.), .01)
        # Deliberately exceed V1's stopped-plan start tolerance.
        pose = (.40, 0., 0.)
        self.feed(mission, 10.05, pose)
        mission.version += 1
        with patch.object(mission.local_planner, 'local_command',
                          wraps=mission.local_planner.local_command) as local:
            linear, _, complete = mission.step(10.05)
        self.assertIs(mission.global_path, replacement)
        self.assertEqual(mission.global_updates, 1)
        self.assertGreater(linear, 0., mission.reason)
        self.assertLessEqual(abs(linear-.12), .4*.05+1e-9)
        self.assertFalse(complete)
        self.assertEqual(local.call_args.args[1], Pose2D(*pose))
        self.assertEqual(local.call_args.args[4:6], (.12, 0.))

    def test_guidance_update_does_not_bypass_latest_local_stopping_check(self):
        mission = self.make_mission(velocity=.2)
        replacement = self.route()
        mission.pending_route = (replacement, '', Pose2D(.1, 0., 0.), .01)
        self.feed(mission, 10.05, velocity=.2)
        grid = self.wall_grid(.31)
        # The body fits now, but a new wall lies inside its braking sweep.
        self.assertTrue(mission.planner.pose_is_collision_free(grid, Pose2D(.2, 0., 0.)))
        with patch.object(mission, '_grid', return_value=grid), patch.object(
                mission.local_planner, 'local_command',
                wraps=mission.local_planner.local_command) as local:
            self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertIs(mission.global_path, replacement)
        self.assertEqual(local.call_count, 1)
        self.assertIs(local.call_args.args[0], grid)

    def test_stale_scan_stops_before_local_motion(self):
        mission = self.make_mission()
        mission.update_odometry((.2, 0., 0.), .12, 0., 11.)
        with patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(11.), (0., 0., False))
        local.assert_not_called()
        self.assertIn('lidar', mission.reason.lower())

    def test_current_body_collision_stops_with_free_centre_cell(self):
        mission = self.make_mission()
        self.feed(mission, 10.05)
        grid = self.wall_grid(.15)
        self.assertFalse(mission.planner.pose_is_collision_free(grid, Pose2D(.2, 0., 0.)))
        with patch.object(mission, '_grid', return_value=grid), patch.object(
                mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(10.05), (0., 0., False))
        local.assert_not_called()

    def test_worker_uses_current_pose_and_reset_discards_inflight_result(self):
        mission = self.make_mission()
        mission.worker = None
        started, release = threading.Event(), threading.Event()
        observed = []

        def search(grid, start, goal):
            observed.append(start)
            started.set()
            release.wait(timeout=2.)
            return self.route((start.x, start.y), (goal.x, goal.y))

        pose = Pose2D(.42, .03, .12)
        with patch.object(mission.local_planner, 'global_route', side_effect=search):
            mission._request_global_plan(pose, 11.)
            worker = mission.worker
            try:
                self.assertTrue(started.wait(timeout=1.))
                self.assertEqual(observed, [pose])
                mission.reset()
            finally:
                release.set()
                worker.join(timeout=2.)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(mission.pending_route)
        self.assertIsNone(mission.global_path)

    def test_worker_result_is_accepted_after_robot_moves_from_search_origin(self):
        mission = self.make_mission()
        mission.worker = None
        start = Pose2D(.2, 0., 0.)
        replacement = self.route((start.x, start.y), (2.5, 0.))
        with patch.object(mission.local_planner, 'global_route', return_value=replacement):
            mission._request_global_plan(start, 11.)
            mission.worker.join(timeout=2.)
        self.assertFalse(mission.worker.is_alive())
        self.feed(mission, 11.05, (.42, 0., 0.))
        mission._accept_global_plan(Pose2D(.42, 0., 0.), mission._grid())
        self.assertIs(mission.global_path, replacement)
        self.assertEqual(mission.last_linear, .12)

    def test_worker_error_retains_route_and_does_not_leave_a_live_worker(self):
        mission = self.make_mission()
        mission.worker = None
        route = mission.global_path
        with patch.object(mission.local_planner, 'global_route',
                          side_effect=RuntimeError('controlled search failure')):
            mission._request_global_plan(Pose2D(.2, 0., 0.), 11.)
            mission.worker.join(timeout=2.)
        self.assertFalse(mission.worker.is_alive())
        self.feed(mission, 11.05)
        linear, _, complete = mission.step(11.05)
        self.assertGreater(linear, 0., mission.reason)
        self.assertFalse(complete)
        self.assertIs(mission.global_path, route)
        self.assertIn('controlled search failure', mission.global_report)

    def test_goal_requires_new_measured_stop_before_completion(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.)
        self.feed(mission, 10.05, (.6, 0., 0.), 0.)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.assertEqual(mission.step(10.06), (0., 0., False))
        self.feed(mission, 10.10, (.6, 0., 0.), .04)
        self.assertEqual(mission.step(10.10), (0., 0., False))
        self.feed(mission, 10.15, (.6, 0., 0.), 0.)
        self.assertEqual(mission.step(10.15), (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')

    def test_heading_drift_while_stopping_is_corrected_before_handoff(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.)
        self.feed(mission, 10.05, (.6, 0., math.radians(9.)), 0., .4)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        # The zero command precedes the physical stop; residual rotation
        # carries the heading beyond tolerance before angular speed reaches 0.
        self.feed(mission, 10.10, (.6, 0., math.radians(10.2)), 0., .2)
        self.assertEqual(mission.step(10.10), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.feed(mission, 10.15, (.6, 0., math.radians(11.)), 0., 0.)
        with patch.object(mission.local_planner, 'local_command') as local:
            linear, angular, complete = mission.step(10.15)
        self.assertEqual(linear, 0.)
        self.assertLess(angular, 0.)
        self.assertFalse(complete)
        self.assertEqual(mission.state, 'FINISHING')
        local.assert_not_called()
        self.feed(mission, 10.20, (.6, 0., math.radians(9.)), 0., -.05)
        self.assertEqual(mission.step(10.20), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.assertEqual(mission.step(10.21), (0., 0., False))
        self.feed(mission, 10.25, (.6, 0., math.radians(9.)), 0., 0.)
        self.assertEqual(mission.step(10.25), (0., 0., True))

    def test_wait_stop_keeps_freshness_and_collision_checks_before_completion(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.)
        self.feed(mission, 10.05, (.6, 0., 0.), 0.)
        mission.step(10.05)
        self.assertEqual(mission.state, 'WAIT_STOP')
        mission.update_odometry((.6, 0., 0.), 0., 0., 11.)
        self.assertEqual(mission.step(11.), (0., 0., False))
        self.assertIn('lidar', mission.reason.lower())
        self.feed(mission, 11.05, (.6, 0., 0.), 0.)
        with patch.object(mission, '_grid', return_value=self.wall_grid(.6)):
            self.assertEqual(mission.step(11.05), (0., 0., False))
        self.assertIn('footprint overlaps', mission.reason)
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.feed(mission, 11.10, (.6, 0., 0.), 0.)
        self.assertEqual(mission.step(11.10), (0., 0., True))

    def test_recorded_exit_miss_finishes_without_turning_back(self):
        # 22:59 trial, rotated into an x-forward exit frame: 6 mm beyond
        # the goal, 35 mm right, heading 10.46 degrees left of the exit.
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=.08,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        self.assertFalse(mission._goal_reached(Pose2D(.39, -.035, 0.)))
        pose = (.606, -.035, math.radians(10.46))
        self.assertGreater(math.hypot(pose[0]-.6, pose[1]), .035)
        self.feed(mission, 10.05, pose, .08)
        with patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(10.05), (0., 0., True))
            local.assert_not_called()
        self.assertEqual(mission.state, 'COMPLETE')
        self.assertGreater(mission.odom[3], mission.control['planning_stopped_linear'])
        self.assertIn('exit crossed', mission.reason)

    def test_latest_exit_accepts_body_outside_aperture_extension_before_goal_plane(self):
        # Latest trial's first full-body exit, rotated into an x-forward frame.
        # Its right edge was 27 mm beyond the opening's imaginary extension;
        # that edge was already outside the actual wall, so it is not blocked.
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=.08,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        self.assertFalse(mission._goal_reached(Pose2D(.39, -.005, math.radians(-41.86))))
        pose = Pose2D(.5405, -.0747, math.radians(-11.55))
        cx, cy, c, s, half_length, half_width = mission.planner._footprint_geometry(pose)
        self.assertLess(cy-half_length*abs(s)-half_width*abs(c), -.15)
        self.assertLess(pose.x, mission.goal.x)
        self.feed(mission, 10.05, (pose.x, pose.y, pose.yaw), .08)
        with patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(10.05), (0., 0., True))
        self.assertTrue(mission.exit_crossed)
        self.assertEqual(mission.state, 'COMPLETE')
        local.assert_not_called()

    def test_physical_exit_hands_off_while_moving_without_heading_or_stop_wait(self):
        mission = self.make_mission(goal=(.6, 0., 120.), velocity=0.,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        mission._goal_reached(Pose2D(.39, 0., 0.))
        mission.last_linear = .12
        mission.last_angular = .3
        self.feed(mission, 10.05, (.56, .05, math.radians(25.)), .12, .3)
        with patch.object(mission.local_planner, 'local_command') as local, patch.object(
                mission.local_planner, 'command_is_safe',
                wraps=mission.local_planner.command_is_safe) as safe:
            self.assertEqual(mission.step(10.05), (0., 0., True))
            local.assert_not_called()
        self.assertEqual(mission.state, 'COMPLETE')
        self.assertEqual(safe.call_count, 1)
        command = safe.call_args.args[2]
        self.assertEqual((command.linear_velocity, command.angular_velocity), (0., 0.))
        self.assertEqual(safe.call_args.args[-2:], (.12, .3))

    def test_unconfigured_exit_keeps_original_position_trigger(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.)
        self.feed(mission, 10.05, (.606, -.035, 0.), 0.)
        _, angular, complete = mission.step(10.05)
        self.assertFalse(complete)
        self.assertNotEqual(angular, 0.)
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIn('turning toward route', mission.reason)

    def test_goal_stop_stays_latched_after_drifting_out_of_radius(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=.12)
        self.feed(mission, 10.05, (.59, 0., 0.), .12)
        first, _, _ = mission.step(10.05)
        self.assertEqual(mission.state, 'FINISHING')
        self.feed(mission, 10.10, (.66, 0., 0.), first)
        mission.pending_route = (self.route((.66, 0.), (.6, 0.)), '', Pose2D(.66, 0., 0.), .01)
        with patch.object(mission.local_planner, 'local_command') as local:
            linear, angular, complete = mission.step(10.10)
        self.assertLess(linear, first)
        self.assertEqual(angular, 0.)
        self.assertFalse(complete)
        self.assertEqual(mission.state, 'FINISHING')
        local.assert_not_called()

    def test_exit_crossing_requires_observed_outward_passage_and_entire_body_clear(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        # Starting already outside, even at the old goal, is not a passage.
        for pose in ((.6, 0., 0.), (.90, 0., 0.)):
            with self.subTest(pose=pose):
                self.assertFalse(mission._goal_reached(Pose2D(*pose)))
        # Reverse passage cannot finish. Forward axle crossing is remembered,
        # but the rotated rear corner still has to clear the wall.
        self.assertFalse(mission._goal_reached(Pose2D(.39, 0., 0.)))
        self.assertFalse(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(Pose2D(.45, 0., math.radians(25.))))
        self.assertTrue(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(Pose2D(.52, 0., math.radians(25.))))
        self.assertTrue(mission._goal_reached(Pose2D(.55, .07, math.radians(25.))))

    def test_exit_passage_interpolates_finite_opening_and_clears_on_return(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=0.,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        # Both endpoints' x coordinates straddle the exit plane, but their
        # interpolated intersection lies outside this particular opening.
        mission._goal_reached(Pose2D(.39, .25, 0.))
        self.assertFalse(mission._goal_reached(Pose2D(.41, .15, 0.)))
        self.assertFalse(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(Pose2D(.60, 0., 0.)))
        mission._goal_reached(Pose2D(.39, .14, 0.))
        # Current axle is outside the lateral range, but its crossing was in it.
        self.assertFalse(mission._goal_reached(Pose2D(.45, .16, 0.)))
        self.assertTrue(mission.exit_crossed)
        self.assertTrue(mission._goal_reached(Pose2D(.60, .17, 0.)))
        mission._goal_reached(Pose2D(.39, .17, 0.))
        self.assertFalse(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(Pose2D(.60, .17, 0.)))

    def test_exit_geometry_uses_portal_normal_not_goal_heading_or_goal_distance(self):
        for goal in ((.6, 0., 0.), (1.2, 0., 120.)):
            for segment in (((.4, -.15), (.4, .15)), ((.4, .15), (.4, -.15))):
                with self.subTest(goal=goal, segment=segment):
                    mission = self.make_mission(goal=goal, velocity=0., exit_crossing=segment)
                    self.assertFalse(mission._goal_reached(Pose2D(.39, 0., 0.)))
                    self.assertTrue(mission._goal_reached(Pose2D(.54, 0., 0.)))
                    mission.reset()
                    self.assertIsNone(mission.exit_previous_position)
                    self.assertFalse(mission.exit_crossed)
                    self.assertFalse(mission._goal_reached(Pose2D(.54, 0., 0.)))

    def test_finishing_still_stops_for_stale_lidar_and_current_collision(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=.12)
        self.feed(mission, 10.05, (.59, 0., 0.), .12)
        mission.step(10.05)
        mission.update_odometry((.64, 0., 0.), .1, 0., 11.)
        self.assertEqual(mission.step(11.), (0., 0., False))
        self.assertEqual(mission.state, 'FINISHING')
        self.assertIn('lidar', mission.reason.lower())
        self.feed(mission, 11.05, (.64, 0., 0.), 0.)
        with patch.object(mission, '_grid', return_value=self.wall_grid(.64)):
            self.assertEqual(mission.step(11.05), (0., 0., False))
        self.assertEqual(mission.state, 'FINISHING')
        self.assertIn('footprint overlaps', mission.reason)

    def test_finishing_still_checks_measured_braking_sweep(self):
        mission = self.make_mission(goal=(.6, 0., 0.), velocity=.12)
        self.feed(mission, 10.05, (.59, 0., 0.), .12)
        with patch.object(mission.local_planner, 'command_is_safe', return_value=False) as safe:
            self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'FINISHING')
        self.assertIn('sweep is blocked', mission.reason)
        self.assertEqual(safe.call_args.args[-2:], (.12, 0.))

    def test_physical_exit_keeps_sensor_collision_and_measured_braking_checks(self):
        mission = self.make_mission(goal=(.6, 0., 90.), velocity=0.,
                                    exit_crossing=((.4, -.15), (.4, .15)))
        mission._goal_reached(Pose2D(.39, 0., 0.))
        self.feed(mission, 10.05, (.54, 0., .2), .12, .3)
        with patch.object(mission.local_planner, 'command_is_safe', return_value=False) as safe:
            self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'FINISHING')
        self.assertEqual(mission.reason, 'exit crossing stopping sweep is blocked')
        self.assertEqual(safe.call_args.args[-2:], (.12, .3))
        mission.update_odometry((.55, 0., .2), 0., 0., 11.)
        self.assertEqual(mission.step(11.), (0., 0., False))
        self.assertIn('lidar', mission.reason.lower())
        self.feed(mission, 11.05, (.55, 0., .2), 0.)
        with patch.object(mission, '_grid', return_value=self.wall_grid(.55)):
            self.assertEqual(mission.step(11.05), (0., 0., False))
        self.assertEqual(mission.state, 'FINISHING')
        self.assertIn('footprint overlaps', mission.reason)
        # Recovery completes on the very first safe fresh frame, even while
        # moving, without returning to the old route or waiting for rest.
        self.feed(mission, 11.10, (.55, 0., .2), .12, .3)
        with patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(11.10), (0., 0., True))
            local.assert_not_called()
        self.assertEqual(mission.state, 'COMPLETE')

    def test_exit_crossing_rejects_invalid_survey_geometry(self):
        segments = ([], ((.4, 0.),), ((.4, 0.), (.4, 0.)),
                    ((.4, -.15), (.4, float('nan'))),
                    ((False, -.15), (.4, .15)),
                    ((.6, -.15), (.6, .15)),  # Guide on the exit plane.
                    ((.4, .1), (.4, .4)))     # Goal outside the portal width.
        for segment in segments:
            with self.subTest(segment=segment):
                config = configuration(goal=(.6, 0., 0.))
                config['goal']['exit_crossing'] = segment
                with self.assertRaises(ValueError):
                    TunnelMissionV2(config)

    def final_mission(self, heading=0.):
        # Exercise final_codes' real parent handoff without changing the
        # already-imported stepbystep package used by the rest of this suite.
        package = ModuleType('final_v2_mission_test')
        package.__path__ = [str(Path(__file__).resolve().parents[1]
                                / 'codes/final_codes/tunnel')]
        context = patch.dict(sys.modules, {package.__name__: package})
        context.start()
        self.addCleanup(context.stop)
        module = importlib.import_module(package.__name__ + '.mission_v2')
        config = configuration(goal=(.6, 0., heading))
        config['sensors']['lane_confirmation_frames'] = 3
        mission = module.TunnelMissionV2(config)
        self.addCleanup(mission.close)
        self.feed(mission, 10., (0., 0., 0.), 0.)
        mission._initialize((0., 0., 0.))
        mission.state = 'FOLLOWING'
        mission.last_time = 10.
        return mission

    def test_final_goal_position_alone_cannot_skip_heading_alignment(self):
        mission = self.final_mission(heading=90.)
        self.feed(mission, 10.05, (.6, 0., 0.), 0.)
        linear, angular, complete = mission.step(10.05)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)
        self.assertFalse(complete)
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIn('aligning heading', mission.reason)

    def test_final_lane_handoff_requires_new_confirmation_frames(self):
        mission = self.final_mission()
        for stamp in (9.9, 9.95, 10.):
            mission.update_lane(True, stamp)
        self.feed(mission, 10.05, (.6, 0., 0.), 0.)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_LANE')
        self.assertEqual(mission.lane_count, 0)
        self.assertEqual(mission.step(10.06), (0., 0., False))
        for index, stamp in enumerate((10.10, 10.15, 10.20)):
            self.feed(mission, stamp, (.6, 0., 0.), 0.)
            mission.update_lane(True, stamp)
            self.assertEqual(mission.step(stamp), (0., 0., index == 2))
        self.assertEqual(mission.state, 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
