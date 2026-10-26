"""ROS-free global-guidance/local-command integration, not driving evidence.

Worker timing is controlled; geometry and local command generation stay real.
"""
from pathlib import Path
import importlib
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
    def make_mission(self, goal=(2.5, 0., 0.), velocity=.12):
        config = configuration(goal=goal)
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
