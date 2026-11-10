"""Inside-door staging, plan ownership and ordinary exit passage."""

import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import Pose2D
from tunnel.mission import RouteUpdate, TunnelMission, load_config
from tunnel.planner import GlobalRoute
from tunnel.tracking import LimitedCommand


CONFIG = Path(__file__).resolve().parents[1] / 'codes/stepbystep/tunnel.real.yaml'
INSIDE = (1.808, 1.480)
OUTSIDE = (1.808, 1.868)


class ExitStagesTest(unittest.TestCase):
    def mission(self, pose=(1.2, 1.48, 0.), staged=True, crossing=True,
                exit_alignment_speed=None, entry_alignment_speed=None):
        config = load_config(CONFIG)
        config.pop('localization', None)
        config['goal'].pop('approach_distance', None)
        if exit_alignment_speed is not None:
            config['goal']['exit_alignment_angular_velocity'] = exit_alignment_speed
        if entry_alignment_speed is not None:
            config['entry']['alignment_angular_velocity'] = entry_alignment_speed
        if staged:
            config['goal']['approach_distance'] = .188
        if not crossing:
            config['goal'].pop('exit_crossing', None)
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        self.feed(mission, pose, 10.)
        mission._initialize((0., 0., 0.))
        mission.state = 'PLANNING'
        mission.last_time = 10.
        return mission

    @staticmethod
    def feed(mission, pose, stamp, linear=.2, angular=0.):
        mission.update_odometry(pose, linear, angular, stamp)
        mission.update_scan([math.inf], 0., 1., .1, 3.,
                            (0., 0., 0.), pose, stamp)

    def join_plan(self, mission):
        worker = mission.worker
        self.assertIsNotNone(worker)
        worker.join(2.)
        self.assertFalse(worker.is_alive(), 'planning worker did not finish')

    @staticmethod
    def route(grid, start, goal):
        return GlobalRoute(np.array(((start.x, start.y), (goal.x, goal.y))))

    def assert_endpoint(self, route, expected):
        np.testing.assert_allclose(route.points[-1], expected, atol=1e-12)

    def start_outside_alignment(self):
        mission = self.mission(pose=(*INSIDE, 0.), exit_alignment_speed=.9,
                               entry_alignment_speed=.3)
        mission.global_path = GlobalRoute(np.array(((1.2, 1.48), INSIDE)))
        mission.state = 'FOLLOWING'
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route):
            mission.step(10.05)
            mission.step(10.10)
            self.join_plan(mission)
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        mission.step(10.15)
        self.assertIsNotNone(mission.route_turn)
        return mission, 10.15

    def test_only_first_outside_alignment_ramps_to_the_configured_faster_speed(self):
        mission, stamp = self.start_outside_alignment()
        turns = []
        previous = mission.last_angular
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, 0.)):
            for _ in range(8):
                stamp += .05
                self.feed(mission, (*INSIDE, 0.), stamp, linear=.4, angular=.3)
                linear, angular, finished = mission.step(stamp)
                self.assertEqual(linear, 0.)
                self.assertFalse(finished)
                self.assertLessEqual(abs(angular-previous),
                                     mission.control['angular_acceleration']*.05+1e-10)
                turns.append(angular)
                previous = angular
        self.assertAlmostEqual(turns[0], .4)
        self.assertAlmostEqual(turns[1], .8)
        np.testing.assert_allclose(turns[2:], .9, atol=1e-10)

    def test_first_inside_route_alignment_uses_the_configured_entry_pivot_speed(self):
        mission = self.mission(pose=(1.2, 1.48, -math.pi/2), exit_alignment_speed=.9,
                               entry_alignment_speed=.3)
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        mission.pending_route = RouteUpdate(
            GlobalRoute(np.array(((1.2, 1.48), INSIDE))), '', .01)
        mission.step(10.05)
        self.assertIsNotNone(mission.route_turn)
        for stamp in (10.10, 10.15, 10.20):
            self.feed(mission, (1.2, 1.48, -math.pi/2), stamp)
            linear, angular, finished = mission.step(stamp)
            self.assertEqual(linear, 0.)
            self.assertAlmostEqual(angular, .3)
            self.assertFalse(finished)

    def test_later_outside_route_change_uses_the_configured_entry_pivot_speed(self):
        mission, stamp = self.start_outside_alignment()
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, 0.)):
            for _ in range(120):
                stamp += .05
                self.feed(mission, (*INSIDE, 0.), stamp)
                mission.step(stamp)
                if mission.route_turn is None:
                    break
            else:
                self.fail('first outside turn did not finish from its sent angle')
            mission.pending_route = RouteUpdate(
                GlobalRoute(np.array((INSIDE, (1.808, .90)))), '', .01)
            stamp += .05
            self.feed(mission, (*INSIDE, 0.), stamp)
            mission.step(stamp)
            self.assertIsNotNone(mission.route_turn)
            for _ in range(3):
                stamp += .05
                self.feed(mission, (*INSIDE, 0.), stamp)
                linear, angular, finished = mission.step(stamp)
                self.assertEqual(linear, 0.)
                self.assertAlmostEqual(angular, -.3)
                self.assertFalse(finished)

    def test_initial_global_search_ends_at_inside_point_only(self):
        mission = self.mission()
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search:
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.)
            self.join_plan(mission)
        target = search.call_args.args[2]
        np.testing.assert_allclose((target.x, target.y), INSIDE, atol=1e-12)
        self.assertTrue(mission._accept_global_plan())
        self.assert_endpoint(mission.global_path, INSIDE)
        self.assertNotIn(OUTSIDE, map(tuple, mission.global_path.points))

    def test_inside_stage_never_gives_local_tracking_the_outside_point(self):
        mission = self.mission()
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route):
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.)
            self.join_plan(mission)
        mission._accept_global_plan()
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, 0.)) as local:
            for i, xy in enumerate(((1.3, 1.48), (1.6, 1.40), (1.808, 1.43))):
                now = 10.05 + .05*i
                self.feed(mission, (*xy, 0.), now)
                self.assertGreater(mission.step(now)[0], 0.)
                self.assert_endpoint(local.call_args.args[2], INSIDE)
                self.assertNotIn(OUTSIDE, map(tuple, local.call_args.args[2].points))

    def test_reaching_inside_clears_old_pending_route_then_aligns_new_outside_route(self):
        mission = self.mission()
        old = GlobalRoute(np.array(((1.2, 1.48), INSIDE)))
        mission.global_path = old
        mission.pending_route = RouteUpdate(old, '', .01)
        mission.state = 'FOLLOWING'
        mission.last_linear = .2
        self.feed(mission, (*INSIDE, 0.), 10.05, linear=.4)
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search, \
                patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(10.05), (0., 0., False))
            self.assertIsNone(mission.global_path)
            self.assertIsNone(mission.pending_route)
            self.assertIsNone(mission.worker)
            self.assertEqual(mission.step(10.10), (0., 0., False))
            self.join_plan(mission)
            target = search.call_args.args[2]
            np.testing.assert_allclose((target.x, target.y), OUTSIDE, atol=1e-12)
            self.assertIsNone(mission.global_path)
            self.assertEqual(mission.step(10.15), (0., 0., False))
            self.assert_endpoint(mission.global_path, OUTSIDE)
            self.assertIsNotNone(mission.route_turn)
            local.assert_not_called()
            # Measured forward speed is deliberately nonzero: no stop confirmation wait.
            self.feed(mission, (*INSIDE, 0.), 10.20, linear=.4)
            linear, angular, finished = mission.step(10.20)
            self.assertEqual(linear, 0.)
            self.assertGreater(angular, 0.)
            self.assertFalse(finished)
            local.assert_not_called()

    def test_old_inside_worker_cannot_replace_new_outside_route(self):
        mission = self.mission()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def search(grid, start, goal):
            if math.isclose(goal.y, INSIDE[1], abs_tol=1e-9):
                entered.set()
                release.wait(3.)
            return self.route(grid, start, goal)

        with patch.object(mission.local_planner, 'global_route', side_effect=search):
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.)
            old_worker = mission.worker
            self.assertTrue(entered.wait(1.))
            try:
                mission.global_path = GlobalRoute(np.array(((1.2, 1.48), INSIDE)))
                mission.state = 'FOLLOWING'
                self.feed(mission, (*INSIDE, 0.), 10.05)
                mission.step(10.05)
                self.assertIsNone(mission.worker)
                mission.step(10.10)
                self.assertIsNot(mission.worker, old_worker)
                self.join_plan(mission)
                outside_update = mission.pending_route
                self.assertIsNotNone(outside_update)
                self.assert_endpoint(outside_update.route, OUTSIDE)
                release.set()
                old_worker.join(2.)
                self.assertFalse(old_worker.is_alive())
                self.assertIs(mission.pending_route, outside_update)
                self.assertTrue(mission._accept_global_plan())
                self.assert_endpoint(mission.global_path, OUTSIDE)
            finally:
                release.set()
                old_worker.join(2.)

    def test_exit_pivot_finishes_from_sent_angle_even_with_stationary_odometry(self):
        mission = self.mission()
        mission.global_path = GlobalRoute(np.array(((1.2, 1.48), INSIDE)))
        mission.state = 'FOLLOWING'
        self.feed(mission, (*INSIDE, 0.), 10.05, linear=.4)
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route):
            mission.step(10.05)
            mission.step(10.10)
            self.join_plan(mission)
        mission.step(10.15)
        self.assertIsNotNone(mission.route_turn)
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, 0.)) as local:
            stamp = 10.15
            for _ in range(200):
                stamp += .05
                self.feed(mission, (*INSIDE, 0.), stamp, linear=.4, angular=.3)
                linear, angular, finished = mission.step(stamp)
                self.assertFalse(finished)
                if mission.route_turn is None:
                    self.assertGreater(linear, 0.)
                    local.assert_called_once()
                    break
                self.assertEqual(linear, 0.)
            else:
                self.fail('exit alignment waited for measured heading instead of sent angle')

    def test_inside_stage_and_current_body_outside_are_required_for_completion(self):
        mission = self.mission(pose=(*INSIDE, math.pi/2))
        self.assertFalse(mission._goal_reached(Pose2D(*INSIDE, math.pi/2)))
        self.assertFalse(mission._goal_reached(Pose2D(1.808, 1.81, math.pi/2)))
        self.assertTrue(mission._advance_exit_goal(Pose2D(*INSIDE, math.pi/2)))
        self.assertFalse(mission._goal_reached(Pose2D(1.808, 1.65, math.pi/2)))
        self.assertFalse(mission._goal_reached(Pose2D(1.808, 1.70, math.pi/2)))
        self.assertIs(mission.route_goal, mission.goal)
        self.assertTrue(mission._goal_reached(Pose2D(1.808, 1.81, math.pi/2)))
        self.assertEqual(mission._finish_goal(Pose2D(1.808, 1.81, math.pi/2), .05, 10.2),
                         (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')

    def test_reset_restores_the_inside_stage(self):
        mission = self.mission(pose=(*INSIDE, 0.))
        mission.global_path = GlobalRoute(np.array(((1.2, 1.48), INSIDE)))
        mission.state = 'FOLLOWING'
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route):
            mission.step(10.05)
            mission.step(10.10)
            self.join_plan(mission)
        mission.reset()
        self.feed(mission, (1.2, 1.48, 0.), 10.2)
        mission._initialize((0., 0., 0.))
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search:
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.2)
            self.join_plan(mission)
        target = search.call_args.args[2]
        np.testing.assert_allclose((target.x, target.y), INSIDE, atol=1e-12)

    def test_approach_uses_door_outward_normal_not_a_fixed_world_axis(self):
        config = load_config(CONFIG)
        config.pop('localization', None)
        config['goal'].update(pose=[1.868, 1.808, 0.],
                              exit_crossing=[[1.668, 1.658], [1.668, 1.958]],
                              approach_distance=.188)
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        self.feed(mission, (1.1, 1.4, 0.), 10.)
        mission._initialize((0., 0., 0.))
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search:
            mission._request_global_plan(Pose2D(1.1, 1.4, 0.), 10.)
            self.join_plan(mission)
        target = search.call_args.args[2]
        np.testing.assert_allclose((target.x, target.y), (1.48, 1.808), atol=1e-12)

    def test_ordinary_goal_without_exit_keeps_direct_routing_and_tolerance(self):
        mission = self.mission(staged=False, crossing=False)
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search:
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.)
            self.join_plan(mission)
        self.assertEqual(search.call_args.args[2], mission.goal)
        self.assertFalse(mission._goal_reached(Pose2D(*INSIDE, 0.)))
        self.assertTrue(mission._goal_reached(Pose2D(*OUTSIDE, 0.)))

    def test_opening_without_staging_keeps_direct_outside_routing(self):
        mission = self.mission(staged=False)
        with patch.object(mission.local_planner, 'global_route', side_effect=self.route) as search:
            mission._request_global_plan(Pose2D(1.2, 1.48, 0.), 10.)
            self.join_plan(mission)
        self.assertEqual(search.call_args.args[2], mission.goal)
        self.assertTrue(mission._accept_global_plan())
        self.assert_endpoint(mission.global_path, OUTSIDE)


if __name__ == '__main__':
    unittest.main()
