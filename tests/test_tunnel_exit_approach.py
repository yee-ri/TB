"""Inside exit arrival-area handoff, with the real vehicle and opening sizes."""

import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))

from tunnel.geometry import Pose2D
from tunnel.mission import RouteTurn, RouteUpdate, TunnelMission, load_config
from tunnel.planner import GlobalRoute


CONFIG = Path(__file__).resolve().parents[1] / 'codes/stepbystep/tunnel.real.yaml'


class ExitApproachAreaTest(unittest.TestCase):
    def make_mission(self, angle=0., reverse=False, approach=True):
        config = load_config(CONFIG)
        # These tests supply map-frame poses directly; no sensor fit is involved.
        config.pop('localization', None)
        if not approach:
            config['goal'].pop('approach_distance')
        if angle:
            c, s = math.cos(angle), math.sin(angle)

            def rotate(point):
                x, y = point
                return [c*x-s*y, s*x+c*y]

            goal = config['goal']['pose']
            config['goal']['pose'] = rotate(goal[:2]) + [goal[2]+math.degrees(angle)]
            config['goal']['exit_crossing'] = [
                rotate(point) for point in config['goal']['exit_crossing']]
        if reverse:
            config['goal']['exit_crossing'].reverse()
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        return mission

    @staticmethod
    def route(start, end):
        return GlobalRoute(np.array([[start.x, start.y], [end.x, end.y]]))

    @staticmethod
    def prepare_following(mission, pose, stamp=10.):
        coordinates = pose.x, pose.y, pose.yaw
        mission.update_odometry(coordinates, .2, .1, stamp)
        mission.update_scan([math.inf], 0., 1., .05, 3.,
                            (0., 0., 0.), coordinates, stamp)
        mission._initialize((0., 0., 0.))
        mission.state = 'FOLLOWING'
        mission.last_time = stamp
        mission.last_linear, mission.last_angular = .2, .1
        mission.global_path = ExitApproachAreaTest.route(pose, mission.route_goal)

    def test_offset_from_old_point_arrives_within_the_opening_area(self):
        mission = self.make_mission()
        target = mission.exit_approach_goal
        pose = Pose2D(target.x+.05, target.y, 0.)
        self.assertGreater(math.hypot(pose.x-target.x, pose.y-target.y),
                           mission.config['goal']['position_tolerance'])
        self.assertTrue(mission._advance_exit_goal(pose))
        self.assertIs(mission.route_goal, mission.goal)

    def test_forward_progress_past_the_old_point_still_arrives(self):
        mission = self.make_mission()
        target = mission.exit_approach_goal
        pose = Pose2D(target.x, target.y+.10, math.pi)
        self.assertTrue(mission._advance_exit_goal(pose))

    def test_real_opening_limits_include_edges_and_reject_outside_points(self):
        for x, y in ((1.7483, 1.445), (1.8677, 1.445),
                     (1.7483, 1.668), (1.8677, 1.668)):
            with self.subTest(inside=(x, y)):
                mission = self.make_mission()
                self.assertTrue(mission._advance_exit_goal(Pose2D(x, y, 0.)))
        for x, y in ((1.74829, 1.55), (1.86771, 1.55),
                     (1.808, 1.44499), (1.808, 1.66801),
                     (1.908, 1.48)):
            with self.subTest(outside=(x, y)):
                mission = self.make_mission()
                self.assertFalse(mission._advance_exit_goal(Pose2D(x, y, 0.)))
                self.assertIs(mission.route_goal, mission.exit_approach_goal)

    def test_region_follows_rotated_opening_and_reversed_endpoints(self):
        for angle in (0., math.pi/2., -.63):
            c, s = math.cos(angle), math.sin(angle)
            for reverse in (False, True):
                for (x, y), expected in (((1.858, 1.48), True),
                                         ((1.808, 1.58), True),
                                         ((1.908, 1.48), False),
                                         ((1.808, 1.44), False)):
                    with self.subTest(angle=angle, reverse=reverse, point=(x, y)):
                        mission = self.make_mission(angle, reverse)
                        pose = Pose2D(c*x-s*y, s*x+c*y, angle)
                        self.assertEqual(mission._advance_exit_goal(pose), expected)

    def test_area_entry_stops_immediately_and_discards_inside_guidance(self):
        mission = self.make_mission()
        target = mission.exit_approach_goal
        pose = Pose2D(target.x+.05, target.y, 0.)
        self.prepare_following(mission, pose)
        mission.pending_route = RouteUpdate(mission.global_path, '', .01)
        mission.route_turn = RouteTurn('PREPARE', math.pi/2., .3)
        with patch.object(mission.local_planner, 'local_command') as local:
            result = mission.step(10.05)
        self.assertEqual(result, (0., 0., False))
        self.assertEqual((mission.last_linear, mission.last_angular), (0., 0.))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertEqual(mission.reason,
                         'exit approach area reached; planning outside route')
        self.assertIs(mission.route_goal, mission.goal)
        self.assertIsNone(mission.global_path)
        self.assertIsNone(mission.pending_route)
        self.assertIsNone(mission.route_turn)
        local.assert_not_called()

    def test_outside_target_stays_latched_until_mission_reset(self):
        mission = self.make_mission()
        inside = mission.exit_approach_goal
        self.assertTrue(mission._advance_exit_goal(inside))
        self.assertFalse(mission._advance_exit_goal(Pose2D(inside.x, inside.y-.3, 0.)))
        self.assertIs(mission.route_goal, mission.goal)
        self.assertFalse(mission._advance_exit_goal(inside))
        mission.reset()
        self.assertIs(mission.route_goal, inside)
        self.assertTrue(mission._advance_exit_goal(inside))

    def test_old_running_search_cannot_restore_inside_guidance(self):
        mission = self.make_mission()
        target = mission.exit_approach_goal
        pose = Pose2D(target.x, target.y-.2, 0.)
        self.prepare_following(mission, pose)
        started, release = threading.Event(), threading.Event()
        old_route = mission.global_path

        def search(grid, start, goal):
            started.set()
            if not release.wait(2.):
                raise RuntimeError('test did not release the pending search')
            return old_route

        with patch.object(mission.local_planner, 'global_route', side_effect=search):
            mission._request_global_plan(pose, 10.05)
            worker = mission.worker
            try:
                self.assertTrue(started.wait(2.))
                with mission.lock:
                    self.assertTrue(mission._advance_exit_goal(target))
            finally:
                release.set()
                worker.join(2.)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(mission.pending_route)
        self.assertIsNone(mission.global_path)
        self.assertIs(mission.route_goal, mission.goal)

    def test_no_approach_goal_keeps_existing_exit_behavior(self):
        mission = self.make_mission(approach=False)
        self.assertIsNone(mission.exit_approach_goal)
        self.assertFalse(mission._advance_exit_goal(Pose2D(1.808, 1.55, 0.)))
        self.assertIs(mission.route_goal, mission.goal)


if __name__ == '__main__':
    unittest.main()
