"""Exit observations correct the moving pose without adding drive waits."""

import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import Pose2D
from tunnel.localization import AlignmentResult
from tunnel.mission import TunnelMission, load_config, map_anchor, relative_pose
from test_tunnel_alignment_mission import CONFIG, SENSOR, compose


class ExitTrackingMissionTest(unittest.TestCase):
    def mission(self, pose=(1.80, 1.50, math.pi/2), frame=(0., 0., 0.)):
        config = load_config(CONFIG)
        config['localization']['exit'] = {}
        mission = TunnelMission(config)
        mission.update_odometry(compose(pose, frame), .2, .1, 10.)
        mission._initialize(frame)
        mission.state = 'FOLLOWING'
        mission.last_linear, mission.last_angular = .2, .1
        mission.localizer.track = Mock(return_value=AlignmentResult(False, reason='wall fixture'))
        self.addCleanup(mission.close)
        self.assertIsNotNone(mission.exit_localizer)
        return mission

    def feed(self, mission, pose, frame=(0., 0., 0.), stamp=10.1):
        odom = compose(pose, frame)
        mission.update_odometry(odom, .2, .1, stamp)
        points = np.array(((.3, .2), (.4, .2)))
        self.assertTrue(mission.update_cloud(points, SENSOR, odom, stamp,
                                             localization_points=points))

    def join(self, mission):
        mission.wall_tracking_worker.join(3.)
        self.assertFalse(mission.wall_tracking_worker.is_alive())

    def test_exit_fit_predicts_latest_motion_from_acquisition_pose_without_speed_reset(self):
        frame = (120., -87., 1.4)
        acquired = (1.77, 1.52, math.pi/2+.025)
        fitted = Pose2D(1.80, 1.50, math.pi/2)
        mission = self.mission(acquired, frame)
        entered, release = threading.Event(), threading.Event()

        def fit(*args):
            entered.set()
            release.wait(3.)
            return AlignmentResult(True, fitted)

        mission.exit_localizer.track = Mock(side_effect=fit)
        self.addCleanup(release.set)
        self.feed(mission, acquired, frame)
        self.assertTrue(entered.wait(1.))
        self.assertEqual(mission.exit_localizer.track.call_args.args[-1], SENSOR[:2])
        latest = compose((1.78, 1.58, math.pi/2+.04), frame)
        mission.update_odometry(latest, .2, .1, 10.15)
        release.set()
        self.join(mission)
        expected = relative_pose(latest, map_anchor(compose(acquired, frame), fitted))
        actual = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((actual.x, actual.y, actual.yaw),
                                   (expected.x, expected.y, expected.yaw), atol=1e-12)
        mission.localizer.track.assert_not_called()
        self.assertEqual((mission.last_linear, mission.last_angular), (.2, .1))
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertEqual(mission.wall_tracking_updates, 1)
        self.assertIn('source=exit opening', mission.wall_tracking_report)

    def test_missing_exit_falls_back_to_wall_fit_in_the_same_worker(self):
        mission = self.mission()
        calls = []

        def exit_fit(*args):
            calls.append('exit')
            return AlignmentResult(False, reason='one jamb hidden')

        fitted = Pose2D(1.81, 1.49, math.pi/2+.01)

        def wall_fit(*args):
            calls.append('wall')
            return AlignmentResult(True, fitted)

        mission.exit_localizer.track = Mock(side_effect=exit_fit)
        mission.localizer.track = Mock(side_effect=wall_fit)
        self.feed(mission, (1.80, 1.50, math.pi/2))
        self.join(mission)
        self.assertEqual(calls, ['exit', 'wall'])
        actual = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((actual.x, actual.y, actual.yaw),
                                   (fitted.x, fitted.y, fitted.yaw), atol=1e-12)
        self.assertEqual((mission.last_linear, mission.last_angular), (.2, .1))
        self.assertEqual(mission.state, 'FOLLOWING')

    def test_unobserved_exit_and_walls_keep_odometry_mapping_and_drive_state(self):
        mission = self.mission()
        anchor = mission.anchor
        mission.exit_localizer.track = Mock(return_value=AlignmentResult(False, reason='no opening'))
        self.feed(mission, (1.80, 1.50, math.pi/2))
        self.join(mission)
        mission.localizer.track.assert_called_once()
        self.assertEqual(mission.anchor, anchor)
        self.assertEqual(mission.mapped_stamp, 10.1)
        self.assertEqual(mission.wall_tracking_updates, 0)
        self.assertEqual((mission.last_linear, mission.last_angular), (.2, .1))
        self.assertEqual(mission.state, 'FOLLOWING')

    def test_stationary_anchor_change_cannot_create_an_exit_passage(self):
        for frame in ((0., 0., 0.), (4., -2., .6)):
            with self.subTest(frame=frame):
                pose = (1.808, 1.65, math.pi/2)
                mission = self.mission(pose, frame)
                self.assertFalse(mission._goal_reached(Pose2D(*pose)))
                odom = compose(pose, frame)
                mission._apply_tracking_pose(odom, Pose2D(1.808, 1.70, math.pi/2+.03))
                corrected = relative_pose(odom, mission.anchor)
                self.assertFalse(mission._goal_reached(corrected))
                self.assertFalse(mission.exit_crossed)

    def test_small_correction_after_real_axle_crossing_preserves_later_body_completion(self):
        mission = self.mission((1.808, 1.65, math.pi/2))
        self.assertFalse(mission._goal_reached(Pose2D(1.808, 1.65, math.pi/2)))
        odom = (1.808, 1.70, math.pi/2)
        mission.update_odometry(odom, .2, 0., 10.1)
        self.assertFalse(mission._goal_reached(Pose2D(*odom)))
        self.assertTrue(mission.exit_crossed)
        mission._apply_tracking_pose(odom, Pose2D(1.808, 1.69, math.pi/2))
        self.assertTrue(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(relative_pose(odom, mission.anchor)))
        latest = (1.808, 1.82, math.pi/2)
        mission.update_odometry(latest, .2, 0., 10.2)
        self.assertTrue(mission._goal_reached(relative_pose(latest, mission.anchor)))

    def test_correction_back_inside_clears_passage_and_real_recrossing_can_finish(self):
        mission = self.mission((1.808, 1.65, math.pi/2))
        mission._goal_reached(Pose2D(1.808, 1.65, math.pi/2))
        odom = (1.808, 1.70, math.pi/2)
        mission.update_odometry(odom, .2, 0., 10.1)
        mission._goal_reached(Pose2D(*odom))
        self.assertTrue(mission.exit_crossed)
        mission._apply_tracking_pose(odom, Pose2D(1.808, 1.64, math.pi/2))
        self.assertFalse(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(relative_pose(odom, mission.anchor)))
        later = (1.808, 1.76, math.pi/2)
        mission.update_odometry(later, .2, 0., 10.2)
        self.assertFalse(mission._goal_reached(relative_pose(later, mission.anchor)))
        self.assertTrue(mission.exit_crossed)
        outside = (1.808, 1.90, math.pi/2)
        mission.update_odometry(outside, .2, 0., 10.3)
        self.assertTrue(mission._goal_reached(relative_pose(outside, mission.anchor)))

    def test_lateral_correction_outside_opening_invalidates_the_recorded_crossing(self):
        frame = (4., -2., .6)
        mission = self.mission((1.70, 1.65, math.pi/2), frame)
        mission._goal_reached(Pose2D(1.70, 1.65, math.pi/2))
        odom = compose((1.70, 1.70, math.pi/2), frame)
        mission.update_odometry(odom, .2, 0., 10.1)
        mission._goal_reached(Pose2D(1.70, 1.70, math.pi/2))
        self.assertTrue(mission.exit_crossed)
        # The measured doorway places the earlier crossing 18 mm left of it.
        mission._apply_tracking_pose(odom, Pose2D(1.64, 1.70, math.pi/2))
        self.assertFalse(mission.exit_crossed)
        self.assertFalse(mission._goal_reached(relative_pose(odom, mission.anchor)))
        outside = compose((1.70, 1.83, math.pi/2), frame)
        mission.update_odometry(outside, .2, 0., 10.2)
        self.assertFalse(mission._goal_reached(relative_pose(outside, mission.anchor)))


if __name__ == '__main__':
    unittest.main()
