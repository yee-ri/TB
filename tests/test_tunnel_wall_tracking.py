"""Moving wall localization without ROS or any drive publisher."""

import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import Pose2D
from tunnel.localization import AlignmentResult
from tunnel.mission import TunnelMission, load_config, map_anchor, relative_pose
from test_tunnel_alignment_mission import CONFIG, SENSOR, compose, fixed_wall_cloud


class WallTrackingMissionTest(unittest.TestCase):
    def mission(self, pose=(.6, .5, 0.), frame=(0., 0., 0.)):
        mission = TunnelMission(load_config(CONFIG))
        mission.update_odometry(compose(pose, frame), .2, 0., 10.)
        mission._initialize(frame)
        mission.state = 'FOLLOWING'
        mission.last_linear, mission.last_angular = .2, .1
        self.addCleanup(mission.close)
        return mission

    def feed(self, mission, stamp=10.1, pose=(.6, .5, 0.), frame=(0., 0., 0.),
             points=((.3, .2),), walls=None):
        odom = compose(pose, frame)
        mission.update_odometry(odom, .2, 0., stamp)
        if walls is None:
            walls = fixed_wall_cloud(pose, mission.config['localization']['wall_rectangles'])
        self.assertTrue(mission.update_cloud(points, SENSOR, odom, stamp, localization_points=walls))

    def join(self, mission):
        mission.wall_tracking_worker.join(3.)
        self.assertFalse(mission.wall_tracking_worker.is_alive())

    def delayed_fit(self, mission, result):
        entered, release = threading.Event(), threading.Event()

        def fit(*args):
            entered.set()
            release.wait(3.)
            return result

        mission.localizer.track = Mock(side_effect=fit)
        self.addCleanup(release.set)
        return entered, release

    def test_fit_applies_acquisition_pose_and_predicts_current_pose_without_speed_reset(self):
        frame = (120., -87., 1.4)
        mission = self.mission(frame=frame)
        fitted = Pose2D(.62, .49, .03)
        entered, release = self.delayed_fit(mission, AlignmentResult(True, fitted))
        self.feed(mission, frame=frame)
        self.assertTrue(entered.wait(1.))
        self.assertEqual(mission.localizer.track.call_args.args[-1], SENSOR[:2])
        latest = compose((.67, .5, .02), frame)
        mission.update_odometry(latest, .2, .1, 10.15)
        release.set()
        self.join(mission)
        expected = relative_pose(latest, map_anchor(compose((.6, .5, 0.), frame), fitted))
        actual = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((actual.x, actual.y, actual.yaw),
                                   (expected.x, expected.y, expected.yaw), atol=1e-12)
        self.assertEqual((mission.last_linear, mission.last_angular), (.2, .1))
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertEqual(mission.wall_tracking_updates, 1)

    def test_corrected_scan_maps_once_in_corrected_frame_and_keeps_old_observations(self):
        mission = self.mission()
        # Keep a pre-existing, correctly mapped obstacle; do not shift it with the robot.
        mission.costmap.update_scan = Mock()
        entered, release = self.delayed_fit(mission, AlignmentResult(True, Pose2D(.62, .5, 0.)))
        self.feed(mission)
        self.assertTrue(entered.wait(1.))
        mission.costmap.update_scan.assert_not_called()
        original_map = mission.costmap
        release.set()
        self.join(mission)
        self.assertIs(mission.costmap, original_map)
        mission.costmap.update_scan.assert_called_once()
        sensor_pose = mission.costmap.update_scan.call_args.args[-1]
        np.testing.assert_allclose(sensor_pose, (.62+SENSOR[0], .5, 0.))
        self.assertEqual(mission.mapped_stamp, 10.1)

    def test_latest_scan_arriving_during_fit_is_mapped_with_its_own_odometry(self):
        mission = self.mission()
        mission.costmap.update_scan = Mock()
        entered, release = self.delayed_fit(mission, AlignmentResult(True, Pose2D(.62, .5, 0.)))
        self.feed(mission)
        self.assertTrue(entered.wait(1.))
        self.feed(mission, 10.2, (.65, .5, 0.))
        mission.costmap.update_scan.assert_not_called()
        release.set()
        self.join(mission)
        self.assertEqual(mission.mapped_stamp, 10.2)
        self.assertEqual(mission.costmap.update_scan.call_count, 2)
        np.testing.assert_allclose(mission.costmap.update_scan.call_args.args[-1],
                                   (.67+SENSOR[0], .5, 0.))

    def test_rejected_fit_keeps_odometry_but_maps_new_obstacles(self):
        mission = self.mission()
        anchor = mission.anchor
        mission.localizer.track = Mock(return_value=AlignmentResult(False, reason='one wall only'))
        self.feed(mission)
        self.join(mission)
        self.assertEqual(mission.anchor, anchor)
        self.assertEqual(mission.mapped_stamp, 10.1)
        self.assertEqual(mission.wall_tracking_updates, 0)
        self.assertIn('odom retained: one wall only', mission.wall_tracking_report)

    def test_timeout_releases_mapping_and_late_result_cannot_change_anchor(self):
        mission = self.mission()
        anchor = mission.anchor
        with patch('tunnel.mission.time.monotonic', return_value=100.) as clock:
            entered, release = self.delayed_fit(mission, AlignmentResult(True, Pose2D(.62, .5, 0.)))
            self.feed(mission)
            self.assertTrue(entered.wait(1.))
            clock.return_value = 100.31
            self.feed(mission, 10.2)
            self.assertEqual(mission.mapped_stamp, 10.2)
            release.set()
            self.join(mission)
        self.assertEqual(mission.anchor, anchor)
        self.assertIn('late wall correction discarded', mission.wall_tracking_report)

    def test_slow_fit_preserves_every_scan_and_queue_overflow_falls_back_to_odom(self):
        for count in (3, 6):
            with self.subTest(scans=count):
                mission = self.mission()
                anchor = mission.anchor
                entered, release = self.delayed_fit(mission, AlignmentResult(True, Pose2D(.62, .5, 0.)))
                mission.costmap.update_scan = Mock()
                for index in range(count):
                    self.feed(mission, 10.1 + index*.02)
                    self.assertTrue(entered.wait(1.))
                release.set()
                self.join(mission)
                self.assertEqual(mission.costmap.update_scan.call_count, count)
                self.assertAlmostEqual(mission.mapped_stamp, 10.1+(count-1)*.02)
                self.assertEqual(mission.wall_tracking_updates, int(count <= 4))
                if count > 4:
                    self.assertEqual(mission.anchor, anchor)

    def test_reset_and_completion_discard_inflight_fit(self):
        for action in ('reset', 'complete'):
            with self.subTest(action=action):
                mission = self.mission()
                entered, release = self.delayed_fit(mission, AlignmentResult(True, Pose2D(.62, .5, 0.)))
                self.feed(mission)
                self.assertTrue(entered.wait(1.))
                if action == 'reset':
                    mission.reset()
                else:
                    mission.state = 'COMPLETE'
                anchor, stamp = mission.anchor, mission.mapped_stamp
                release.set()
                self.join(mission)
                self.assertEqual(mission.anchor, anchor)
                self.assertEqual(mission.mapped_stamp, stamp)
                self.assertEqual(mission.wall_tracking_updates, 0)

    def test_fit_does_not_wait_for_global_worker_or_block_control(self):
        mission = self.mission()
        mission.worker = Mock(is_alive=Mock(return_value=True))
        entered, release = self.delayed_fit(mission, AlignmentResult(False, reason='not enough walls'))
        self.feed(mission)
        self.assertTrue(entered.wait(1.))
        # Control can enter the mission lock and continue its normal checks.
        mission.global_path = object()
        mission.local_planner.local_command = Mock(return_value=None)
        mission.step(10.1)
        self.assertIn('no local command', mission.reason)
        self.assertFalse(release.is_set())
        release.set()
        self.join(mission)

    def test_weak_wall_only_cloud_supports_tracking_without_free_space_rays(self):
        mission = self.mission()
        self.feed(mission, points=())
        self.join(mission)
        self.assertEqual(mission.wall_tracking_updates, 1)
        self.assertTrue(np.isnan(mission.scan[0]).all())
        self.assertFalse(any(mission.costmap._dynamic_occupied))
        self.assertFalse(mission.update_cloud([], SENSOR, (.6, .5, 0.), 10.2, localization_points=[]))

    def test_real_wall_fit_corrects_drift_and_does_not_use_entrance_bounds(self):
        actual = (1.8, 1.5, math.pi/2)
        drifted = (1.77, 1.52, math.pi/2+.025)
        mission = self.mission(drifted)
        walls = fixed_wall_cloud(actual, mission.config['localization']['wall_rectangles'])
        self.feed(mission, pose=drifted, walls=walls)
        self.join(mission)
        pose = relative_pose(mission.odom[:3], mission.anchor)
        self.assertEqual(mission.wall_tracking_updates, 1, mission.wall_tracking_report)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), actual, atol=.003)

    def test_corrected_body_still_inside_cannot_finish_on_uncorrected_exit_position(self):
        mission = self.mission((1.81, 1.81, math.pi/2))
        mission.exit_crossed = True
        mission.localizer.track = Mock(return_value=AlignmentResult(True, Pose2D(1.81, 1.75, math.pi/2)))
        self.feed(mission, pose=(1.81, 1.81, math.pi/2))
        self.join(mission)
        self.assertTrue(mission._goal_reached(Pose2D(1.81, 1.81, math.pi/2)))
        corrected = relative_pose(mission.odom[:3], mission.anchor)
        self.assertFalse(mission._goal_reached(corrected))

    def test_period_limits_work_and_disabled_tracking_maps_immediately(self):
        mission = self.mission()
        mission.localizer.track = Mock(return_value=AlignmentResult(False, reason='fixture'))
        self.feed(mission)
        self.join(mission)
        self.feed(mission, 10.2)
        self.assertEqual(mission.mapped_stamp, 10.2)
        mission.localizer.track.assert_called_once()
        self.feed(mission, 10.3)
        self.join(mission)
        self.assertEqual(mission.localizer.track.call_count, 2)
        mission.wall_tracking_period = 0.
        self.feed(mission, 10.6)
        self.assertEqual(mission.mapped_stamp, 10.6)
        self.assertEqual(mission.localizer.track.call_count, 2)

    def test_invalid_tracking_configuration_fails_at_startup(self):
        for key, value in (('tracking_period', -1.), ('tracking_period', True),
                           ('tracking_max_translation', 0.), ('tracking_max_yaw_deg', 90.),
                           ('tracking_max_yaw_deg', float('nan'))):
            with self.subTest(key=key, value=value):
                config = load_config(CONFIG)
                config['localization'][key] = value
                with self.assertRaises(ValueError):
                    TunnelMission(config)


if __name__ == '__main__':
    unittest.main()
