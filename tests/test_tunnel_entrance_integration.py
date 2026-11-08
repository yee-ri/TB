"""Initial entrance landmarks through the real mission sensor/worker pipeline."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))

from tunnel.entrance_localization import EntranceLandmarkLocalizer
from tunnel.mission import TunnelMission, load_config, relative_pose


def compose(pose, frame):
    c, s = math.cos(frame[2]), math.sin(frame[2])
    return (frame[0] + c*pose[0] - s*pose[1],
            frame[1] + s*pose[0] + c*pose[1], frame[2] + pose[2])


def side_wall_cloud(pose, sensor, wall):
    """Independently raycast only the physical right wall, including its end."""
    origin = np.asarray(compose(sensor, pose)[:2])
    angles = -math.pi + (np.arange(1440)+.5)*2*math.pi/1440
    base_angles = angles + sensor[2]
    directions = np.column_stack((np.cos(base_angles+pose[2]),
                                  np.sin(base_angles+pose[2])))
    with np.errstate(divide='ignore', invalid='ignore'):
        first = (np.asarray(wall[:2])-origin)/directions
        second = (np.asarray(wall[2:])-origin)/directions
        near = np.minimum(first, second).max(axis=1)
        far = np.maximum(first, second).min(axis=1)
    hit = (far >= np.maximum(near, 0.)) & (near >= .05) & (near <= 3.)
    return np.asarray(sensor[:2]) + near[hit, None]*np.column_stack(
        (np.cos(base_angles[hit]), np.sin(base_angles[hit])))


class EntranceIntegrationTest(unittest.TestCase):
    SENSOR = (-.033073, 0., 0.)
    POSE = (.28, -.02, .10)

    def mission(self, tracking=False):
        config = load_config(ROOT / 'codes/stepbystep/tunnel.real.yaml')
        config['localization']['observation_window'] = 0.
        config['localization']['tracking_period'] = .2 if tracking else 0.
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        return mission

    def feed(self, mission, stamp, pose=POSE, sensor=SENSOR,
             frame=(1.2, -.7, .4), obstacle_points=None):
        points = side_wall_cloud(
            pose, sensor, mission.config['localization']['wall_rectangles']['right_side_wall'])
        odom = compose(pose, frame)
        self.assertTrue(mission.update_odometry(odom, .02, .10, stamp))
        self.assertTrue(mission.update_cloud(
            points if obstacle_points is None else obstacle_points,
            sensor, odom, stamp, localization_points=points))
        return points, odom

    def initialize(self, mission, **feed_args):
        points, odom = self.feed(mission, 10., **feed_args)
        # Initial positioning uses entrance features, not the moving wall tracker.
        with patch.object(mission.localizer, 'track',
                          side_effect=AssertionError('wall tracking used for initialization')):
            mission.step(10.)
            worker = mission.localization_worker
            self.assertIsNotNone(worker)
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive())
            result = mission.localization_result[0]
            self.assertTrue(result.accepted, result.reason)
            linear, angular, complete = mission.step(10.01)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIsNotNone(mission.anchor)
        self.assertGreater(abs(linear)+abs(angular), 0.)
        self.assertFalse(complete)
        return points, odom

    def assert_map_pose(self, mission, odom, expected):
        actual = relative_pose(odom, mission.anchor)
        self.assertLess(math.hypot(actual.x-expected[0], actual.y-expected[1]), .002)
        self.assertLess(abs(actual.yaw-expected[2]), 1e-6)

    def test_real_profile_uses_its_surveyed_entrance_side(self):
        mission = self.mission()
        self.assertIsInstance(mission.entrance_localizer, EntranceLandmarkLocalizer)
        wall = mission.config['localization']['wall_rectangles']['right_side_wall']
        self.assertEqual(mission.entrance_localizer.side_start_x, wall[0])
        self.assertEqual(mission.entrance_localizer.side_y, wall[3])

    def test_existing_profile_without_localization_is_not_implicitly_enabled(self):
        config = load_config(ROOT / 'codes/final_codes/tunnel.real.yaml')
        self.assertNotIn('localization', config)
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        self.assertIsNone(mission.entrance_localizer)
        self.assertIsNone(mission.localizer)
        self.assertEqual(mission.state, 'ENTRY')

    def test_one_side_and_entrance_end_initialize_without_perpendicular_wall(self):
        mission = self.mission()
        points, odom = self.initialize(mission)
        self.assert_map_pose(mission, odom, self.POSE)
        result = mission.localizer.track(points, self.POSE, sensor_origin=self.SENSOR[:2])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'no extended perpendicular wall support')

    def test_backward_points_and_real_sensor_origin_reach_initial_worker(self):
        mission = self.mission()
        with patch.object(mission.entrance_localizer, 'estimate',
                          wraps=mission.entrance_localizer.estimate) as estimate:
            points, odom = self.initialize(mission)
        estimate.assert_called_once()
        passed, origin = estimate.call_args.args
        self.assertGreater(np.count_nonzero(points[:, 0] < 0.), 0)
        np.testing.assert_array_equal(passed, points)
        np.testing.assert_allclose(origin, self.SENSOR[:2], atol=0.)
        self.assert_map_pose(mission, odom, self.POSE)

    def test_lateral_sensor_offset_and_rotated_odometry_frame_are_not_double_transformed(self):
        mission = self.mission()
        sensor = (-.025, .012, .08)
        pose = (.226, -.016, .02932153)
        _, odom = self.initialize(mission, pose=pose, sensor=sensor,
                                  frame=(-2.1, 1.4, -.65))
        self.assert_map_pose(mission, odom, pose)

    def test_wall_height_points_can_initialize_without_becoming_obstacle_hits(self):
        mission = self.mission()
        _, odom = self.initialize(mission, obstacle_points=np.empty((0, 2)))
        self.assert_map_pose(mission, odom, self.POSE)
        self.assertEqual(sum(mission.costmap._dynamic_occupied), 0)

    def test_no_visible_entrance_returns_uses_prior_without_requiring_another_scan(self):
        mission = self.mission()
        odom = compose(self.POSE, (1.2, -.7, .4))
        mission.update_odometry(odom, 0., 0., 10.)
        mission.update_cloud(((1., 0.),), self.SENSOR, odom, 10.,
                             localization_points=np.empty((0, 2)))
        self.assertEqual(mission.step(10.), (0., 0., False))
        worker = mission.localization_worker
        self.assertIsNotNone(worker)
        worker.join(timeout=3.)
        self.assertFalse(worker.is_alive())
        result = mission.localization_result[0]
        self.assertFalse(result.accepted)
        linear, angular, complete = mission.step(10.01)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIn(result.reason, mission.localization_report)
        localized = relative_pose(odom, mission.anchor)
        np.testing.assert_allclose((localized.x, localized.y, localized.yaw),
                                   mission.entrance_localizer.prior, atol=1e-12)
        self.assertEqual(mission.mapped_stamp, 10.)
        self.assertGreater(abs(linear)+abs(angular), 0.)
        self.assertFalse(complete)

    def test_missing_perpendicular_support_during_tracking_keeps_entry_motion(self):
        mission = self.mission(tracking=True)
        self.initialize(mission)
        anchor = mission.anchor
        with patch.object(mission.localizer, 'track', wraps=mission.localizer.track) as track:
            self.feed(mission, 10.2)
            worker = mission.wall_tracking_worker
            self.assertIsNotNone(worker)
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive())
        track.assert_called_once()
        self.assertEqual(mission.anchor, anchor)
        self.assertEqual(mission.wall_tracking_updates, 0)
        self.assertIn('no extended perpendicular wall support', mission.wall_tracking_report)
        linear, angular, complete = mission.step(10.2)
        self.assertGreater(abs(linear)+abs(angular), 0.)
        self.assertFalse(complete)


if __name__ == '__main__':
    unittest.main()
