import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from tunnel.entrance_localization import EntranceLandmarkLocalizer


def in_base(mapped, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (np.asarray(mapped, dtype=float) - np.asarray(pose[:2])) @ np.array(
        ((c, -s), (s, c)))


def line_x(first=.14, last=.50, y=-.15, count=501):
    return np.column_stack((np.linspace(first, last, count), np.full(count, y)))


def line_y(x, first, last, count):
    return np.column_stack((np.full(count, x), np.linspace(first, last, count)))


class EntranceConnectivityTests(unittest.TestCase):
    origin = (-.033073, 0.)

    def assert_pose(self, mapped, pose, tolerance=1e-7):
        result = EntranceLandmarkLocalizer().estimate(in_base(mapped, pose), self.origin)
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose(
            [result.pose.x, result.pose.y, result.pose.yaw], pose,
            atol=tolerance, rtol=0.)
        return result

    def test_observed_corner_is_used_at_different_positions_and_headings(self):
        for pose in [(.163, -.022, 0.), (.313, -.022, 0.),
                     (.213, -.050, 0.), (.213, .055, 0.),
                     (.213, -.035, .20), (.213, .055, -.20),
                     (.280, .025, math.radians(30)),
                     (.175, -.050, math.radians(-30))]:
            with self.subTest(pose=pose):
                self.assert_pose(line_x(), pose)

    def test_disconnected_collinear_returns_do_not_move_corner_metres_away(self):
        mapped = np.vstack((line_x(),
                            line_x(-3.5, -3.1, count=1200),
                            line_x(1.2, 1.7, count=1300)))
        for pose in [(.213, -.022, 0.), (.263, -.010, .04)]:
            with self.subTest(pose=pose):
                self.assert_pose(mapped, pose)

    def test_more_left_and_cross_wall_returns_do_not_replace_right_wall(self):
        mapped = np.vstack((line_x(count=200),
                            line_x(y=.15, count=2000),
                            line_y(.142, .15, .65, 3000)))
        for pose in [(.213, -.022, 0.), (.263, .015, -.15)]:
            with self.subTest(pose=pose):
                self.assert_pose(mapped, pose)

    def test_connected_perpendicular_face_does_not_rotate_right_wall_fit(self):
        mapped = np.vstack((line_x(count=300),
                            line_y(.50, -.15, .30, 1200)))
        self.assert_pose(mapped, (.226, -.016, .03), tolerance=.002)

    def test_connected_nonuniform_samples_keep_the_physical_endpoint(self):
        along = np.unique(np.concatenate((
            [.14], np.arange(.151, .205, .011),
            np.linspace(.205, .315, 701), np.arange(.325, .50, .013))))
        mapped = np.column_stack((along, np.full(len(along), -.15)))
        self.assert_pose(mapped, (.263, -.008, -.07))

    def test_noisy_wall_direction_uses_observation_support_before_pose_prior(self):
        pose = np.array([.226, -.016, math.radians(1.68)])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        sensor = np.asarray(self.origin) @ np.array(((c, s), (-s, c))) + pose[:2]
        first = math.atan2(-.15-sensor[1], .14-sensor[0])
        last = math.atan2(-.15-sensor[1], 1.25-sensor[0])
        angles = np.tile(np.linspace(first, last, 400), 4)
        ranges = (-.15-sensor[1]) / np.sin(angles)
        # Equal angular spacing gives denser returns near the entrance. Range
        # noise makes short tilted lines plausible, but the full wall stays straight.
        ranges += np.random.RandomState(0).normal(0., .009, len(angles))
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        mapped = sensor + directions * ranges[:, None]

        result = EntranceLandmarkLocalizer().estimate(in_base(mapped, pose), self.origin)

        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose([result.pose.x, result.pose.y], pose[:2],
                                   atol=.003, rtol=0.)
        self.assertLess(abs(result.pose.yaw-pose[2]), math.radians(.5))

    def test_shuffle_and_repeated_returns_do_not_change_the_selected_corner(self):
        mapped = np.vstack((line_x(count=101),
                            line_x(.22, .28, count=1001),
                            line_x(y=.15, count=451),
                            line_x(-2.0, -1.7, count=151)))
        rng = np.random.RandomState(42)
        pose = (.253, -.002, .08)
        for repetition in range(3):
            with self.subTest(repetition=repetition):
                rng.shuffle(mapped)
                self.assert_pose(mapped, pose)

    def test_nonfinite_rows_are_ignored_when_connected_wall_is_available(self):
        pose = (.226, -.016, .03)
        points = np.vstack((in_base(line_x(), pose),
                            [[np.nan, 0.], [0., np.inf], [-np.inf, np.nan]]))
        result = EntranceLandmarkLocalizer().estimate(points, self.origin)
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose(
            [result.pose.x, result.pose.y, result.pose.yaw], pose,
            atol=1e-7, rtol=0.)

    def test_bad_point_shape_and_sensor_origin_raise_clear_errors(self):
        for points in [np.zeros(4), np.zeros((5, 3)), np.zeros((2, 3, 4))]:
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                EntranceLandmarkLocalizer().estimate(points)
        for origin in [(0.,), (0., 0., 0.), (np.nan, 0.), (0., np.inf)]:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                EntranceLandmarkLocalizer().estimate(np.zeros((10, 2)), origin)

    def test_invalid_connectivity_settings_are_rejected(self):
        for name in ['cluster_distance', 'sampling_resolution']:
            for value in [0., -.01, float('nan'), float('inf')]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    EntranceLandmarkLocalizer(**{name: value})


if __name__ == '__main__':
    unittest.main()
