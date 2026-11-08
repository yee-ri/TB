"""Entrance corner selection with nearby walls and unrelated room returns."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.entrance_localization import EntranceLandmarkLocalizer


SENSOR = (-.033073, 0.)


def base_points(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (np.asarray(points) - pose[:2]) @ np.array(((c, -s), (s, c)))


def right_wall(xs):
    return np.column_stack((xs, np.full(len(xs), -.15)))


class EntranceLandmarkTest(unittest.TestCase):
    def test_nearby_wall_corner_measures_pose(self):
        for pose in ((.213, -.022, 0.), (.225, -.03, math.radians(8)),
                     (.20, -.015, math.radians(-7))):
            with self.subTest(pose=pose):
                matcher = EntranceLandmarkLocalizer()
                cloud = base_points(right_wall(np.linspace(.14, .32, 40)), pose)
                result = matcher.estimate(cloud, SENSOR)
                self.assertTrue(result.accepted, result)
                np.testing.assert_allclose(
                    (result.pose.x, result.pose.y, result.pose.yaw), pose, atol=1e-10)

    def test_unrelated_rear_wall_does_not_move_entrance(self):
        for yaw in (0., math.radians(10)):
            with self.subTest(yaw=yaw):
                pose = (.213, -.022, yaw)
                matcher = EntranceLandmarkLocalizer(prior_pose=pose)
                nearby = base_points(right_wall(np.linspace(.14, .32, 40)), pose)
                rear = base_points(right_wall(np.linspace(-2.7, -1.2, 60)), pose)
                expected = matcher.estimate(nearby, SENSOR)
                actual = matcher.estimate(np.vstack((nearby, rear)), SENSOR)
                self.assertTrue(expected.accepted, expected)
                self.assertTrue(actual.accepted, actual)
                np.testing.assert_allclose(
                    (actual.pose.x, actual.pose.y, actual.pose.yaw),
                    (expected.pose.x, expected.pose.y, expected.pose.yaw), atol=1e-10)

    def test_remote_wall_alone_cannot_supply_entrance(self):
        matcher = EntranceLandmarkLocalizer()
        rear = base_points(right_wall(np.linspace(-2.7, -1.2, 60)),
                           (.213, -.022, 0.))
        result = matcher.estimate(rear, SENSOR)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)


if __name__ == '__main__':
    unittest.main()
