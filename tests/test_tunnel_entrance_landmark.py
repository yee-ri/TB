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
        # Dense returns must also be rejected: sparse points alone happen to
        # split into tiny clusters and do not exercise candidate selection.
        for count in (60, 160):
            with self.subTest(count=count):
                matcher = EntranceLandmarkLocalizer()
                rear = base_points(right_wall(np.linspace(-2.7, -1.2, count)),
                                   (.213, -.022, 0.))
                result = matcher.estimate(rear, SENSOR)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.pose)


class RecordedEntranceTest(unittest.TestCase):
    """Nov 9 21:09 failed run, independent of ROS or access to the full bag.

    The fixture retains XY at the last scan's acquisition pose, after the
    runtime height/range filters and odom compensation. Offsets delimit 11
    scans: offset[1:] is the exact ten-scan fit; the first is pre-reset data.
    Long tunnel-wall observations support a heading near zero. The physical
    corner has no surveyed ground truth in this recording.
    """

    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent / 'fixtures/entrance_20261109_2109.npz') as data:
            cls.points = data['points']
            cls.offsets = data['offsets']
            cls.sensor = data['sensor']

    def test_recorded_heading_is_stable_when_one_scan_changes(self):
        headings = []
        for first in (0, 1, 2):
            matcher = EntranceLandmarkLocalizer(opposite_side_y=1.666)
            result = matcher.estimate(self.points[self.offsets[first]:], self.sensor)
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(matcher.diagnostics['heading_source'], 'opposite wall')
            self.assertLess(abs(result.pose.yaw), math.radians(1.))
            headings.append(result.pose.yaw)
        self.assertLess(max(headings)-min(headings), math.radians(.1))

    def test_noisy_near_returns_alone_cannot_confirm_heading(self):
        nearby = self.points[(np.linalg.norm(self.points-self.sensor, axis=1) < .3)
                             & (self.points[:, 1] < 0.)]
        for wrong_wall in ('missing', 'wrong width', 'behind entrance'):
            with self.subTest(wall=wrong_wall):
                observations = nearby
                if wrong_wall != 'missing':
                    xs = np.linspace(.7, 1.3, 100)
                    y = 1.2 if wrong_wall == 'wrong width' else 1.666
                    if wrong_wall == 'behind entrance':
                        xs -= 2.
                    observations = np.vstack((nearby, np.column_stack((xs, np.full(len(xs), y)))))
                result = EntranceLandmarkLocalizer(opposite_side_y=1.666).estimate(
                    observations, self.sensor)
                self.assertFalse(result.accepted)
                self.assertIn('heading uncertain', result.reason)

    def test_unrelated_long_line_cannot_rotate_a_clean_right_wall(self):
        nearby = base_points(right_wall(np.linspace(.14, .32, 40)), (.213, -.022, 0.))
        for degrees in (5., 10., 20., 30.):
            with self.subTest(degrees=degrees):
                angle = math.radians(degrees)
                tangent = np.array((math.cos(angle), math.sin(angle)))
                normal = np.array((-tangent[1], tangent[0]))
                # The wrong wall even has exactly the expected normal gap.
                offset = np.median(nearby @ normal)+1.816
                other = np.linspace(1.3, 1.9, 100)[:, None]*tangent+offset*normal
                matcher = EntranceLandmarkLocalizer(opposite_side_y=1.666)
                result = matcher.estimate(np.vstack((nearby, other)), SENSOR)
                self.assertTrue(result.accepted, result.reason)
                self.assertAlmostEqual(result.pose.yaw, 0., places=9)
                self.assertEqual(matcher.diagnostics['heading_source'], 'right wall')


if __name__ == '__main__':
    unittest.main()
