"""Pure geometric initial registration; no ROS, robot or Gazebo transport."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes/stepbystep'))
from tunnel.localization import EntranceLocalizer


WALLS = json.loads((ROOT / 'codes/stepbystep/maps/tunnel_practice_1820.geometry.json')
                   .read_text())['wall_rectangles']


def wall_cloud(pose, bins=1440):
    """Independent ray/rectangle fixture, returning base-frame hit endpoints."""
    sensor = np.array((-.033073, 0.))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    origin = np.array(pose[:2]) + np.array(((c, -s), (s, c))) @ sensor
    angles = -math.pi+(np.arange(bins)+.5)*2*math.pi/bins
    directions = np.column_stack((np.cos(angles+pose[2]), np.sin(angles+pose[2])))
    ranges = np.full(bins, np.inf)
    for rectangle in WALLS.values():
        low = (np.array(rectangle[:2])-origin)/directions
        high = (np.array(rectangle[2:])-origin)/directions
        near = np.minimum(low, high).max(axis=1)
        far = np.maximum(low, high).min(axis=1)
        hits = far >= np.maximum(near, 0.)
        distances = np.where(near >= 0., near, far)
        ranges = np.minimum(ranges, np.where(hits, distances, np.inf))
    hit = np.isfinite(ranges) & (ranges >= .05) & (ranges <= 3.)
    return sensor + ranges[hit, None]*np.column_stack((np.cos(angles[hit]), np.sin(angles[hit])))


class EntranceLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.matcher = EntranceLocalizer(WALLS)

    def assert_pose(self, result, expected, position=.006, yaw=.004):
        self.assertTrue(result.accepted, result)
        self.assertLess(math.hypot(result.pose.x-expected[0], result.pose.y-expected[1]), position)
        self.assertLess(abs(result.pose.yaw-expected[2]), yaw)

    def test_different_longitudinal_lateral_and_heading_starts(self):
        for pose in ((0., 0., 0.), (.04, 0., 0.), (-.2, .035, .17),
                     (.30, -.035, -.2), (.08, -.03, .3)):
            with self.subTest(pose=pose):
                self.assert_pose(self.matcher.estimate(wall_cloud(pose)), pose)

    def test_noise_and_unmapped_obstacles(self):
        rng = np.random.default_rng(20261012)
        pose = (.04, -.02, -.08)
        cloud = wall_cloud(pose)
        cloud += rng.normal(0., .003, cloud.shape)
        obstacles = rng.uniform((.35, -.08), (1.6, 1.4), (60, 2))
        self.assert_pose(self.matcher.estimate(np.vstack((cloud, obstacles))), pose,
                         position=.008, yaw=.006)

    def test_one_wall_has_no_lateral_observability(self):
        far = np.column_stack((np.full(120, 1.958), np.linspace(-.1, 1.3, 120)))
        result = self.matcher.estimate(far)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)

    def test_far_wall_and_short_bottle_are_not_perpendicular_walls(self):
        far = np.column_stack((np.full(100, 1.958), np.linspace(-.05, 1.3, 100)))
        angles = np.linspace(0., 2.*math.pi, 200)
        bottle = np.column_stack((1.4+.025*np.cos(angles), -.125+.025*np.sin(angles)))
        result = self.matcher.estimate(np.vstack((far, bottle)))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)

    def test_separated_row_of_bottles_does_not_supply_wall_extent(self):
        far = np.column_stack((np.full(100, 1.958), np.linspace(-.05, 1.3, 100)))
        angle = np.linspace(0., 2.*math.pi, 200)
        bottles = np.vstack([np.column_stack((x+.025*np.cos(angle),
                                               -.125+.025*np.sin(angle)))
                             for x in (.6, .9, 1.2, 1.5)])
        self.assertFalse(self.matcher.estimate(np.vstack((far, bottles))).accepted)

    def test_room_walls_and_starts_outside_entrance_search_are_rejected(self):
        room = np.vstack((np.column_stack((np.full(150, 2.3), np.linspace(.4, 1.5, 150))),
                          np.column_stack((np.linspace(.5, 2.3, 150), np.full(150, .4)))))
        self.assertFalse(self.matcher.estimate(room).accepted)
        for pose in ((.8, 0., 0.), (-.6, 0., 0.), (.04, .17, 0.)):
            with self.subTest(pose=pose):
                self.assertFalse(self.matcher.estimate(wall_cloud(pose)).accepted)

    def test_inconsistent_wall_measurements_fail_residual_gate(self):
        cloud = wall_cloud((.04, -.02, -.08))
        cloud += np.random.default_rng(615).normal(0., .045, cloud.shape)
        self.assertFalse(self.matcher.estimate(cloud).accepted)

    def test_bottles_alone_and_empty_cloud_do_not_initialize(self):
        angle = np.linspace(0., 2.*math.pi, 150)
        cloud = np.vstack([np.column_stack((x+.03*np.cos(angle), y+.03*np.sin(angle)))
                           for x, y in ((.4, .3), (.9, .4), (1.3, -.1))])
        for points in (cloud, np.empty((0, 2)), np.full((100, 2), np.nan)):
            with self.subTest(points=len(points)):
                self.assertFalse(self.matcher.estimate(points).accepted)

    def test_ambiguous_two_equal_maps_do_not_initialize(self):
        points = np.vstack((wall_cloud((-.12, 0., 0.)), wall_cloud((.12, 0., 0.))))
        # Even if two hypotheses individually contain enough inliers, they
        # must not choose one arbitrarily. Shared horizontal walls are real.
        matcher = EntranceLocalizer(WALLS, minimum_inlier_fraction=.35)
        result = matcher.estimate(points)
        self.assertFalse(result.accepted, result)

    def test_invalid_geometry_and_input(self):
        with self.assertRaises(ValueError):
            EntranceLocalizer([[0., 0., 0., 1.]])
        with self.assertRaises(ValueError):
            EntranceLocalizer(WALLS, search_bounds=(1., 0., 0., 1.))
        with self.assertRaises(ValueError):
            self.matcher.estimate(np.zeros((10, 3)))


if __name__ == '__main__':
    unittest.main()
