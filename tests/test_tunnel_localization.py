"""Pure geometric registration and tracking; no ROS or robot transport."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from tunnel.geometry import Pose2D, normalize_angle
from tunnel.localization import WallLocalizer


WALLS = json.loads((ROOT / 'codes/stepbystep/maps/tunnel_practice_1820.geometry.json')
                   .read_text())['wall_rectangles']


def wall_cloud(pose, bins=1440, walls=None):
    """Independent ray/rectangle fixture, returning base-frame hit endpoints."""
    sensor = np.array((-.033073, 0.))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    origin = np.array(pose[:2]) + np.array(((c, -s), (s, c))) @ sensor
    angles = -math.pi+(np.arange(bins)+.5)*2*math.pi/bins
    directions = np.column_stack((np.cos(angles+pose[2]), np.sin(angles+pose[2])))
    ranges = np.full(bins, np.inf)
    for rectangle in (WALLS if walls is None else walls).values():
        low = (np.array(rectangle[:2])-origin)/directions
        high = (np.array(rectangle[2:])-origin)/directions
        near = np.minimum(low, high).max(axis=1)
        far = np.maximum(low, high).min(axis=1)
        hits = far >= np.maximum(near, 0.)
        distances = np.where(near >= 0., near, far)
        ranges = np.minimum(ranges, np.where(hits, distances, np.inf))
    hit = np.isfinite(ranges) & (ranges >= .05) & (ranges <= 3.)
    return sensor + ranges[hit, None]*np.column_stack((np.cos(angles[hit]), np.sin(angles[hit])))


def noisy_wall_cloud(pose):
    """Raycast wall endpoints with reproducible Cartesian measurement noise."""
    cloud = wall_cloud(pose)
    return cloud + np.random.default_rng(615).normal(0., .016, cloud.shape)


class WallGeometryTest(unittest.TestCase):
    def setUp(self):
        self.matcher = WallLocalizer(WALLS)

    def assert_pose(self, result, expected, position=.006, yaw=.004):
        self.assertTrue(result.accepted, result)
        self.assertLess(math.hypot(result.pose.x-expected[0], result.pose.y-expected[1]), position)
        self.assertLess(abs(result.pose.yaw-expected[2]), yaw)

    def test_wall_visibility_uses_sensor_origin_when_base_passes_wall_end(self):
        matcher = WallLocalizer([[.15, .2, .2, 1.2], [.15, 1.2, 1.5, 1.25]])
        pose = np.array((.17, 0., 0.))
        points = np.column_stack((np.full(40, -.02), np.linspace(.25, 1.1, 40)))
        self.assertGreater(matcher._score(points, pose), .001)
        self.assertLess(matcher._score(points, pose, (-.033073, 0.)), 1e-15)

        rotated = WallLocalizer([[.2, .15, 1.2, .2], [1.2, .15, 1.25, 1.5]])
        rotated_points = points * (1., -1.)
        self.assertLess(rotated._score(rotated_points, (0., .17, math.pi/2),
                                       (-.033073, 0.)), 1e-15)

    def test_room_walls_do_not_supply_a_valid_tunnel_match(self):
        room = np.vstack((np.column_stack((np.full(150, 2.3), np.linspace(.4, 1.5, 150))),
                          np.column_stack((np.linspace(.5, 2.3, 150), np.full(150, .4)))))
        self.assertFalse(self.matcher.track(room, (0., 0., 0.)).accepted)

    def test_invalid_geometry_and_input(self):
        with self.assertRaises(ValueError):
            WallLocalizer([[0., 0., 0., 1.]])
        with self.assertRaises(ValueError):
            WallLocalizer([[0., 0., .05, 2.]])
        with self.assertRaises(ValueError):
            self.matcher.track(np.zeros((10, 3)), (0., 0., 0.))

    def test_coarse_distances_match_finite_segment_projection(self):
        # Independently project onto the actual finite line segments.  Include
        # thick rectangles contributing both horizontal and vertical pairs,
        # endpoints, exterior points and batched coarse-search queries.
        rng = np.random.default_rng(20261026)
        for walls in (WALLS, [[0., 0., .5, 2.], [.5, 2., 2.5, 2.5]]):
            matcher = WallLocalizer(walls)
            points = rng.uniform(-1., 3., (7, 19, 2))
            expected = np.full(points.shape[:-1], np.inf)
            for axis, normal, low, high in matcher.segments:
                axis = int(axis)
                start, end = np.zeros(2), np.zeros(2)
                start[axis] = end[axis] = normal
                start[1-axis], end[1-axis] = low, high
                direction = end-start
                fraction = np.sum((points-start)*direction, axis=-1)/np.dot(direction, direction)
                projection = start+np.clip(fraction, 0., 1.)[..., None]*direction
                squared = np.sum((points-projection)**2, axis=-1)
                expected = np.minimum(expected, squared)
            np.testing.assert_allclose(matcher._distances(points, nearest=False),
                                       expected, rtol=1e-14, atol=1e-14)
            # The optimized minimum must also remain bit-exact with the
            # original nearest-face calculation used by refinement/support.
            for query in (points, points.reshape((-1, 2)), points[0, 0],
                          np.empty((0, 2))):
                np.testing.assert_array_equal(matcher._distances(query, nearest=False),
                                              matcher._distances(query)[0])

    def test_nearest_segment_ties_keep_first_face(self):
        matcher = WallLocalizer([[0., 0., .5, 2.], [.5, 2., 2.5, 2.5]])
        points = np.array(((.25, 1.), (-.25, -.5), (0., 0.)))
        square, indices = matcher._distances(points)
        np.testing.assert_array_equal(indices, (0, 0, 0))
        np.testing.assert_array_equal(square, (.0625, .3125, 0.))
        np.testing.assert_array_equal(matcher._distances(points, nearest=False), square)

    def test_thick_gazebo_walls_correct_near_entrance_poses(self):
        walls = json.loads((ROOT / 'tests/fixtures/autorace_gazebo.geometry.json')
                           .read_text())['wall_rectangles']
        matcher = WallLocalizer(walls)
        for pose in ((0., 0., 0.), (-.2, .035, .17), (.3, -.035, -.2)):
            with self.subTest(pose=pose):
                prior = np.add(pose, (.02, -.015, .025))
                self.assert_pose(matcher.track(wall_cloud(pose, bins=720, walls=walls), prior,
                                               sensor_origin=(-.033073, 0.)), pose)

    def test_opposite_face_is_not_visible_through_a_solid_wall(self):
        matcher = WallLocalizer([[0., 0., .05, 2.], [.05, 2., 2., 2.05]])
        point = np.array([[.051, 1.]])
        # The east face is closest but invisible from the west side.
        squared, segment = matcher._distances(point, origins=np.array([-.2, 1.]))
        self.assertEqual(segment[0], 0)
        self.assertAlmostEqual(squared[0], .051**2)
        origins = np.array([[-.2, 1.], [.2, 1.]])
        queries = np.repeat(point[None, :, :], 2, axis=0)
        expected = np.array([[.051**2], [.001**2]])
        np.testing.assert_allclose(matcher._distances(queries, origins=origins)[0], expected)
        np.testing.assert_allclose(matcher._distances(queries, nearest=False, origins=origins), expected)


class WallTrackingTest(unittest.TestCase):
    def setUp(self):
        self.matcher = WallLocalizer(WALLS)

    def assert_pose(self, result, expected, position=.006, yaw=.004):
        self.assertTrue(result.accepted, result)
        self.assertLess(math.hypot(result.pose.x-expected[0], result.pose.y-expected[1]),
                        position)
        self.assertLess(abs(normalize_angle(result.pose.yaw-expected[2])), yaw)

    def test_tracking_rejects_high_rms_wall_scan(self):
        pose = (.04, -.02, -.08)
        result = self.matcher.track(noisy_wall_cloud(pose), pose,
                                    sensor_origin=(-.033073, 0.))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)
        self.assertGreater(result.rms, .012)
        self.assertEqual(result.reason, 'wall residual too large')

    def test_inside_corner_recovers_xy_and_yaw_for_both_turn_directions(self):
        for actual, error in (((1.71, .12, .32), (.045, -.025, .065)),
                              ((1.75, 1.44, 1.35), (-.04, .03, -.07)),
                              ((.45, .85, -2.9), (.03, -.04, .06))):
            with self.subTest(actual=actual):
                prior = Pose2D(*np.add(actual, error))
                self.assert_pose(self.matcher.track(wall_cloud(actual), prior), actual)

    def test_rear_half_plane_and_yaw_wrap_remain_valid_wall_evidence(self):
        actual = (1.4, .8, math.pi-.025)
        cloud = wall_cloud(actual)
        cloud = cloud[cloud[:, 0] < 0.]
        prior = (actual[0]-.025, actual[1]+.025, -math.pi+.03)
        self.assert_pose(self.matcher.track(cloud, prior), actual)

    def test_out_of_bounds_xy_and_yaw_corrections_are_rejected(self):
        actual = (1.7, .14, .45)
        cloud = wall_cloud(actual)
        for error in ((.18, 0., 0.), (0., -.18, 0.),
                      (0., 0., .25), (.065, .065, 0.)):
            with self.subTest(error=error):
                result = self.matcher.track(cloud, np.add(actual, error))
                self.assertFalse(result.accepted, result)
                self.assertIsNone(result.pose)

    def test_custom_bounds_limit_corrections(self):
        actual = (1.72, 1.4, 1.3)
        cloud = wall_cloud(actual)
        self.assertFalse(self.matcher.track(cloud, (1.76, 1.4, 1.3),
                                            max_translation=.02).accepted)
        self.assertFalse(self.matcher.track(cloud, (1.72, 1.4, 1.35),
                                            max_yaw=.02).accepted)

    def test_single_wall_and_bottle_clusters_do_not_supply_two_wall_support(self):
        prior = np.array((1.7, .12, 0.))
        far = np.column_stack((np.full(130, 1.958), np.linspace(-.05, 1.5, 130)))
        angle = np.linspace(0., 2.*math.pi, 200)
        bottles = np.vstack([np.column_stack((x+.025*np.cos(angle),
                                               -.125+.025*np.sin(angle)))
                             for x in (.6, .9, 1.2, 1.5)])
        for mapped in (far, bottles, np.vstack((far, bottles))):
            with self.subTest(points=len(mapped)):
                result = self.matcher.track(mapped-prior[:2], prior)
                self.assertFalse(result.accepted, result)
                self.assertIsNone(result.pose)

    def test_noisy_scan_with_unmapped_obstacles_still_corrects(self):
        actual = (1.72, 1.4, 1.3)
        rng = np.random.default_rng(20261029)
        cloud = wall_cloud(actual)
        cloud += rng.normal(0., .003, cloud.shape)
        obstacles = rng.uniform((-1.8, -1.8), (1.8, 1.8), (200, 2))
        result = self.matcher.track(np.vstack((cloud, obstacles)), (1.75, 1.37, 1.35))
        self.assert_pose(result, actual, position=.008, yaw=.006)

    def test_competing_symmetric_local_matches_are_rejected(self):
        prior = (1., .75, .4)
        clouds = (wall_cloud((prior[0]-.05, prior[1], prior[2])),
                  wall_cloud((prior[0]+.05, prior[1], prior[2])))
        result = self.matcher.track(np.vstack(clouds), prior)
        self.assertFalse(result.accepted, result)
        self.assertIsNone(result.pose)
        self.assertEqual(result.reason, 'ambiguous tracking wall alignment')

    def test_many_duplicate_returns_cannot_dominate_or_grow_fit(self):
        actual = (1.7, .14, .45)
        cloud = wall_cloud(actual, bins=14400)
        repeated_obstacle = np.repeat(np.array(((.4, .3),)), 20000, axis=0)
        result = self.matcher.track(np.vstack((cloud, repeated_obstacle)),
                                    (1.74, .12, .5))
        self.assert_pose(result, actual)
        self.assertLessEqual(result.total, 500)

    def test_empty_and_nonfinite_scans_and_invalid_parameters(self):
        for cloud in (np.empty((0, 2)), np.full((100, 2), np.nan)):
            self.assertFalse(self.matcher.track(cloud, (1., .7, .4)).accepted)
        for kwargs in ({'max_translation': 0.}, {'max_translation': math.inf},
                       {'max_yaw': 0.}, {'max_yaw': math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.matcher.track(np.empty((0, 2)), (1., .7, .4), **kwargs)
        for prior in ((1., 2.), (1., 2., math.inf)):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                self.matcher.track(np.empty((0, 2)), prior)
        with self.assertRaises(ValueError):
            self.matcher.track(np.zeros((10, 3)), (1., .7, .4))


if __name__ == '__main__':
    unittest.main()
