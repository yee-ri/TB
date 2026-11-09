import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from tunnel.exit_localization import ExitLandmarkLocalizer
from tunnel.geometry import Pose2D
from tunnel.localization import WallLocalizer
from test_tunnel_alignment_mission import fixed_wall_cloud, SENSOR


GEOMETRY = json.loads((ROOT / 'codes/stepbystep/maps/tunnel_practice_1820.geometry.json').read_text())


def base_points(mapped, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (np.asarray(mapped)-pose[:2]) @ np.array(((c, -s), (s, c)))


def visible_exit(pose, count=121, length=.4):
    left = np.column_stack((np.linspace(1.658-length, 1.658, count), np.full(count, 1.666)))
    right = np.column_stack((np.full(count, 1.958), np.linspace(1.668-length, 1.668, count)))
    return base_points(np.concatenate((left, right)), pose)


class ExitLandmarkTests(unittest.TestCase):
    def setUp(self):
        self.localizer = ExitLandmarkLocalizer.from_geometry(GEOMETRY)
        self.pose = np.array([1.78, 1.35, math.radians(84.)])

    def assert_pose(self, result, expected, tolerance=1e-8):
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose([result.pose.x, result.pose.y, result.pose.yaw],
                                   expected, atol=tolerance, rtol=0.)

    def test_geometry_uses_actual_visible_inner_faces(self):
        np.testing.assert_allclose(self.localizer.landmarks, [[1.658, 1.666], [1.958, 1.668]])
        np.testing.assert_allclose(self.localizer.directions, [[-1., 0.], [0., -1.]])

    def test_two_observed_jambs_correct_both_position_axes_and_yaw(self):
        prior = self.pose+np.array([-.04, .03, math.radians(4.)])
        self.assert_pose(self.localizer.track(visible_exit(self.pose), prior), self.pose)
        self.assertAlmostEqual(self.localizer.diagnostics['observed_width'], math.hypot(.3, .002))

    def test_pose_object_and_sensor_offset(self):
        prior = Pose2D(self.pose[0]+.03, self.pose[1]-.02, self.pose[2]-.03)
        for origin in [(-.033073, 0.), (.02, -.01)]:
            with self.subTest(origin=origin):
                self.assert_pose(self.localizer.track(visible_exit(self.pose), prior,
                                                     sensor_origin=origin), self.pose)

    def test_range_noise_with_clutter_still_corrects_pose(self):
        rng = np.random.RandomState(11)
        clean = visible_exit(self.pose, count=601)
        origin = np.array([-.033073, 0.])
        rays = clean-origin
        noise = rng.normal(0., .002, (len(clean), 1))
        noisy = clean+noise*rays/np.linalg.norm(rays, axis=1)[:, None]
        clutter = base_points(rng.uniform([.3, .1], [1.1, .9], (200, 2)), self.pose)
        prior = self.pose+np.array([.03, -.025, .04])
        self.assert_pose(self.localizer.track(np.concatenate((noisy, clutter)), prior,
                                              sensor_origin=origin), self.pose, .004)

    def test_short_exit_faces_correct_pose_without_extended_wall_support(self):
        pose = np.array([1.78, 1.35, 1.4])
        origin = np.array([-.033073, 0.])
        points = visible_exit(pose, count=101, length=.10)
        rays = points-origin
        points += np.random.RandomState(3).normal(0., .001, (len(points), 1))*rays/np.linalg.norm(rays, axis=1)[:, None]
        prior = pose+np.array([.025, -.02, .04])
        generic = WallLocalizer(GEOMETRY['wall_rectangles'])
        self.assertFalse(generic.track(points, prior, sensor_origin=origin).accepted)
        # Relax only point counts to isolate the generic 25cm extent requirement.
        extent_only = WallLocalizer(GEOMETRY['wall_rectangles'], minimum_inliers=6,
                                   minimum_axis_inliers=3)
        rejected = extent_only.track(points, prior, sensor_origin=origin)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, 'no extended perpendicular wall support')
        fitted = self.localizer.track(points, prior, sensor_origin=origin)
        self.assertTrue(fitted.accepted, fitted.reason)
        self.assertLess(np.linalg.norm([fitted.pose.x-pose[0], fitted.pose.y-pose[1]]), .003)
        self.assertLess(abs(fitted.pose.yaw-pose[2]), math.radians(.3))

    def test_short_noisy_wall_does_not_override_long_wall_direction(self):
        localizer = ExitLandmarkLocalizer([[0., 0.], [.3, 0.]], [[-1., 0.], [0., -1.]])
        pose = np.array([-.3, -.65, 0.])
        for seed in range(10):
            rng = np.random.RandomState(seed)
            left = np.column_stack((np.linspace(-.8, 0., 160), rng.normal(0., .001, 160)))
            right = np.column_stack((.3+rng.normal(0., .002, 20), np.linspace(-.06, 0., 20)))
            points = base_points(np.concatenate((left, right)), pose)
            with self.subTest(seed=seed):
                result = localizer.track(points, pose+[.02, .03, .03])
                self.assertTrue(result.accepted, result.reason)
                self.assertLess(np.linalg.norm([result.pose.x-pose[0], result.pose.y-pose[1]]), .002)
                self.assertLess(abs(result.pose.yaw-pose[2]), math.radians(.1))

    def test_connected_l_shaped_group_uses_wall_line_not_whole_cluster_pca(self):
        pose = np.array([1.70, 1.20, math.pi/2])
        points = visible_exit(pose, count=201, length=.8)
        bottom = np.column_stack((np.linspace(1.658-.8, 1.958, 301), np.full(301, 1.668-.8)))
        connector = np.column_stack((np.full(201, 1.658-.8), np.linspace(1.668-.8, 1.666, 201)))
        points = np.concatenate((points, base_points(bottom, pose), base_points(connector, pose)))
        self.assert_pose(self.localizer.track(points, pose+[.02, .02, .02]), pose, 1e-5)

    def test_one_jamb_missing_never_invents_other_from_known_width(self):
        points = visible_exit(self.pose)
        for subset in [points[:121], points[121:]]:
            with self.subTest(first=subset[0]):
                result = self.localizer.track(subset, self.pose)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.pose)

    def test_truncated_wall_does_not_pass_known_gap_width(self):
        points = visible_exit(self.pose)
        points = np.concatenate((points[:100], points[121:]))
        self.assertFalse(self.localizer.track(points, self.pose).accepted)

    def test_wrong_gap_width_and_direction_are_rejected(self):
        mapped = np.concatenate((np.column_stack((np.linspace(1.258, 1.658, 121), np.full(121, 1.666))),
                                 np.column_stack((np.full(121, 1.858), np.linspace(1.268, 1.668, 121)))))
        self.assertFalse(self.localizer.track(base_points(mapped, self.pose), self.pose).accepted)
        mapped[121:] = np.column_stack((np.linspace(1.958, 2.358, 121), np.full(121, 1.668)))
        self.assertFalse(self.localizer.track(base_points(mapped, self.pose), self.pose).accepted)

    def test_two_equally_plausible_openings_do_not_change_pose(self):
        first = visible_exit(self.pose, length=.15)
        second = visible_exit(self.pose+np.array([.05, .05, 0.]), length=.15)
        result = self.localizer.track(np.concatenate((first, second)),
                                      self.pose+np.array([.025, .025, 0.]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'ambiguous observed exit opening')
        self.assertIsNone(result.pose)

    def test_similar_opening_far_from_prior_is_not_applied(self):
        points = visible_exit(self.pose)
        result = self.localizer.track(points, self.pose+[.2, 0., 0.])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)

    def test_empty_invalid_and_degenerate_clouds_do_not_return_a_pose(self):
        for points in [np.empty((0, 2)), np.full((30, 2), np.nan), np.zeros((30, 2))]:
            with self.subTest(size=len(points)):
                result = self.localizer.track(points, self.pose)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.pose)

    def test_existing_translation_and_yaw_bounds_keep_visible_endpoints(self):
        for pose in (np.array([1.1, 1.35, .2]), np.array([1.808, 1.7, math.pi/2])):
            for origin in ((0., 0.), (-.033073, .02)):
                prior = pose+np.array([.079, 0., math.radians(5.9)])
                with self.subTest(pose=pose, origin=origin):
                    self.assert_pose(self.localizer.track(visible_exit(pose), prior,
                                                          sensor_origin=origin), pose)

    def test_point_order_does_not_change_fit(self):
        points = visible_exit(self.pose)
        rng = np.random.RandomState(7)
        for _ in range(3):
            rng.shuffle(points)
            self.assert_pose(self.localizer.track(points, self.pose+[.02, -.03, .02]), self.pose)

    def test_first_hit_raycast_handles_thin_perpendicular_exit_walls(self):
        for x in (1.72, 1.81, 1.90):
            for y in (1.20, 1.45, 1.60):
                pose = np.array([x, y, math.pi/2])
                points = fixed_wall_cloud(pose, GEOMETRY['wall_rectangles'])
                prior = pose+np.array([.025, -.02, math.radians(3.)])
                with self.subTest(x=x, y=y):
                    self.assert_pose(self.localizer.track(points, prior,
                                                        sensor_origin=SENSOR[:2]), pose, .009)

    def test_raycast_occluded_jamb_rejects_measurement_only(self):
        rectangles = dict(GEOMETRY['wall_rectangles'])
        rectangles['occluder'] = [1.55, 1.45, 1.72, 1.60]
        points = fixed_wall_cloud(self.pose, rectangles)
        result = self.localizer.track(points, self.pose, sensor_origin=SENSOR[:2])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.pose)

    def test_invalid_configuration_and_input_are_reported(self):
        for settings in [dict(cluster_distance=0), dict(minimum_points=2),
                         dict(sampling_resolution=.05), dict(maximum_width_error=math.inf)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExitLandmarkLocalizer.from_geometry(GEOMETRY, **settings)
        with self.assertRaises(ValueError):
            self.localizer.track(np.ones((10, 3)), self.pose)
        with self.assertRaises(ValueError):
            self.localizer.track(visible_exit(self.pose), [math.nan, 0., 0.])


if __name__ == '__main__':
    unittest.main()
