import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from tunnel.entrance_localization import EntranceLandmarkLocalizer
from tunnel.geometry import Pose2D
from tunnel.mission import map_anchor, relative_pose


def base_points(mapped, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (np.asarray(mapped) - pose[:2]) @ np.array(((c, -s), (s, c)))


def wall(pose, first=.14, last=.50, count=501):
    return base_points(np.column_stack((np.linspace(first, last, count),
                                       np.full(count, -.15))), pose)


class CornerSideTests(unittest.TestCase):
    def check_pose(self, result, expected, tolerance=1e-8):
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose([result.pose.x, result.pose.y, result.pose.yaw],
                                   expected, atol=tolerance, rtol=0.)

    def test_only_one_side_wall_and_its_endpoint_are_sufficient(self):
        pose = np.array([.226, -.016, math.radians(1.68)])
        self.check_pose(EntranceLandmarkLocalizer().estimate(wall(pose), (-.033073, 0.)), pose)

    def test_geometry_does_not_require_entry_cross_wall_or_second_endpoint(self):
        geometry = {'wall_rectangles': {'right_side_wall': [.14, -.152, 1.94, -.15]}}
        pose = np.array([.213, -.022, 0.])
        self.check_pose(EntranceLandmarkLocalizer.from_geometry(geometry).estimate(
            wall(pose), (-.033073, 0.)), pose)

    def test_observed_geometry_overrides_nearby_prior(self):
        pose = np.array([.228, -.009, math.radians(-3.44)])
        for prior in [(.213, -.022, 0.), (.220, -.020, -.02), (.205, -.017, .01)]:
            with self.subTest(prior=prior):
                self.check_pose(EntranceLandmarkLocalizer(prior_pose=prior).estimate(
                    wall(pose), (-.033073, 0.)), pose)

    def test_sensor_offset_and_heading_are_respected(self):
        for pose in [(.202, -.030, -.06), (.228, -.009, .038)]:
            for origin in [(-.033073, 0.), (-.05, .01), (.01, -.02)]:
                with self.subTest(pose=pose, origin=origin):
                    self.check_pose(EntranceLandmarkLocalizer().estimate(wall(np.array(pose)), origin), pose)

    def test_empty_nonfinite_or_degenerate_input_never_returns_prior(self):
        for points in [np.empty((0, 2)), np.full((10, 2), np.nan), np.tile([0., -.13], (20, 1))]:
            with self.subTest(points=len(points)):
                result = EntranceLandmarkLocalizer().estimate(points)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.pose)

    def test_result_outside_prior_neighbourhood_is_accepted(self):
        pose = np.array([.263, -.022, 0.])
        self.check_pose(EntranceLandmarkLocalizer().estimate(wall(pose), (-.033073, 0.)), pose)

    def test_short_observed_side_with_entrance_end_is_sufficient(self):
        pose = np.array([.226, -.016, .03])
        self.check_pose(EntranceLandmarkLocalizer().estimate(
            wall(pose, last=.24, count=21), (-.033073, 0.)), pose)

    def test_prior_longitudinal_error_does_not_crop_the_observed_wall_end(self):
        for shift in [.05, .10, -.05, -.10]:
            with self.subTest(shift=shift):
                pose = np.array([.213 + shift, -.022, 0.])
                estimator = EntranceLandmarkLocalizer(prior_pose=(.213, -.022, 0.))
                self.check_pose(estimator.estimate(wall(pose), (-.033073, 0.)), pose)

    def test_absent_side_wall_cannot_use_cross_wall(self):
        pose = np.array([.213, -.022, 0.])
        cross_wall = base_points(np.column_stack((np.full(201, .142), np.linspace(.15, .65, 201))), pose)
        result = EntranceLandmarkLocalizer().estimate(cross_wall, (-.033073, 0.))
        self.assertFalse(result.accepted)

    def test_range_noise_endpoint_uses_ray_and_line_not_minimum_point(self):
        pose = np.array([.226, -.016, .03])
        origin = np.array([-.033073, 0.])
        clean = wall(pose, last=.34, count=2000)
        delta = clean - origin
        unit = delta / np.linalg.norm(delta, axis=1)[:, None]
        # A few endpoint returns are longer while their bearings remain valid.
        noisy = clean.copy()
        noisy[:8] += .018 * unit[:8]
        estimator = EntranceLandmarkLocalizer()
        result = estimator.estimate(noisy, origin)
        self.check_pose(result, pose, tolerance=.002)
        raw_end = estimator.diagnostics['raw_endpoint_along']
        ray_end = estimator.diagnostics['ray_endpoint_along']
        self.assertGreater(abs(raw_end-ray_end), .005)

        ignored_origin = EntranceLandmarkLocalizer().estimate(noisy, (0., 0.))
        self.assertGreater(abs(result.pose.x-ignored_origin.pose.x), .0005)

    def test_scan_anchor_propagates_new_odom_without_relabeling_scan_pose(self):
        scan_odom = (.8, -.4, .2)
        fitted = Pose2D(.226, -.016, .03)
        anchor = map_anchor(scan_odom, fitted)
        recovered = relative_pose(scan_odom, anchor)
        np.testing.assert_allclose([recovered.x, recovered.y, recovered.yaw],
                                   [fitted.x, fitted.y, fitted.yaw], atol=1e-12)
        advanced = relative_pose((scan_odom[0]+.1*math.cos(scan_odom[2]),
                                  scan_odom[1]+.1*math.sin(scan_odom[2]), scan_odom[2]), anchor)
        np.testing.assert_allclose([advanced.x, advanced.y],
            [fitted.x+.1*math.cos(fitted.yaw), fitted.y+.1*math.sin(fitted.yaw)], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
