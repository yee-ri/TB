"""Offline unit conversion at the tunnel adapter boundary; no ROS transport."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCALE = .0955 / .033


def implementations():
    for folder in ('stepbystep', 'final_codes'):
        path = ROOT / 'codes/tunnel/odometry.py'
        spec = importlib.util.spec_from_file_location('odometry_test_' + folder, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield folder, module


class TunnelOdometryTests(unittest.TestCase):
    def test_default_and_physical_profiles_have_separate_unit_contracts(self):
        for folder, module in implementations():
            with self.subTest(folder=folder):
                self.assertEqual(module.read_linear_scale({}), 1.)
                self.assertEqual(module.read_linear_scale({'odometry': {}}), 1.)
                self.assertEqual(module.read_angular_command_scale({}), 1.)
                self.assertEqual(module.read_angular_command_scale({'odometry': {}}), 1.)
                separate = {'odometry': {'linear_scale': SCALE, 'angular_command_scale': 3.}}
                self.assertAlmostEqual(module.read_linear_scale(separate), SCALE)
                self.assertEqual(module.read_angular_command_scale(separate), 3.)
                for filename in ('tunnel.yaml', 'tunnel.competition.yaml', 'tunnel.real.yaml'):
                    path = ROOT / 'codes' / folder / filename
                    if not path.exists():
                        continue
                    config = yaml.safe_load(path.read_text())
                    expected = SCALE if filename == 'tunnel.real.yaml' else 1.
                    self.assertAlmostEqual(module.read_linear_scale(config), expected)
                    expected_angular = 3.3 if filename == 'tunnel.real.yaml' else 1.
                    self.assertEqual(module.read_angular_command_scale(config), expected_angular)

    def test_invalid_scales_fail_instead_of_changing_units_silently(self):
        invalid = (None, True, False, np.bool_(True), '2.9', 0., -1.,
                   math.nan, math.inf, -math.inf, [], {})
        for folder, module in implementations():
            for key, read in (('linear_scale', module.read_linear_scale),
                              ('angular_command_scale', module.read_angular_command_scale)):
                for value in invalid:
                    with self.subTest(folder=folder, key=key, value=value), self.assertRaises(ValueError):
                        read({'odometry': {key: value}})

    def test_pose_conversion_preserves_heading_and_does_not_mutate_input(self):
        pose = np.array((1.25, -.375, 2.8))
        for folder, module in implementations():
            with self.subTest(folder=folder):
                converted = module.scale_pose(pose, SCALE)
                np.testing.assert_allclose(converted, (1.25 * SCALE, -.375 * SCALE, 2.8))
                np.testing.assert_array_equal(pose, (1.25, -.375, 2.8))
                np.testing.assert_allclose(module.scale_pose(converted, 1. / SCALE), pose)
                np.testing.assert_array_equal(module.scale_pose(pose, 1.), pose)


if __name__ == '__main__':
    unittest.main()
