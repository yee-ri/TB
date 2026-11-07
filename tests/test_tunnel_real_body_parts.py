"""Configured hardware coverage and the recorded entrance rotation geometry."""

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))

from tunnel.geometry import CollisionGeometry, Pose2D, RectangularFootprint
from tunnel.mission import TunnelMission, load_config
from tunnel.planner import load_navigation_settings
from test_tunnel_body_parts import point_obstacle, world_point


CONFIGS = tuple(
    ROOT / 'codes' / directory / name
    for directory in ('stepbystep', 'final_codes')
    for name in ('tunnel.real.yaml', 'tunnel.competition.yaml')
)


def configured_geometry(path):
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    return CollisionGeometry(RectangularFootprint(**config['footprint']))


class RealBodyPartsTest(unittest.TestCase):
    def test_real_profiles_keep_full_hardware_extents_without_added_padding(self):
        for path in CONFIGS:
            with self.subTest(config=str(path.relative_to(ROOT))):
                body = configured_geometry(path).footprint
                self.assertTrue(body.parts)
                self.assertEqual(body.padding, 0.)
                self.assertAlmostEqual(min(p[0] for p in body.parts), -.118073)
                self.assertAlmostEqual(max(p[2] for p in body.parts), .0678, places=6)
                self.assertAlmostEqual(min(p[1] for p in body.parts), -.0903)
                self.assertAlmostEqual(max(p[3] for p in body.parts), .0903)

    def test_chassis_camera_and_outer_wheel_hardware_all_block_obstacles(self):
        # The side probes extend beyond the Gazebo tire width (79.75 mm).
        material = ((-.11805, 0.), (.06762, .05), (-.10, .06108),
                    (-.10, -.06108), (.06772, 0.),
                    (0., .09025), (0., -.09025),
                    (-.04770, .088), (.04770, .088),
                    (-.04770, -.088), (.04770, -.088))
        for path in CONFIGS:
            checker = configured_geometry(path)
            for yaw in (0., .61, math.pi / 2., -math.pi / 2.):
                pose = Pose2D(.2, -.2, yaw)
                for point in material:
                    with self.subTest(config=path, yaw=yaw, local_point=point):
                        grid = point_obstacle(world_point(pose, point), half_size=.00001)
                        self.assertFalse(checker.pose_is_collision_free(grid, pose))

    def test_empty_rear_corners_are_not_treated_as_body_material(self):
        for path in CONFIGS:
            checker = configured_geometry(path)
            envelope = CollisionGeometry(replace(checker.footprint, parts=()))
            for yaw in (0., .61, math.pi / 2., -math.pi / 2.):
                pose = Pose2D(.2, -.2, yaw)
                for side in (-1., 1.):
                    with self.subTest(config=path, yaw=yaw, side=side):
                        grid = point_obstacle(world_point(pose, (-.11, side * .085)))
                        self.assertTrue(checker.pose_is_collision_free(grid, pose))
                        self.assertFalse(envelope.pose_is_collision_free(grid, pose))

    def test_clear_points_outside_measured_surfaces_do_not_gain_padding(self):
        for path in CONFIGS:
            checker = configured_geometry(path)
            for point in ((-.1186, 0.), (.0683, 0.), (-.10, .0616),
                          (-.10, -.0616), (0., .0908), (0., -.0908)):
                with self.subTest(config=path, point=point):
                    self.assertTrue(checker.pose_is_collision_free(
                        point_obstacle(point, half_size=.00001), Pose2D(0., 0., 0.)))

    def test_mid_rotation_geometry_detects_overlap_between_free_endpoints(self):
        grid = point_obstacle((-.117, 0.), half_size=.0005)
        start = Pose2D(0., 0., -math.pi / 4.)
        finish = Pose2D(0., 0., math.pi / 4.)
        for path in CONFIGS:
            with self.subTest(config=path):
                checker = configured_geometry(path)
                self.assertTrue(checker.pose_is_collision_free(grid, start))
                self.assertTrue(checker.pose_is_collision_free(grid, finish))
                self.assertFalse(checker.pose_is_collision_free(grid, Pose2D(0., 0., 0.)))

    def test_recorded_entrance_static_wall_rotation_is_clear_for_configured_parts(self):
        # Rounded Nov 2 reconstructed pose and target; this is not a sensor replay.
        pose = Pose2D(.22014, -.029996, math.radians(2.5465))
        angle = math.radians(22.0288)
        # Compare the four body configurations on the same recorded-run map.
        mission = TunnelMission(load_config(ROOT / 'codes/stepbystep/tunnel.real.yaml'))
        try:
            mission._initialize((0., 0., 0.))
            grid = mission._grid()
            self.assertTrue(mission.static_cell_rectangles)
            for path in CONFIGS:
                with self.subTest(config=path):
                    checker = configured_geometry(path)
                    samples = [Pose2D(pose.x, pose.y, pose.yaw+offset)
                               for offset in np.linspace(0., angle, 221)]
                    self.assertTrue(all(checker.pose_is_collision_free(grid, sample)
                                        for sample in samples))
                    envelope = CollisionGeometry(replace(checker.footprint, parts=()))
                    self.assertFalse(all(envelope.pose_is_collision_free(grid, sample)
                                         for sample in samples))
        finally:
            mission.close()

    def test_body_configuration_does_not_restore_small_angle_alignment(self):
        for path in CONFIGS:
            with self.subTest(config=path):
                config = yaml.safe_load(path.read_text(encoding='utf-8'))
                navigation = load_navigation_settings(config.get('navigation'))
                self.assertEqual(navigation['route_change_alignment_angle_deg'], 90.)


if __name__ == '__main__':
    unittest.main()
