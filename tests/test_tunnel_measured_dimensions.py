"""Measured tunnel geometry, map loading, and synthetic sensor checks."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'scripts')]

from build_practice_map import build
from tunnel.geometry import Pose2D
from tunnel.mission import TunnelMission, load_config
from test_tunnel_localization import wall_cloud


MAPS = ROOT / 'codes/stepbystep/maps'
CONFIG = ROOT / 'codes/stepbystep/tunnel.real.yaml'


class MeasuredTunnelTest(unittest.TestCase):
    def mission(self, config=CONFIG):
        mission = TunnelMission(load_config(config))
        self.addCleanup(mission.close)
        mission._initialize((0., 0., 0.))
        return mission

    def test_outer_size_wall_thickness_and_clear_openings(self):
        survey = json.loads((MAPS / 'tunnel_1800x1820.geometry.json').read_text())
        x0, y0, x1, y1 = survey['outer_bounds']
        self.assertAlmostEqual(x1 - x0, 1.820)  # Depth, not entrance-facing width.
        self.assertAlmostEqual(y1 - y0, 1.800)
        ix0, iy0, ix1, iy1 = survey['inner_clear_bounds']
        np.testing.assert_allclose([ix0-x0, iy0-y0, x1-ix1, y1-iy1], .010)
        self.assertAlmostEqual(ix1 - ix0, 1.800)
        self.assertAlmostEqual(iy1 - iy0, 1.780)
        for name in ('entry_clear_segment', 'exit_clear_segment'):
            segment = np.asarray(survey[name])
            self.assertAlmostEqual(np.linalg.norm(segment[1] - segment[0]), .300)

    def test_checked_in_raster_and_survey_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            build(directory)
            for suffix in ('.pgm', '.geometry.json'):
                name = 'tunnel_1800x1820' + suffix
                self.assertEqual((Path(directory) / name).read_bytes(),
                                 (MAPS / name).read_bytes())

    def test_both_real_configs_load_matching_walls_and_goal(self):
        survey = json.loads((MAPS / 'tunnel_1800x1820.geometry.json').read_text())
        for folder in ('stepbystep', 'final_codes'):
            with self.subTest(folder=folder):
                mission = self.mission(ROOT / 'codes' / folder / 'tunnel.real.yaml')
                self.assertEqual(mission.config['planner']['keep_in_rectangles'],
                                 survey['keep_in_rectangles'])
                np.testing.assert_allclose(mission.config['goal']['pose'], [1.800, 1.840, 90.])
                grid = mission._grid()
                walls = grid.occupied_cell_geometry(0, grid.width - 1, 0, grid.height - 1)
                self.assertGreater(len(walls), 0)
                # Coarse 20 mm cells must retain the physical 10 mm wall thickness.
                self.assertTrue(np.all(np.min(2 * walls[:, 2:], axis=1) <= .010 + 1e-9))
                np.testing.assert_allclose(
                    np.r_[np.min(walls[:, :2] - walls[:, 2:], axis=0),
                          np.max(walls[:, :2] + walls[:, 2:], axis=0)],
                    survey['outer_bounds'])
                self.assertAlmostEqual(mission.footprint.half_width, .0903)

    def test_entrance_fit_uses_the_new_opposite_wall(self):
        mission = self.mission()
        self.assertAlmostEqual(mission.entrance_localizer.opposite_side_y, 1.630)
        walls = mission.config['localization']['wall_rectangles']
        for pose in ((.235, -.022, 0.), (.235, -.022, math.radians(5)), (.280, 0., 0.)):
            with self.subTest(pose=pose):
                result = mission.entrance_localizer.estimate(
                    wall_cloud(pose, walls=walls), (-.033073, 0.))
                self.assertTrue(result.accepted, result.reason)
                self.assertLess(math.hypot(result.pose.x-pose[0], result.pose.y-pose[1]), .003)
                self.assertLess(abs(result.pose.yaw-pose[2]), math.radians(.2))

    def test_exit_fit_uses_the_new_ten_mm_jambs(self):
        mission = self.mission()
        walls = mission.config['localization']['wall_rectangles']
        for pose in ((1.8, 1.35, math.pi/2), (1.55, 1.35, math.pi/2)):
            with self.subTest(pose=pose):
                result = mission.exit_localizer.track(
                    wall_cloud(pose, walls=walls), pose, .08, math.radians(6), (-.033073, 0.))
                self.assertTrue(result.accepted, result.reason)
                self.assertLess(math.hypot(result.pose.x-pose[0], result.pose.y-pose[1]), .006)
                self.assertLess(abs(result.pose.yaw-pose[2]), math.radians(.5))

    def test_global_routes_reach_inside_and_outside_goals(self):
        mission = self.mission()
        np.testing.assert_allclose(
            [mission.exit_approach_goal.x, mission.exit_approach_goal.y], [1.800, 1.452])
        grid = mission._grid()
        stages = ((Pose2D(.235, -.022, 0.), mission.exit_approach_goal),
                  (mission.exit_approach_goal, mission.goal))
        for start, goal in stages:
            with self.subTest(goal=goal):
                route = mission.local_planner.global_route(grid, start, goal)
                self.assertIsNotNone(route)
                np.testing.assert_allclose(route.points[-1], [goal.x, goal.y])


if __name__ == '__main__':
    unittest.main()
