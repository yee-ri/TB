"""Exact surveyed walls retain padding and swept collision checks."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))

from tunnel.geometry import CollisionGeometry, OccupancyGrid, RectangularFootprint


class StaticWallGeometryTest(unittest.TestCase):
    def setUp(self):
        self.values = np.zeros((100, 100), dtype=np.int8)
        self.rectangles = {}
        # The physical right wall is 2 mm thick. Its raster cell extends
        # 16 mm farther into the lane than the measured wall face.
        for column in range(35, 66):
            self.values[47, column] = 100
            left = -1. + column * .02
            self.rectangles[47 * 100 + column] = [
                (left, -.058, left + .02, -.056)]
        self.planner = CollisionGeometry(
            RectangularFootprint(.08, .12, .04, .01),
            minimum_turning_radius=.2, collision_check_step=.02)

    def grid(self, **kwargs):
        return OccupancyGrid(self.values, .02, -1., -1.,
                             static_cell_rectangles=self.rectangles, **kwargs)

    def test_sixteen_mm_body_clearance_retains_ten_mm_padding(self):
        coarse = OccupancyGrid(self.values, .02, -1., -1.)
        refined = self.grid()
        self.assertFalse(self.planner.pose_is_collision_free(coarse, (0., 0., 0.)))
        self.assertTrue(self.planner.pose_is_collision_free(refined, (0., 0., 0.)))
        self.assertTrue(self.planner.primitive_is_collision_free(
            refined, (0., 0., 0.), 0., .1))
        self.assertEqual(self.planner.footprint.padding, .01)
        self.assertEqual(refined.resolution, .02)
        self.assertAlmostEqual(refined.collision_resolution, .002)

    def test_padding_contact_and_physical_contact_remain_collisions(self):
        refined = self.grid()
        for y in (-.006, -.016):
            with self.subTest(y=y):
                self.assertFalse(self.planner.pose_is_collision_free(refined, (0., y, 0.)))
        unpadded = CollisionGeometry(RectangularFootprint(.08, .12, .04))
        # A 5 mm body gap is clear for bare geometry but violates 10 mm padding.
        self.assertTrue(unpadded.pose_is_collision_free(refined, (0., -.011, 0.)))
        self.assertFalse(self.planner.pose_is_collision_free(refined, (0., -.011, 0.)))

    def test_thin_wall_crossing_between_free_endpoints_is_blocked(self):
        values = np.zeros_like(self.values)
        rectangles = {}
        for row in (49, 50):
            values[row, 55] = 100
            bottom = -1. + row * .02
            rectangles[row * 100 + 55] = [(.104, bottom, .106, bottom + .02)]
        grid = OccupancyGrid(values, .02, -1., -1.,
                             static_cell_rectangles=rectangles)
        planner = CollisionGeometry(
            RectangularFootprint(.0002, .0002, .0002),
            collision_check_step=.02)
        self.assertTrue(planner.pose_is_collision_free(grid, (.101, 0., 0.)))
        self.assertTrue(planner.pose_is_collision_free(grid, (.111, 0., 0.)))
        self.assertAlmostEqual(grid.collision_resolution, .002)
        self.assertFalse(planner.primitive_is_collision_free(grid, (.101, 0., 0.), 0., .010))

    def test_static_dynamic_and_unrefined_obstacles_coexist(self):
        self.values[50, 60] = self.values[50, 65] = 100
        grid = self.grid(dynamic_cell_rectangles={
            50 * 100 + 65: [(.315, 0., .32, .02)]})
        planner = CollisionGeometry(RectangularFootprint(.003, .003, .003))
        self.assertTrue(planner.pose_is_collision_free(grid, (.305, .01, 0.)))
        self.assertFalse(planner.pose_is_collision_free(grid, (.313, .01, 0.)))
        self.assertFalse(planner.pose_is_collision_free(grid, (.205, .01, 0.)))
        self.assertFalse(planner.pose_is_collision_free(grid, (0., -.056, 0.)))

    def test_explicit_inflation_keeps_conservative_coarse_geometry(self):
        planner = CollisionGeometry(self.planner.footprint, inflation_radius=.001)
        self.assertFalse(planner.pose_is_collision_free(self.grid(), (0., 0., 0.)))
        # Adjacent originally free cells are still included by grid inflation.
        planner = CollisionGeometry(self.planner.footprint, inflation_radius=.021)
        self.assertFalse(planner.pose_is_collision_free(self.grid(), (0., .025, 0.)))

    def test_unknown_cells_remain_fully_occupied(self):
        self.values[50, 50] = -1
        planner = CollisionGeometry(RectangularFootprint(.003, .003, .003))
        self.assertFalse(planner.pose_is_collision_free(self.grid(), (.01, .01, 0.)))
        self.assertTrue(planner.pose_is_collision_free(
            self.grid(unknown_is_occupied=False), (.01, .01, 0.)))
        for source in ('static_cell_rectangles', 'dynamic_cell_rectangles'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                OccupancyGrid(self.values, .02, -1., -1., **{
                    source: {50 * 100 + 50: [(.015, 0., .02, .02)]}})

    def test_same_cell_cannot_replace_static_with_dynamic_geometry(self):
        index = next(iter(self.rectangles))
        with self.assertRaisesRegex(ValueError, 'both static and dynamic'):
            self.grid(dynamic_cell_rectangles={index: self.rectangles[index]})

    def test_grid_preserves_both_refinement_sources(self):
        self.values[50, 65] = 100
        grid = OccupancyGrid(
            self.values, .02, -1., -1.,
            dynamic_cell_rectangles={50 * 100 + 65: [(.315, 0., .32, .02)]},
            static_cell_rectangles=self.rectangles)
        self.assertTrue(self.planner.pose_is_collision_free(grid, (0., 0., 0.)))
        planner = CollisionGeometry(RectangularFootprint(.003, .003, .003))
        self.assertTrue(planner.pose_is_collision_free(grid, (.305, .01, 0.)))
        self.assertFalse(planner.pose_is_collision_free(grid, (.313, .01, 0.)))

    def test_static_refinement_rejects_invalid_geometry(self):
        index = next(iter(self.rectangles))
        valid = self.rectangles[index][0]
        cases = (
            {index: []},
            {index: [(valid[0], valid[1], float('nan'), valid[3])]},
            {index: [(valid[0], valid[1], valid[0], valid[3])]},
            {index: [(valid[0] - .001, valid[1], valid[2], valid[3])]},
            {index - 100: [valid]},
            {self.values.size: [valid]},
        )
        for rectangles in cases:
            with self.subTest(rectangles=rectangles), self.assertRaises(ValueError):
                OccupancyGrid(self.values, .02, -1., -1.,
                              static_cell_rectangles=rectangles)

    def test_clear_sweep_uses_exact_wall_bounds_without_millimetre_sampling(self):
        with patch.object(self.planner, '_motion_samples',
                          side_effect=AssertionError('a clear swept AABB needs no samples')):
            self.assertTrue(self.planner.primitive_is_collision_free(
                self.grid(), (0., 0., 0.), 0., .1))

    def test_exact_broad_phase_matches_unskipped_collision_sampling(self):
        self.values[50, 60] = self.values[50, 65] = 100
        grid = self.grid(dynamic_cell_rectangles={
            50 * 100 + 65: [(.315, 0., .32, .02)]})
        random = np.random.RandomState(1820)
        for index in range(160):
            pose = (random.uniform(-.2, .32), random.uniform(-.075, .15),
                    random.uniform(-.5, .5))
            curvature, distance = random.uniform(-3., 3.), random.uniform(.005, .2)
            with patch.object(self.planner, '_primitive_region_is_empty', return_value=False):
                sampled = self.planner.primitive_is_collision_free(grid, pose, curvature, distance)
            with self.subTest(index=index, pose=pose):
                self.assertEqual(self.planner.primitive_is_collision_free(
                    grid, pose, curvature, distance), sampled)

    def test_exact_broad_phase_retains_map_and_keep_in_limits(self):
        grid = self.grid()
        self.assertFalse(self.planner.primitive_is_collision_free(
            grid, (.85, .2, 0.), 0., .1))
        planner = CollisionGeometry(
            self.planner.footprint, keep_in_rectangles=[(-.2, -.2, .2, .2)])
        self.assertFalse(planner.primitive_is_collision_free(
            grid, (.18, 0., 0.), 0., .1))


if __name__ == '__main__':
    unittest.main()
