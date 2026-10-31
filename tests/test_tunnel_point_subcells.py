"""Isolated 5 mm sensor-hit geometry and conservative source-map geometry."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.costmap import TunnelCostmap
from tunnel.geometry import CollisionGeometry, OccupancyGrid, RectangularFootprint


class PointSubcellTest(unittest.TestCase):
    index = 50*100 + 65

    def make_map(self, static_value=0):
        data = [0]*10000
        data[self.index] = static_value
        return TunnelCostmap(data, 100, 100, .02, -1., -1.,
                             mark_observations=2, clear_observations=2,
                             dynamic_cell_size=.005)

    def hit(self, grid, point=(.3175, .0025)):
        return grid.update_scan([math.hypot(*point)], math.atan2(point[1], point[0]),
                                .01, .1, 3., (0., 0., 0.))

    def face(self, grid):
        points = ((.3175, .0025), (.3175, .0075))
        angles = [math.atan2(y, x) for x, y in points]
        return grid.update_scan([math.hypot(x, y) for x, y in points],
                                angles[0], angles[1]-angles[0], .1, 3., (0., 0., 0.))

    def free_pose(self, grid, pose):
        values = np.array(grid.to_occupancy_data()).reshape(100, 100)
        snapshot = OccupancyGrid(values, .02, -1., -1.,
                                 dynamic_cell_rectangles=grid.refined_dynamic_cells())
        geometry = CollisionGeometry(RectangularFootprint(.067645, .118073, .0903, .01))
        return geometry.pose_is_collision_free(snapshot, pose)

    def test_hit_requires_two_observations_then_blocks_measured_point(self):
        grid = self.make_map()
        self.hit(grid)
        self.assertFalse(grid._dynamic_occupied[self.index])
        self.hit(grid)
        self.assertTrue(grid._dynamic_occupied[self.index])
        rectangles = grid.refined_dynamic_cells()[self.index]
        self.assertEqual(len(rectangles), 1)
        x0, y0, x1, y1 = rectangles[0]
        self.assertAlmostEqual(x1-x0, .005)
        self.assertAlmostEqual(y1-y0, .005)
        self.assertLessEqual(x0, .3175); self.assertLessEqual(.3175, x1)
        self.assertLessEqual(y0, .0025); self.assertLessEqual(.0025, y1)
        # Nominal TurtleBot body plus its existing 10 mm margin stays checked.
        self.assertFalse(self.free_pose(grid, (.250, 0., 0.)))
        self.assertFalse(self.free_pose(grid, (.3175, .0025, math.pi/2)))
        self.assertTrue(self.free_pose(grid, (.231, 0., 0.)))

    def test_new_hit_needs_confirmation_and_preserves_unseen_surface(self):
        grid = self.make_map()
        self.face(grid); self.face(grid)
        before = set(grid.refined_dynamic_cells()[self.index])
        self.assertGreaterEqual(len(before), 2)
        self.hit(grid, (.3075, .0175))
        self.assertEqual(set(grid.refined_dynamic_cells()[self.index]), before)
        self.hit(grid, (.3075, .0175))
        after = set(grid.refined_dynamic_cells()[self.index])
        self.assertTrue(before <= after)
        self.assertGreater(len(after), len(before))
        self.assertFalse(self.free_pose(grid, (.250, 0., 0.)))

    def test_invalid_scan_does_not_clear_measured_hits(self):
        grid = self.make_map()
        self.hit(grid); self.hit(grid)
        before = grid.refined_dynamic_cells()
        grid.update_scan([math.nan, -math.inf, 0.], 0., .01, .1, 3., (0., 0., 0.))
        self.assertEqual(grid.refined_dynamic_cells(), before)

    def test_static_and_unknown_source_cells_are_never_refined(self):
        for static_value in (100, -1):
            with self.subTest(static_value=static_value):
                grid = self.make_map(static_value)
                self.hit(grid); self.hit(grid)
                self.assertNotIn(self.index, grid.refined_dynamic_cells())
                self.assertFalse(self.free_pose(grid, (.231, 0., 0.)))
                self.assertEqual(grid._static_data[self.index], static_value)


if __name__ == '__main__':
    unittest.main()
