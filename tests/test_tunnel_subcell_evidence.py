"""Confirm and clear fine obstacle cells using evidence at the same resolution."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.costmap import TunnelCostmap


class SubcellEvidenceTest(unittest.TestCase):
    width = 50
    height = 50
    resolution = .02
    parent_index = 5 * width + 25
    near_x = .501
    far_x = .516
    ray_y = .102

    def make_map(self, static_value=0, dynamic_cell_size=.005, **settings):
        data = [0] * (self.width * self.height)
        data[self.parent_index] = static_value
        return TunnelCostmap(
            data, self.width, self.height, self.resolution, 0., 0.,
            mark_observations=2, clear_observations=2,
            dynamic_cell_size=dynamic_cell_size, **settings)

    def hit(self, grid, x, y=None):
        return grid.update_scan(
            [x - .1], 0., .01, .01, .8,
            (.1, self.ray_y if y is None else y, 0.))

    def clear_ray(self, grid):
        return grid.update_scan(
            [math.inf], 0., .01, .01, .8, (.1, self.ray_y, 0.))

    def unobserved(self, grid):
        return grid.update_scan(
            [math.nan, -math.inf, 0.], 0., .01, .01, .8,
            (.1, self.ray_y, 0.))

    def rectangles(self, grid):
        return grid.refined_dynamic_cells().get(self.parent_index, ())

    def contains(self, grid, x, y=None):
        y = self.ray_y if y is None else y
        return any(x0 <= x <= x1 and y0 <= y <= y1
                   for x0, y0, x1, y1 in self.rectangles(grid))

    def test_different_fine_hit_cannot_confirm_earlier_point(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.hit(grid, self.far_x)
        self.assertEqual(self.rectangles(grid), ())
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 0)

        self.hit(grid, self.far_x)
        self.assertFalse(self.contains(grid, self.near_x))
        self.assertTrue(self.contains(grid, self.far_x))

    def test_501_then_516_does_not_accumulate_a_ghost(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        for _ in range(21):
            self.hit(grid, self.far_x)
        self.assertFalse(self.contains(grid, self.near_x))
        self.assertTrue(self.contains(grid, self.far_x))
        self.assertEqual(len(self.rectangles(grid)), 1)

    def test_new_point_needs_its_own_evidence_in_confirmed_parent(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.hit(grid, self.near_x)
        self.hit(grid, self.far_x, y=.117)
        self.assertTrue(self.contains(grid, self.near_x))
        self.assertFalse(self.contains(grid, self.far_x, y=.117))

        update = self.hit(grid, self.far_x, y=.117)
        self.assertTrue(self.contains(grid, self.near_x))
        self.assertTrue(self.contains(grid, self.far_x, y=.117))
        self.assertGreater(update.refined_cells, 0)

    def test_two_rays_clear_fine_cell_while_parent_has_other_hit(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.hit(grid, self.near_x)
        self.hit(grid, self.far_x)
        self.assertTrue(self.contains(grid, self.near_x))
        self.assertFalse(self.contains(grid, self.far_x))

        update = self.hit(grid, self.far_x)
        self.assertFalse(self.contains(grid, self.near_x))
        self.assertTrue(self.contains(grid, self.far_x))
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 100)
        self.assertGreater(update.refined_cells, 0)

    def test_unobserved_candidate_keeps_its_hit_evidence(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.unobserved(grid)
        self.hit(grid, .401)
        self.assertFalse(self.contains(grid, self.near_x))

        self.hit(grid, self.near_x)
        self.assertTrue(self.contains(grid, self.near_x))

    def test_clear_evidence_resets_unconfirmed_hit_evidence(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.clear_ray(grid)
        self.hit(grid, self.near_x)
        self.assertFalse(self.contains(grid, self.near_x))
        self.hit(grid, self.near_x)
        self.assertTrue(self.contains(grid, self.near_x))

    def test_unobserved_and_occluded_known_surface_remains(self):
        grid = self.make_map()
        self.hit(grid, self.near_x)
        self.hit(grid, self.near_x)
        before = self.rectangles(grid)
        for _ in range(5):
            self.unobserved(grid)
            self.hit(grid, .401)
            self.hit(grid, self.far_x, y=.117)
            self.assertTrue(self.contains(grid, self.near_x))
        self.assertTrue(set(before).issubset(self.rectangles(grid)))

    def test_many_beams_in_one_scan_count_as_one_observation(self):
        grid = self.make_map()
        grid.update_scan([self.near_x - .1] * 8, 0., 0., .01, .8,
                         (.1, self.ray_y, 0.))
        self.assertFalse(self.contains(grid, self.near_x))
        self.hit(grid, self.near_x)
        self.assertTrue(self.contains(grid, self.near_x))

    def test_hit_wins_over_clear_in_the_same_fine_cell(self):
        grid = self.make_map()
        for _ in range(2):
            grid.update_scan([self.near_x - .1, math.inf], 0., 0., .01, .8,
                             (.1, self.ray_y, 0.))
        self.assertTrue(self.contains(grid, self.near_x))

        self.clear_ray(grid)
        self.assertTrue(self.contains(grid, self.near_x))
        self.clear_ray(grid)
        self.assertFalse(self.contains(grid, self.near_x))
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 0)

    def test_static_and_unknown_cells_keep_conservative_geometry(self):
        for value in (100, -1):
            with self.subTest(static_value=value):
                grid = self.make_map(static_value=value)
                self.hit(grid, self.near_x)
                self.hit(grid, self.near_x)
                self.assertNotIn(self.parent_index, grid.refined_dynamic_cells())
                self.assertIn(grid.to_occupancy_data()[self.parent_index],
                              (value, 100))
                self.clear_ray(grid)
                self.clear_ray(grid)
                self.assertEqual(grid.to_occupancy_data()[self.parent_index], value)
                self.assertNotIn(self.parent_index, grid.refined_dynamic_cells())

    def test_coarse_mode_keeps_original_parent_observation_contract(self):
        grid = self.make_map(dynamic_cell_size=0.)
        self.hit(grid, self.near_x)
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 0)
        self.hit(grid, self.far_x)
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 100)
        self.assertEqual(grid.refined_dynamic_cells(), {})
        for _ in range(3):
            self.hit(grid, self.far_x)
            self.assertEqual(grid.to_occupancy_data()[self.parent_index], 100)
        self.clear_ray(grid)
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 100)
        self.clear_ray(grid)
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 0)

    def test_endpoint_guard_preserves_surface_without_confirming_new_hit(self):
        grid = self.make_map(endpoint_clear_guard_radius=.03)
        self.hit(grid, self.near_x)
        self.hit(grid, self.near_x)
        self.hit(grid, self.far_x)
        self.assertTrue(self.contains(grid, self.near_x))
        self.assertFalse(self.contains(grid, self.far_x))
        self.hit(grid, self.far_x)
        self.assertTrue(self.contains(grid, self.near_x))
        self.assertTrue(self.contains(grid, self.far_x))
        self.clear_ray(grid)
        self.clear_ray(grid)
        self.assertEqual(self.rectangles(grid), ())

    def test_configured_decay_removes_unseen_fine_surface(self):
        grid = self.make_map(decay_updates=2)
        self.hit(grid, self.near_x)
        self.hit(grid, self.near_x)
        self.unobserved(grid)
        self.assertTrue(self.contains(grid, self.near_x))
        update = self.unobserved(grid)
        self.assertFalse(self.contains(grid, self.near_x))
        self.assertEqual(grid.to_occupancy_data()[self.parent_index], 0)
        self.assertEqual(update.decayed_cells, 1)

    def test_diagonal_corner_touch_does_not_clear_adjacent_cell(self):
        grid = self.make_map()
        points = ((.5025, .5025), (.5075, .5025), (.4975, .5025))
        for x, y in points:
            for _ in range(2):
                grid.update_scan([.001], math.pi / 2., .01, 0., .1,
                                 (x, y-.001, 0.))
        for _ in range(2):
            grid.update_scan([math.inf], math.pi / 4., .01, .01, 1.,
                             (.1, .1, 0.))
        remaining = [box for boxes in grid.refined_dynamic_cells().values()
                     for box in boxes]
        for x, y in points:
            present = any(x0 < x < x1 and y0 < y < y1
                          for x0, y0, x1, y1 in remaining)
            self.assertEqual(present, x != .5025)


class FineRayTraversalTest(unittest.TestCase):
    @staticmethod
    def crosses_interior(start, end, box):
        """Split a segment at rectangle edges, then inspect interval midpoints."""
        dx, dy = end[0]-start[0], end[1]-start[1]
        fractions = [0., 1.]
        for position, delta, lower, upper in (
                (start[0], dx, box[0], box[2]),
                (start[1], dy, box[1], box[3])):
            if delta:
                fractions.extend((edge-position)/delta for edge in (lower, upper)
                                 if 0. < (edge-position)/delta < 1.)
        fractions = sorted(set(fractions))
        for left, right in zip(fractions, fractions[1:]):
            t = (left+right)/2.
            x, y = start[0]+t*dx, start[1]+t*dy
            if (box[0]+1e-11 < x < box[2]-1e-11
                    and box[1]+1e-11 < y < box[3]-1e-11):
                return True
        return False

    def test_public_scan_clearing_matches_segment_geometry_after_clipping(self):
        rng = random.Random(734)
        cases = [((-.3, .4825), (1.3, .4825)),
                 ((1.3, .4825), (-.3, .4825)),
                 ((.4825, -.3), (.4825, 1.3)),
                 ((.4825, 1.3), (.4825, -.3)),
                 ((-.3, -.3), (1.3, 1.3)),
                 ((-.3, .8), (1.3, 1.2))]
        points = [(.4025 + i*.04, .4025 + j*.04)
                  for i in range(5) for j in range(5)]
        for _ in range(12):
            x, y = rng.choice(points)
            angle = rng.uniform(-math.pi, math.pi)
            c, s = math.cos(angle), math.sin(angle)
            cases.append(((x-.9*c, y-.9*s), (x+.8*c, y+.8*s)))

        for origin in ((0., 0.), (-.373, -.271)):
            for number, (start, end) in enumerate(cases):
                with self.subTest(origin=origin, ray=number):
                    grid = TunnelCostmap(
                        [0]*2500, 50, 50, .02, *origin,
                        mark_observations=2, clear_observations=2,
                        dynamic_cell_size=.005)
                    for x, y in points:
                        for _ in range(2):
                            grid.update_scan([.001], 0., .01, 0., .1,
                                             (x+origin[0]-.001,
                                              y+origin[1], 0.))
                    before = {box for boxes in grid.refined_dynamic_cells().values()
                              for box in boxes}
                    self.assertEqual(len(before), len(points))
                    start = (start[0]+origin[0], start[1]+origin[1])
                    end = (end[0]+origin[0], end[1]+origin[1])
                    expected = {box for box in before
                                if not self.crosses_interior(start, end, box)}
                    angle = math.atan2(end[1]-start[1], end[0]-start[0])
                    length = math.hypot(end[0]-start[0], end[1]-start[1])
                    grid.update_scan([math.inf], angle, .01, 0., length,
                                     (*start, 0.))
                    first = {box for boxes in grid.refined_dynamic_cells().values()
                             for box in boxes}
                    self.assertEqual(first, before)
                    grid.update_scan([math.inf], angle, .01, 0., length,
                                     (*start, 0.))
                    after = {box for boxes in grid.refined_dynamic_cells().values()
                             for box in boxes}
                    self.assertEqual(after, expected)


if __name__ == '__main__':
    unittest.main()
