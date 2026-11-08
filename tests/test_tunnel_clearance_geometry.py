"""Geometric preferences use component clearance without changing hard checks."""
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import CollisionGeometry, OccupancyGrid, Pose2D, RectangularFootprint
from collision_inspection import CollisionInspector, inspect_geometry


PARTS = ((-.118073, -.0611, .067645, .0611),
         (-.04775, .06845, .04775, .07975),
         (-.04775, -.07975, .04775, -.06845),
         (.0335251942, -.021, .0677998152, .021))


def vertices(box, pose=(0., 0., 0.), padding=0.):
    x0, y0, x1, y1 = np.asarray(box) + [-padding, -padding, padding, padding]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1))) @ np.array([[c, s], [-s, c]]) + pose[:2]


def polygon_distance(first, second):
    """Independent edge-intersection and vertex-to-segment distance oracle."""
    def cross(a, b, p):
        d, q = b-a, p-a
        return d[0]*q[1]-d[1]*q[0]

    def inside(p, poly):
        signs = [cross(poly[i], poly[(i+1) % 4], p) for i in range(4)]
        return min(signs) >= -1e-12 or max(signs) <= 1e-12

    if inside(first[0], second) or inside(second[0], first):
        return 0.
    for a, b in zip(first, np.roll(first, -1, axis=0)):
        for c, d in zip(second, np.roll(second, -1, axis=0)):
            if (cross(a, b, c)*cross(a, b, d) <= 0.
                    and cross(c, d, a)*cross(c, d, b) <= 0.
                    and np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                               <= np.minimum(np.maximum(a, b), np.maximum(c, d))+1e-12)):
                return 0.
    distances = []
    for points, edges in ((first, second), (second, first)):
        for point in points:
            for a, b in zip(edges, np.roll(edges, -1, axis=0)):
                edge = b-a
                t = np.clip(np.dot(point-a, edge)/np.dot(edge, edge), 0., 1.)
                distances.append(float(np.linalg.norm(point-a-t*edge)))
    return min(distances)


def exact_grid(rectangles=(), source='static'):
    values = np.zeros((100, 100), dtype=np.int8)
    resolution, origin = .02, -1.
    pieces = {}
    for left, bottom, right, top in rectangles:
        for row in range(math.floor((bottom-origin)/resolution), math.ceil((top-origin)/resolution)):
            for column in range(math.floor((left-origin)/resolution), math.ceil((right-origin)/resolution)):
                x, y = origin+column*resolution, origin+row*resolution
                box = (max(left, x), max(bottom, y), min(right, x+resolution), min(top, y+resolution))
                if box[0] < box[2] and box[1] < box[3]:
                    values[row, column] = 100
                    pieces.setdefault(row*100+column, []).append(box)
    return OccupancyGrid(values, resolution, origin, origin,
                         **{source+'_cell_rectangles': pieces})


class ClearanceGeometryTest(unittest.TestCase):
    def checker(self, padding=.01):
        return CollisionGeometry(
            RectangularFootprint(.0677998152, .118073, .0903, padding, PARTS),
        )

    def test_empty_map_returns_cap_including_start_and_endpoint(self):
        checker = self.checker()
        result = checker.primitive_clearances(exact_grid(), (0., 0., 0.), 0., .12, .2)
        np.testing.assert_allclose(result, [.2]*9)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(checker.primitive_clearances(exact_grid(), (0., 0., 0.), 0., 0., .2)), 1)

    def test_straight_samples_measure_body_edge_not_base_center(self):
        checker = CollisionGeometry(RectangularFootprint(.1, .1, .05))
        grid = exact_grid(((.30, -.04, .31, .04),))
        np.testing.assert_allclose(
            checker.primitive_clearances(grid, (0., 0., 0.), 0., .1, .3),
            np.linspace(.2, .1, 9), atol=1e-12)

    def test_static_and_dynamic_refinement_are_not_replaced_by_parent_cells(self):
        checker = CollisionGeometry(RectangularFootprint(.1, .1, .05))
        for source in ('static', 'dynamic'):
            with self.subTest(source=source):
                grid = exact_grid(((.317, .001, .318, .004),), source)
                result = checker.primitive_clearances(grid, (.2, 0., 0.), 0., 0., .2)
                np.testing.assert_allclose(result, [.017], atol=1e-12)

    def test_padding_and_protruding_wheel_determine_distance(self):
        grid = exact_grid(((.01, .095, .015, .1),))
        np.testing.assert_allclose(self.checker().primitive_clearances(grid, (0., 0., 0.), 0., 0., .1), [.00525], atol=1e-12)
        np.testing.assert_allclose(self.checker(0.).primitive_clearances(grid, (0., 0., 0.), 0., 0., .1), [.01525], atol=1e-12)

    def test_camera_protrusion_is_included(self):
        grid = exact_grid(((.075, -.001, .076, .001),))
        result = self.checker(0.).primitive_clearances(grid, (0., 0., 0.), 0., 0., .1)
        np.testing.assert_allclose(result, [.075-.0677998152], atol=1e-12)

    def test_empty_envelope_corner_is_not_invented_material(self):
        grid = exact_grid(((-.111, .082, -.109, .084),))
        result = self.checker().primitive_clearances(grid, (0., 0., 0.), 0., 0., .1)
        np.testing.assert_allclose(result, [.082-.0711], atol=1e-12)

    def test_rotated_arc_matches_independent_polygon_distance_at_every_sample(self):
        rectangles = ((.15, .13, .19, .17), (-.22, .08, -.18, .12), (.24, -.12, .27, -.08))
        grid, checker = exact_grid(rectangles), self.checker()
        start = (.02, -.02, .61)
        curvature, distance, cap = 2.7, .24, .2
        expected = []
        for t in np.linspace(0., distance, 9):
            yaw = start[2]+curvature*t
            pose = (start[0]+(math.sin(yaw)-math.sin(start[2]))/curvature,
                    start[1]+(math.cos(start[2])-math.cos(yaw))/curvature, yaw)
            expected.append(min([cap]+[polygon_distance(vertices(part, pose, .01), vertices(obstacle))
                                       for part in PARTS for obstacle in rectangles]))
        np.testing.assert_allclose(checker.primitive_clearances(grid, start, curvature, distance, cap), expected, atol=1e-10)

    def test_rotated_pair_distance_matches_independent_oracle(self):
        rng = random.Random(571)
        checker = self.checker()
        for index in range(35):
            pose = (rng.uniform(-.1, .1), rng.uniform(-.1, .1), rng.uniform(-math.pi, math.pi))
            x, y = rng.uniform(-.3, .3), rng.uniform(-.3, .3)
            rectangle = (x, y, x+rng.uniform(.003, .07), y+rng.uniform(.003, .07))
            expected = min([.3]+[polygon_distance(vertices(part, pose, .01), vertices(rectangle)) for part in PARTS])
            with self.subTest(index=index):
                np.testing.assert_allclose(checker.primitive_clearances(exact_grid((rectangle,)), pose, 0., 0., .3), [expected], atol=1e-10)

    def test_edge_crossing_without_contained_vertices_has_zero_distance(self):
        checker = CollisionGeometry(RectangularFootprint(.15, .15, .01))
        grid = exact_grid(((-.01, -.15, .01, .15),))
        np.testing.assert_allclose(checker.primitive_clearances(grid, (0., 0., 0.), 0., 0., .2), [0.])

    def test_map_exterior_is_scored_but_keep_in_boundary_is_not(self):
        grid = exact_grid()
        checker = CollisionGeometry(RectangularFootprint(.1, .1, .05),
                                    keep_in_rectangles=[(-.2, -.2, .2, .2)])
        np.testing.assert_allclose(checker.primitive_clearances(grid, (.83, 0., 0.), 0., 0., .2), [.07], atol=1e-12)
        np.testing.assert_allclose(checker.primitive_clearances(grid, (1.2, 0., 0.), 0., 0., .2), [0.])
        self.assertFalse(inspect_geometry(checker).pose_is_collision_free(grid, (.83, 0., 0.)))

    def test_unknown_policy_and_soft_cost_are_preserved(self):
        values = np.zeros((20, 20), dtype=np.int8)
        values[10, 12] = -1
        checker = CollisionGeometry(RectangularFootprint(.1, .1, .05))
        for occupied in (True, False):
            grid = OccupancyGrid(values, .1, -1., -1., unknown_is_occupied=occupied)
            expected = .1 if occupied else .3
            np.testing.assert_allclose(checker.primitive_clearances(grid, (0., 0., 0.), 0., 0., .3), [expected], atol=1e-12)
        values[10, 12] = 40
        grid = OccupancyGrid(values, .1, -1., -1., occupied_threshold=65,
                             soft_cost_data=np.maximum(values, 0))
        np.testing.assert_allclose(checker.primitive_clearances(grid, (0., 0., 0.), 0., 0., .3), [.3])

    def test_invalid_inputs_and_zero_cap(self):
        checker, grid = self.checker(), exact_grid()
        for curvature, distance, cap in ((math.nan, .1, .2), (0., -.1, .2),
                                         (0., .1, -.2), (0., math.inf, .2)):
            with self.subTest(curvature=curvature, distance=distance, cap=cap), self.assertRaises(ValueError):
                checker.primitive_clearances(grid, (0., 0., 0.), curvature, distance, cap)
        np.testing.assert_array_equal(checker.primitive_clearances(grid, (0., 0., 0.), 0., .1, 0.), np.zeros(9))
        for samples in (None, True, 1, 1.5, 2.5, math.nan, math.inf, 'invalid'):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                checker.primitive_clearances(grid, (0., 0., 0.), 0., .1, .2, samples=samples)

    def test_soft_sample_count_does_not_relax_hard_sweep(self):
        checker = CollisionInspector(RectangularFootprint(.01, .01, .01),
                                     collision_check_step=.01, collision_check_angle=.05)
        grid = exact_grid(((.095, -.002, .105, .002),))
        start = Pose2D(0., 0., 0.)
        hard_samples = list(checker._motion_samples(start, 0., .2, checker.collision_check_step))
        self.assertFalse(inspect_geometry(checker).primitive_is_collision_free(grid, start, 0., .2))
        for samples in (2, 7, 9, 17):
            result = checker.primitive_clearances(grid, start, 0., .2, .2, samples=samples)
            self.assertEqual(len(result), samples)
            self.assertFalse(inspect_geometry(checker).primitive_is_collision_free(grid, start, 0., .2))
            self.assertEqual(list(checker._motion_samples(start, 0., .2, checker.collision_check_step)), hard_samples)
        self.assertTrue(np.all(checker.primitive_clearances(grid, start, 0., .2, .2, samples=2) > 0.))
        self.assertEqual(len(checker.primitive_clearances(grid, start, 0., 0., .2, samples=17)), 1)


if __name__ == '__main__':
    unittest.main()
