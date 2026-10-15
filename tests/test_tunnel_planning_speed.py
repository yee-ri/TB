"""Planning certificates preserve precise live checks without a CPU-time budget."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))

from tunnel.planner import HybridAStarPlanner, OccupancyGrid, Pose2D, RectangularFootprint


def refined_grid(static=(), dynamic=(), occupancy=()):
    """Rasterize rectangular fixtures while retaining their exact dimensions."""
    data = np.zeros((100, 100), dtype=np.int8)
    sources = []
    for boxes in (static, dynamic):
        cells = {}
        for left, bottom, right, top in boxes:
            for row in range(math.floor((bottom + 1.) / .02),
                             math.ceil((top + 1.) / .02)):
                for column in range(math.floor((left + 1.) / .02),
                                    math.ceil((right + 1.) / .02)):
                    x, y = -1. + column * .02, -1. + row * .02
                    box = (max(left, x), max(bottom, y),
                           min(right, x + .02), min(top, y + .02))
                    if box[2] - box[0] < 1e-10 or box[3] - box[1] < 1e-10:
                        continue
                    data[row, column] = 100
                    cells.setdefault(row * 100 + column, []).append(box)
        sources.append(cells)
    for row, column, value in occupancy:
        data[row, column] = value
    return OccupancyGrid(data, .02, -1., -1.,
                         static_cell_rectangles=sources[0],
                         dynamic_cell_rectangles=sources[1])


class PlanningSweepCertificateTest(unittest.TestCase):
    def test_envelope_contains_dense_padded_corner_arcs(self):
        random = np.random.RandomState(713)
        for index in range(100):
            footprint = RectangularFootprint(
                random.uniform(.02, .15), random.uniform(.16, .3),
                random.uniform(.02, .07), .01)
            planner = HybridAStarPlanner(footprint)
            start = Pose2D(*random.uniform(-3., 3., size=3))
            distance = random.uniform(.001, .7)
            angle = (0., 1e-13, math.pi, -math.pi)[index % 4] if index < 12 else random.uniform(-math.pi, math.pi)
            curvature = angle / distance
            cx, cy, c, s, half_length, half_width = planner._swept_footprint_envelope(
                start, curvature, distance)

            # Independent integrated circular motion, including every padded
            # corner and both yaw-wrap directions; do not reuse propagation.
            station = np.linspace(0., distance, 101)
            theta = curvature * station
            local_x = station * np.sinc(theta / math.pi)
            local_y = station * theta / 2. * np.sinc(theta / (2. * math.pi)) ** 2
            x = start.x + math.cos(start.yaw) * local_x - math.sin(start.yaw) * local_y
            y = start.y + math.sin(start.yaw) * local_x + math.cos(start.yaw) * local_y
            yaw = start.yaw + theta
            for front in (footprint.front + .01, -footprint.rear - .01):
                for side in (-footprint.half_width - .01, footprint.half_width + .01):
                    dx = x + front * np.cos(yaw) - side * np.sin(yaw) - cx
                    dy = y + front * np.sin(yaw) + side * np.cos(yaw) - cy
                    with self.subTest(index=index, corner=(front, side)):
                        self.assertLessEqual(np.max(np.abs(c * dx + s * dy)), half_length + 2e-9)
                        self.assertLessEqual(np.max(np.abs(-s * dx + c * dy)), half_width + 2e-9)

    def test_diagonal_clear_sweep_avoids_dense_sampling(self):
        grid = refined_grid(static=[(0., .055, .002, .057)])
        planner = HybridAStarPlanner(RectangularFootprint(.03, .05, .01, .003))
        start = Pose2D(0., 0., math.pi / 4.)
        self.assertTrue(planner.pose_is_collision_free(grid, start))
        self.assertFalse(planner._primitive_region_is_empty(grid, start, 0., .2))
        with patch.object(planner, '_motion_samples',
                          side_effect=AssertionError('a disjoint swept box needs no samples')):
            self.assertTrue(planner._planning_primitive_is_collision_free(grid, start, 0., .2))

    def test_thin_wall_and_curved_rear_contact_between_clear_endpoints(self):
        cases = (
            (refined_grid(static=[(.104, -.02, .106, .02)]),
             RectangularFootprint(.0002, .0002, .0002), Pose2D(.101, 0., 0.), 0., .01),
            (refined_grid(dynamic=[(.033, -.097, .035, -.095)]),
             RectangularFootprint(.08, .19, .03, .01), Pose2D(0., 0., 0.), 4., .35),
        )
        for grid, footprint, start, curvature, distance in cases:
            planner = HybridAStarPlanner(footprint, collision_check_step=.02)
            with self.subTest(curvature=curvature):
                self.assertTrue(planner.pose_is_collision_free(grid, start))
                self.assertTrue(planner.pose_is_collision_free(
                    grid, planner._propagate(start, curvature, distance)))
                self.assertFalse(planner.primitive_is_collision_free(grid, start, curvature, distance))
                self.assertFalse(planner._planning_primitive_is_collision_free(grid, start, curvature, distance))

    def test_map_boundary_and_nonconvex_keep_in_are_checked_between_endpoints(self):
        grid = refined_grid()
        footprint = RectangularFootprint(.005, .005, .005)
        planner = HybridAStarPlanner(footprint)
        start, curvature, distance = Pose2D(.85, 0., -math.pi / 2.), 10., 2. * math.pi / 10.
        self.assertTrue(planner.pose_is_collision_free(grid, start))
        self.assertTrue(planner.pose_is_collision_free(grid, planner._propagate(start, curvature, distance)))
        self.assertFalse(planner._planning_primitive_is_collision_free(grid, start, curvature, distance))
        for gap, expected in ((0., True), (.02, False)):
            planner = HybridAStarPlanner(footprint, keep_in_rectangles=[
                (-.1, -.1, .12, .1), (.12 + gap, -.05, .4, .05)])
            with self.subTest(gap=gap):
                self.assertEqual(planner._planning_primitive_is_collision_free(
                    grid, Pose2D(0., 0., 0.), 0., .3), expected)

    def test_unknown_cells_and_inflation_remain_hard_obstacles(self):
        data = np.zeros((100, 100), dtype=np.int8)
        data[50, 55] = -1
        start = Pose2D(-.1, .01, 0.)
        footprint = RectangularFootprint(.005, .005, .005)
        for unknown, expected in ((True, False), (False, True)):
            grid = OccupancyGrid(data, .02, -1., -1., unknown_is_occupied=unknown)
            planner = HybridAStarPlanner(footprint)
            self.assertTrue(planner.pose_is_collision_free(grid, start))
            self.assertEqual(planner._planning_primitive_is_collision_free(grid, start, 0., .3), expected)
        grid = refined_grid(static=[(.06, .061, .08, .063)])
        for inflation, expected in ((0., True), (.061, False)):
            planner = HybridAStarPlanner(footprint, inflation_radius=inflation)
            self.assertTrue(planner.pose_is_collision_free(grid, start))
            self.assertEqual(planner._planning_primitive_is_collision_free(grid, start, 0., .3), expected)

    def test_seeded_sweeps_never_accept_a_precise_collision(self):
        grid = refined_grid(static=[(.15, -.15, .152, .15), (-.2, -.125, .1, -.123)],
                            dynamic=[(.30, .10, .315, .115)],
                            occupancy=[(55, 45, 100), (52, 55, -1)])
        planner = HybridAStarPlanner(RectangularFootprint(.04, .16, .035, .01))
        random = np.random.RandomState(1042)
        checked = accepted = blocked = 0
        for index in range(180):
            start = Pose2D(random.uniform(-.3, .35), random.uniform(-.2, .2),
                           random.uniform(-math.pi, math.pi))
            if not planner.pose_is_collision_free(grid, start):
                continue  # Search helper accepts only already checked starts.
            curvature, distance = random.uniform(-8., 8.), random.uniform(.002, .5)
            with patch.object(planner, '_primitive_region_is_empty', return_value=False):
                precise = planner.primitive_is_collision_free(grid, start, curvature, distance)
            planning = planner._planning_primitive_is_collision_free(grid, start, curvature, distance)
            with self.subTest(index=index, start=start, curvature=curvature, distance=distance):
                if not precise:
                    self.assertFalse(planning)
            checked += 1
            accepted += planning
            blocked += not planning
        self.assertGreater(checked, 40)
        self.assertGreater(accepted, 10)
        self.assertGreater(blocked, 10)

    def test_public_checker_retains_precise_samples_independent_of_search(self):
        grid = refined_grid(static=[(.104, -.02, .106, .02)])
        planner = HybridAStarPlanner(RectangularFootprint(.0002, .0002, .0002),
                                    collision_check_step=.02)
        with patch.object(planner, '_planning_primitive_is_collision_free', return_value=True) as planning, \
                patch.object(planner, '_motion_samples', wraps=planner._motion_samples) as samples:
            self.assertFalse(planner.primitive_is_collision_free(grid, (.101, 0., 0.), 0., .01))
        planning.assert_not_called()
        samples.assert_called_once()
        self.assertLessEqual(samples.call_args.args[3], .001 + 1e-12)

    def test_search_edges_and_analytic_connection_use_planning_checker(self):
        grid = refined_grid()
        for analytic in (False, True):
            planner = HybridAStarPlanner(RectangularFootprint(.02, .03, .01),
                                        enable_analytic_goal_connection=analytic,
                                        primitive_step=.1, maximum_iterations=5)
            with patch.object(planner, '_planning_primitive_is_collision_free', return_value=False) as check:
                self.assertIsNone(planner.plan(grid, (0., 0., 0.), (.5, 0., 0.)))
            distances = [call.args[3] for call in check.call_args_list]
            with self.subTest(analytic=analytic):
                self.assertIn(.1, distances)
                self.assertEqual(.5 in distances, analytic)


if __name__ == '__main__':
    unittest.main()
