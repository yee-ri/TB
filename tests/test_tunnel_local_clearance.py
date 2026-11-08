"""Full-body clearance scores steer local paths without hard overlap rejection."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import CollisionGeometry, OccupancyGrid, Pose2D, RectangularFootprint
from collision_inspection import inspect_geometry
from tunnel.planner import GlobalLocalPlanner, GlobalRoute


CONTROL = dict(period=.05, cruise_velocity=.32, maximum_angular_velocity=1.4,
               maximum_lateral_acceleration=.30, linear_acceleration=.35,
               linear_deceleration=.40, angular_acceleration=8.,
               safety_linear_deceleration=.15)
PARTS = ((-.118073, -.0611, .067645, .0611),
         (-.04775, .06845, .04775, .07975),
         (-.04775, -.07975, .04775, -.06845),
         (.03352519420082252, -.021, .06779981518632729, .021))


def room(rectangles=()):
    resolution, origin_x, origin_y = .005, -.5, -.5
    values = np.zeros((200, 400), dtype=np.int8)
    for xmin, ymin, xmax, ymax in rectangles:
        x0 = max(0, math.floor((xmin-origin_x)/resolution+1e-9))
        x1 = min(values.shape[1], math.ceil((xmax-origin_x)/resolution-1e-9))
        y0 = max(0, math.floor((ymin-origin_y)/resolution+1e-9))
        y1 = min(values.shape[0], math.ceil((ymax-origin_y)/resolution-1e-9))
        values[y0:y1, x0:x1] = 100
    return OccupancyGrid(values, resolution, origin_x, origin_y)


class LocalClearancePreferenceTest(unittest.TestCase):
    def setUp(self):
        self.geometry = CollisionGeometry(
            RectangularFootprint(.06779981518632729, .118073, .0903,
                                 padding=.01, parts=PARTS),
            minimum_turning_radius=.18)
        self.pose = Pose2D(0., 0., 0.)
        self.straight = GlobalRoute(np.array(((0., 0.), (.8, 0.))))

    def planner(self, weight):
        return GlobalLocalPlanner(self.geometry, {
            'local_clearance_distance': .04, 'local_clearance_weight': weight})

    def command(self, planner, grid, route=None):
        return planner.local_command(grid, self.pose, self.straight if route is None else route,
                                     CONTROL, .32, 0., CONTROL['period'])

    def test_empty_room_preserves_both_straight_and_curved_tracking_commands(self):
        for endpoint in ((.8, 0.), (.8, .2), (.8, -.2)):
            with self.subTest(endpoint=endpoint):
                route = GlobalRoute(np.array(((0., 0.), endpoint)))
                before = self.command(self.planner(0.), room(), route)
                after = self.command(self.planner(.20), room(), route)
                self.assertIsNotNone(before)
                self.assertEqual(after, before)

    def test_near_side_wall_prefers_motion_with_more_body_clearance(self):
        # Guidance drifts gently toward the nearby lower wall. A valid path
        # remains available, but the new preference should reduce that drift.
        grid = room(((-.3, -.3, 1., -.105),))
        route = GlobalRoute(np.array(((0., 0.), (.8, -.02))))
        baseline, preferred = self.planner(0.), self.planner(.20)
        before = self.command(baseline, grid, route)
        after = self.command(preferred, grid, route)
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertGreater(after.angular_velocity, before.angular_velocity)
        before_curve = before.angular_velocity/before.linear_velocity
        after_curve = after.angular_velocity/after.linear_velocity
        before_gaps = self.geometry.primitive_clearances(grid, self.pose, before_curve, .30, .04)
        after_gaps = self.geometry.primitive_clearances(grid, self.pose, after_curve, .30, .04)
        self.assertGreater(float(np.mean(after_gaps)), float(np.mean(before_gaps)))

    def test_symmetric_walls_keep_centered_straight_command(self):
        grid = room(((-.3, -.3, 1., -.11), (-.3, .11, 1., .3)))
        command = self.command(self.planner(.20), grid)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertAlmostEqual(command.angular_velocity, 0., places=12)

    def test_small_positive_gap_does_not_become_a_new_hard_clearance_limit(self):
        grid = room(((-.3, -.3, 1., -.095), (-.3, .095, 1., .3)))
        gaps = self.geometry.primitive_clearances(grid, self.pose, 0., .30, .04)
        self.assertGreater(float(np.min(gaps)), 0.)
        self.assertLess(float(np.max(gaps)), .01)
        self.assertTrue(inspect_geometry(self.geometry).primitive_is_collision_free(grid, self.pose, 0., .30))
        for weight in (.20, 2.):
            with self.subTest(weight=weight):
                planner = self.planner(weight)
                command = self.command(planner, grid)
                self.assertIsNotNone(command)
                self.assertGreater(command.linear_velocity, 0.)
                self.assertAlmostEqual(command.angular_velocity, 0., places=12)

    def test_nonoverlapping_arc_is_preferred_when_clearance_scoring_is_enabled(self):
        grid = room(((-.3, -.3, 1., -.105),))
        route = GlobalRoute(np.array(((0., 0.), (.8, -.12))))
        for weight in (.20, .40):
            with self.subTest(weight=weight):
                planner = self.planner(weight)
                tx, ty, _, _ = planner.select_target(route, self.pose, planner.local_lookahead)
                desired = 2.*ty/(tx*tx+ty*ty)
                self.assertGreater(planner._clearance_score(grid, self.pose, desired, .30)[0], 0)
                command = self.command(planner, grid, route)
                self.assertIsNotNone(command)
                self.assertGreater(command.linear_velocity, 0.)
                curvature = command.angular_velocity/command.linear_velocity
                self.assertEqual(planner._clearance_score(grid, self.pose, curvature, .30)[0], 0)

    def test_zero_weight_disables_both_overlap_and_distance_preferences(self):
        grid = room(((-.3, -.3, 1., -.105),))
        route = GlobalRoute(np.array(((0., 0.), (.8, -.12))))
        planner = self.planner(0.)
        self.assertEqual(planner._clearance_score(grid, self.pose, -.98, .30), (0, 0.))
        self.assertEqual(self.command(planner, grid, route), self.command(planner, room(), route))

    def test_starting_overlap_prefers_leaving_the_obstacle_without_stopping(self):
        grid = room(((-.3, .08, 1., .3),))
        self.assertFalse(inspect_geometry(self.geometry).pose_is_collision_free(grid, self.pose))
        planner = self.planner(.20)
        maximum = 1./self.geometry.minimum_turning_radius
        for curvature in np.linspace(-maximum, maximum, planner.local_steering_samples):
            self.assertGreater(planner._clearance_score(grid, self.pose, curvature, .30)[0], 0)
        command = self.command(planner, grid)
        self.assertIsNotNone(command)
        self.assertGreater(command.linear_velocity, 0.)
        self.assertLess(command.angular_velocity, 0.)
        chosen = planner._clearance_score(grid, self.pose,
                                          command.angular_velocity/command.linear_velocity, .30)
        straight = planner._clearance_score(grid, self.pose, 0., .30)
        self.assertLess(chosen[0], straight[0])

    def test_overlapping_forward_paths_keep_a_command_and_nonzero_clearance_cost(self):
        grid = room(((.11, -.5, .3, .5),))
        self.assertTrue(inspect_geometry(self.geometry).pose_is_collision_free(grid, self.pose))
        self.assertFalse(inspect_geometry(self.geometry).primitive_is_collision_free(grid, self.pose, 0., .30))
        preferred = self.planner(.20)
        self.assertGreater(preferred._clearance_score(grid, self.pose, 0., .30)[0], 0)
        self.assertGreater(preferred._clearance_score(grid, self.pose, 0., .30)[1], 0.)
        self.assertEqual(preferred._clearance_score(room(), self.pose, 0., .30), (0, 0.))
        for weight in (0., .20):
            with self.subTest(weight=weight):
                command = self.command(self.planner(weight), grid)
                self.assertIsNotNone(command)
                self.assertGreater(command.linear_velocity, 0.)


if __name__ == '__main__':
    unittest.main()
