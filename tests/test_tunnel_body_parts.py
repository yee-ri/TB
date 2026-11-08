"""Body-part coverage and swept-motion regressions for tunnel collisions."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import OccupancyGrid, Pose2D, RectangularFootprint
from collision_inspection import CollisionInspector


FRONT = .06779981518632729
REAR = .118073
HALF_WIDTH = .0903
PARTS = (
    (-.118073, -.0611, .067645, .0611),
    (-.04775, .06845, .04775, .07975),
    (-.04775, -.07975, .04775, -.06845),
    (.03352519420082252, -.021, FRONT, .021),
)


def obstacle_grid(rectangles=()):
    """Place exact obstacle rectangles in otherwise free 20 mm raster cells."""
    resolution, origin, size = .02, -1., 100
    values = np.zeros((size, size), dtype=np.int8)
    pieces = {}
    for left, bottom, right, top in rectangles:
        c0, c1 = math.floor((left-origin)/resolution), math.ceil((right-origin)/resolution)
        r0, r1 = math.floor((bottom-origin)/resolution), math.ceil((top-origin)/resolution)
        for row in range(r0, r1):
            for column in range(c0, c1):
                x, y = origin+column*resolution, origin+row*resolution
                clipped = (max(left, x), max(bottom, y),
                           min(right, x+resolution), min(top, y+resolution))
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    continue
                values[row, column] = 100
                pieces.setdefault(row*size+column, []).append(clipped)
    return OccupancyGrid(values, resolution, origin, origin,
                         static_cell_rectangles=pieces)


def point_obstacle(point, half_size=.001):
    x, y = point
    return obstacle_grid(((x-half_size, y-half_size, x+half_size, y+half_size),))


def world_point(pose, local):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return (pose.x+c*local[0]-s*local[1], pose.y+s*local[0]+c*local[1])


def geometry(padding=0., parts=PARTS):
    return CollisionInspector(
        RectangularFootprint(FRONT, REAR, HALF_WIDTH, padding, parts=parts),
        collision_check_step=.02)


class BodyPartCollisionTest(unittest.TestCase):
    def test_empty_rear_corner_does_not_count_as_material(self):
        grid = obstacle_grid(((-.113, .077, -.108, .081),))
        self.assertTrue(geometry().pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertFalse(geometry(parts=()).pose_is_collision_free(grid, (0., 0., 0.)))

    def test_each_body_part_blocks_an_obstacle_at_multiple_robot_headings(self):
        # The camera point lies beyond the chassis box, so dropping that part fails.
        contacts = {'body': (-.10, 0.), 'left wheel': (0., .0741),
                    'right wheel': (0., -.0741), 'camera': (.06772, 0.)}
        checker = geometry()
        for yaw in (0., .61, math.pi/2., -math.pi/2.):
            pose = Pose2D(.2, -.2, yaw)
            for name, local in contacts.items():
                with self.subTest(part=name, yaw=yaw):
                    grid = point_obstacle(world_point(pose, local), half_size=.00001)
                    self.assertFalse(checker.pose_is_collision_free(grid, pose))

    def test_padding_is_applied_to_body_and_each_wheel(self):
        for point in ((-.123, 0.), (-.10, .0661), (.05275, .0741), (.05275, -.0741)):
            with self.subTest(point=point):
                grid = point_obstacle(point, half_size=.0005)
                self.assertTrue(geometry().pose_is_collision_free(grid, (0., 0., 0.)))
                self.assertFalse(geometry(.01).pose_is_collision_free(grid, (0., 0., 0.)))

    def test_padding_does_not_fill_the_entire_envelope_corner(self):
        grid = point_obstacle((-.11, .085), half_size=.001)
        self.assertTrue(geometry(.01).pose_is_collision_free(grid, (0., 0., 0.)))
        self.assertFalse(geometry(.01, parts=()).pose_is_collision_free(grid, (0., 0., 0.)))

    def test_rotated_body_can_overlap_between_free_endpoint_poses(self):
        checker = geometry()
        grid = point_obstacle((-.117, 0.), half_size=.0005)
        self.assertTrue(checker.pose_is_collision_free(grid, Pose2D(0., 0., -math.pi/4.)))
        self.assertTrue(checker.pose_is_collision_free(grid, Pose2D(0., 0., math.pi/4.)))
        self.assertFalse(checker.pose_is_collision_free(grid, Pose2D(0., 0., 0.)))

    def test_rotation_samples_distinguish_real_parts_from_rectangular_envelope(self):
        grid = point_obstacle((-.14, 0.), half_size=.0005)
        poses = [Pose2D(0., 0., yaw) for yaw in np.linspace(-math.pi/2., math.pi/2., 721)]
        self.assertTrue(all(geometry().pose_is_collision_free(grid, pose) for pose in poses))
        self.assertFalse(all(geometry(parts=()).pose_is_collision_free(grid, pose) for pose in poses))

    def test_curved_sweep_detects_each_wheel_between_free_endpoints(self):
        checker = geometry()
        radius, distance = .3, .3*math.pi/2.
        start = Pose2D(-.3, 0., 0.)
        end = Pose2D(0., .3, math.pi/2.)
        middle = Pose2D(-.3+.3/math.sqrt(2.), .3-.3/math.sqrt(2.), math.pi/4.)
        for wheel_y in (-.0741, .0741):
            with self.subTest(wheel_y=wheel_y):
                grid = point_obstacle(world_point(middle, (0., wheel_y)), half_size=.001)
                self.assertTrue(checker.pose_is_collision_free(grid, start))
                self.assertTrue(checker.pose_is_collision_free(grid, end))
                self.assertFalse(checker.pose_is_collision_free(grid, middle))
                self.assertFalse(checker.primitive_is_collision_free(grid, start, 1./radius, distance))

    def test_straight_sweep_retains_wheel_contact_between_free_endpoints(self):
        checker = geometry()
        grid = point_obstacle((0., -.0741), half_size=.001)
        start, end = Pose2D(-.3, 0., 0.), Pose2D(.3, 0., 0.)
        self.assertTrue(checker.pose_is_collision_free(grid, start))
        self.assertTrue(checker.pose_is_collision_free(grid, end))
        self.assertFalse(checker.primitive_is_collision_free(grid, start, 0., .6))

    def test_single_full_size_part_retains_legacy_rectangle_decisions(self):
        part = ((-REAR, -HALF_WIDTH, FRONT, HALF_WIDTH),)
        for padding in (0., .01):
            legacy, explicit = geometry(padding, parts=()), geometry(padding, parts=part)
            for rectangle in ((-.11, .08, -.10, .085), (.25, -.005, .26, .005),
                              (.35, .15, .36, .16)):
                grid = obstacle_grid((rectangle,))
                for pose in (Pose2D(0., 0., 0.), Pose2D(.2, .1, .7)):
                    with self.subTest(padding=padding, rectangle=rectangle, pose=pose):
                        self.assertEqual(legacy.pose_is_collision_free(grid, pose),
                                         explicit.pose_is_collision_free(grid, pose))
                        for curvature in (0., 3.):
                            self.assertEqual(
                                legacy.primitive_is_collision_free(grid, pose, curvature, .4),
                                explicit.primitive_is_collision_free(grid, pose, curvature, .4))


class BodyPartValidationTest(unittest.TestCase):
    def test_nonempty_finite_positive_area_rectangles_are_required(self):
        invalid = ((), (0., 0., .01), (0., 0., .01, .01, .02),
                   (.01, 0., 0., .01), (0., .01, .01, 0.),
                   (0., 0., 0., .01), (0., 0., .01, 0.),
                   (math.nan, 0., .01, .01), (0., 0., math.inf, .01))
        for part in invalid:
            with self.subTest(part=part), self.assertRaises(ValueError):
                RectangularFootprint(FRONT, REAR, HALF_WIDTH, parts=(part,))

    def test_parts_cannot_extend_beyond_the_declared_outer_bounds(self):
        invalid = ((-REAR-.001, 0., 0., .01), (0., 0., FRONT+.001, .01),
                   (0., -HALF_WIDTH-.001, .01, 0.), (0., 0., .01, HALF_WIDTH+.001))
        for part in invalid:
            with self.subTest(part=part), self.assertRaises(ValueError):
                RectangularFootprint(FRONT, REAR, HALF_WIDTH, parts=(part,))


if __name__ == '__main__':
    unittest.main()
