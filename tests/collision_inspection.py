"""Test-only geometric overlap oracle; driving uses clearance scores."""

import math

import numpy as np

from tunnel.geometry import CollisionGeometry, OccupancyGrid

_ANGLE_EPSILON = 1e-12


def collision_resolution(grid):
    """Smallest refined rectangle width used for swept inspection sampling."""
    return min([grid.resolution] + [float(np.min(2.0 * boxes[:, 2:]))
               for boxes in grid._refined_cell_geometry.values()])


def inspect_geometry(geometry):
    """Inspect the same body and bounds without adding APIs to driving objects."""
    if isinstance(geometry, CollisionInspector):
        return geometry
    return CollisionInspector(
        geometry.footprint,
        minimum_turning_radius=geometry.minimum_turning_radius,
        inflation_radius=geometry.inflation_radius,
        collision_check_step=.01,
        keep_in_rectangles=geometry.keep_in_rectangles,
    )


class CollisionInspector(CollisionGeometry):
    """Offline boolean pose/sweep checks for tests and simulation observers."""

    def __init__(self, footprint, minimum_turning_radius=.30,
                 inflation_radius=0., collision_check_step=.025,
                 collision_check_angle=math.radians(3.), keep_in_rectangles=None):
        super().__init__(footprint, minimum_turning_radius, inflation_radius,
                         keep_in_rectangles)
        if collision_check_step <= 0. or collision_check_angle <= 0.:
            raise ValueError('collision sampling intervals must be positive')
        self.collision_check_step = float(collision_check_step)
        self.collision_check_angle = float(collision_check_angle)

    def _motion_samples(
        self,
        start,
        curvature,
        distance,
        maximum_distance_step,
    ):
        count = self._motion_sample_count(
            curvature, distance, maximum_distance_step
        )
        for index in range(1, count + 1):
            yield self.propagate(
                start,
                curvature,
                distance * index / count,
            )

    def _motion_sample_count(
        self, curvature, distance, maximum_distance_step
    ):
        return max(
            1,
            int(math.ceil(abs(distance) / maximum_distance_step)),
            int(
                math.ceil(
                    abs(distance * curvature) / self.collision_check_angle
                )
            ),
        )

    def _pose_collision_query(self, grid, pose):
        """Return all oriented body rectangles and their shared grid bounds."""
        if not self._pose_is_within_keep_in(pose):
            return None

        epsilon = 1e-10
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        absolute_cosine = abs(cosine)
        absolute_sine = abs(sine)
        footprints, bounds = [], []
        for x, y, half_length, half_width in self._body_rectangles:
            centre_x = pose.x + x*cosine - y*sine
            centre_y = pose.y + x*sine + y*cosine
            extent_x = absolute_cosine*half_length + absolute_sine*half_width
            extent_y = absolute_sine*half_length + absolute_cosine*half_width
            footprints.append((centre_x, centre_y, cosine, sine, half_length, half_width))
            bounds.append((centre_x-extent_x, centre_y-extent_y,
                           centre_x+extent_x, centre_y+extent_y))
        footprint_minimum_x = min(b[0] for b in bounds)
        footprint_minimum_y = min(b[1] for b in bounds)
        footprint_maximum_x = max(b[2] for b in bounds)
        footprint_maximum_y = max(b[3] for b in bounds)
        if (
            footprint_minimum_x < grid.origin_x - epsilon
            or footprint_maximum_x > grid.maximum_x + epsilon
            or footprint_minimum_y < grid.origin_y - epsilon
            or footprint_maximum_y > grid.maximum_y + epsilon
        ):
            return None

        minimum_x = max(
            0,
            int(
                math.floor(
                    (footprint_minimum_x - epsilon - grid.origin_x) / grid.resolution
                )
            ),
        )
        maximum_x = min(
            grid.width - 1,
            int(
                math.floor(
                    (footprint_maximum_x + epsilon - grid.origin_x) / grid.resolution
                )
            ),
        )
        minimum_y = max(
            0,
            int(
                math.floor(
                    (footprint_minimum_y - epsilon - grid.origin_y) / grid.resolution
                )
            ),
        )
        maximum_y = min(
            grid.height - 1,
            int(
                math.floor(
                    (footprint_maximum_y + epsilon - grid.origin_y) / grid.resolution
                )
            ),
        )
        return footprints, (minimum_x, maximum_x, minimum_y, maximum_y)

    def _pose_is_collision_free(self, grid, pose):
        query = self._pose_collision_query(grid, pose)
        if query is None:
            return False
        footprints, bounds = query
        return self._footprints_are_collision_free(grid, bounds, footprints)

    def _footprints_are_collision_free(self, grid, bounds, footprints):
        """Test body rectangles against the same occupied-cell query."""
        if (
            grid.occupied_count(
                *bounds,
                self.inflation_radius,
            )
            == 0
        ):
            return True
        geometry = grid.occupied_cell_geometry(
            *bounds, self.inflation_radius
        )
        if geometry.shape[0] == 0:
            return True
        # Bound broadcast matrices to 32 body rectangles by 256 cells.
        for body in range(0, len(footprints), 32):
            fields = np.asarray(footprints[body:body + 32], dtype=np.float64).T[:, :, None]
            for first in range(0, geometry.shape[0], 256):
                if self._footprints_overlap_cells(fields, geometry[first:first + 256]):
                    return False
        return True

    @staticmethod
    def _footprints_overlap_cells(fields, geometry):
        centre_x, centre_y, cosine, sine, half_length, half_width = fields
        absolute_cosine, absolute_sine = np.abs(cosine), np.abs(sine)
        epsilon = 1e-10
        cell_x, cell_y, cell_half_x, cell_half_y = geometry.T
        delta_x = cell_x - centre_x
        delta_y = cell_y - centre_y

        # Separating-axis test for the robot OBB and axis-aligned grid cells.
        along_robot_x = np.abs(delta_x * cosine + delta_y * sine)
        along_robot_y = np.abs(-delta_x * sine + delta_y * cosine)
        overlap = (
            (
                along_robot_x
                <= half_length
                + cell_half_x * absolute_cosine + cell_half_y * absolute_sine
                + epsilon
            )
            & (
                along_robot_y
                <= half_width
                + cell_half_x * absolute_sine + cell_half_y * absolute_cosine
                + epsilon
            )
            & (
                np.abs(delta_x)
                <= half_length * absolute_cosine
                + half_width * absolute_sine
                + cell_half_x
                + epsilon
            )
            & (
                np.abs(delta_y)
                <= half_length * absolute_sine
                + half_width * absolute_cosine
                + cell_half_y
                + epsilon
            )
        )
        return bool(np.any(overlap))

    def _sampled_poses_are_collision_free(self, grid, poses):
        """Batch consecutive equal-AABB samples without adding or removing cells."""
        footprints, bounds = [], None
        for pose in poses:
            query = self._pose_collision_query(grid, pose)
            if query is None:
                return False
            body, current_bounds = query
            if footprints and (current_bounds != bounds or len(footprints) + len(body) > 32):
                if not self._footprints_are_collision_free(grid, bounds, footprints):
                    return False
                footprints = []
            bounds = current_bounds
            footprints.extend(body)
        return not footprints or self._footprints_are_collision_free(grid, bounds, footprints)

    def pose_is_collision_free(self, grid, pose):
        """Check every body component against bounds and occupied cells."""
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        return self._pose_is_collision_free(grid, self._coerce_pose(pose))

    def _pose_is_within_keep_in(self, pose):
        if not self.keep_in_rectangles:
            return True
        epsilon = 1e-10
        return any(
            minimum_x - epsilon <= pose.x <= maximum_x + epsilon
            and minimum_y - epsilon <= pose.y <= maximum_y + epsilon
            for minimum_x, minimum_y, maximum_x, maximum_y
            in self.keep_in_rectangles
        )

    def _swept_bounds(self, start, curvature, distance):
        """Conservative AABBs of the complete footprint/base circular sweep.

        Each body corner follows a circle about the same instantaneous centre.
        Its arc stays within the endpoint chord plus its sagitta. Expanding the
        two endpoint OBB bounds by the largest corner sagitta therefore covers
        every intermediate rectangle, not only the existing sampled poses.
        Longer arcs retain the ordinary sample-by-sample narrow phase.
        """
        angle = abs(curvature * distance)
        if angle > math.pi:
            return None
        end = self.propagate(start, curvature, distance)
        body_sagitta = base_sagitta = 0.0
        if abs(curvature) > _ANGLE_EPSILON:
            base_radius = 1.0 / abs(curvature)
            body_radius = math.hypot(
                max(self.footprint.front, self.footprint.rear)
                + self.footprint.padding,
                self.footprint.half_width + self.footprint.padding,
            )
            factor = 1.0 - math.cos(0.5 * angle)
            base_sagitta = base_radius * factor
            body_sagitta = (base_radius + body_radius) * factor
        # Include floating-point roundoff at cell and map boundaries in the
        # broad phase. A borderline query falls back to the SAT test.
        body_sagitta += 1e-10
        base_sagitta += 1e-10
        endpoint_bounds = []
        for pose in (start, end):
            x, y, cosine, sine, half_length, half_width = self.footprint_geometry(pose)
            extent_x = abs(cosine) * half_length + abs(sine) * half_width
            extent_y = abs(sine) * half_length + abs(cosine) * half_width
            endpoint_bounds.append((x - extent_x, y - extent_y, x + extent_x, y + extent_y))
        first, last = endpoint_bounds
        body = (
            min(first[0], last[0]) - body_sagitta,
            min(first[1], last[1]) - body_sagitta,
            max(first[2], last[2]) + body_sagitta,
            max(first[3], last[3]) + body_sagitta,
        )
        base = (
            min(start.x, end.x) - base_sagitta,
            min(start.y, end.y) - base_sagitta,
            max(start.x, end.x) + base_sagitta,
            max(start.y, end.y) + base_sagitta,
        )
        return body, base

    def _primitive_region_is_empty(self, grid, start, curvature, distance):
        bounds = self._swept_bounds(start, curvature, distance)
        if bounds is None:
            return False
        body, base = bounds
        if (body[0] < grid.origin_x or body[1] < grid.origin_y
                or body[2] > grid.maximum_x or body[3] > grid.maximum_y):
            return False
        if self.keep_in_rectangles and not any(
            base[0] >= x0 and base[1] >= y0
            and base[2] <= x1 and base[3] <= y1
            for x0, y0, x1, y1 in self.keep_in_rectangles
        ):
            return False
        x0, y0 = grid.world_to_grid(body[0], body[1])
        x1, y1 = grid.world_to_grid(body[2], body[3])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(grid.width - 1, x1), min(grid.height - 1, y1)
        if x0 > x1 or y0 > y1:
            return False
        if grid.occupied_count(x0, x1, y0, y1, self.inflation_radius) == 0:
            return True
        if not grid._refined_cell_geometry or self.inflation_radius > 0.0:
            return False
        # The conservative raster can touch a sweep which stays wholly clear
        # of its surveyed/sensor rectangles. Reject those broad-phase false
        # overlaps before sampling at the smallest refined wall thickness.
        # The same complete corner-arc bounds above cover every swept pose;
        # contact or an uncertain overlap takes the SAT path.
        geometry = grid.occupied_cell_geometry(x0, x1, y0, y1)
        centres, halves = geometry[:, :2], geometry[:, 2:]
        epsilon = 1e-10
        overlap = (np.all(centres + halves >= np.asarray(body[:2]) - epsilon, axis=1)
                   & np.all(centres - halves <= np.asarray(body[2:]) + epsilon, axis=1))
        return not np.any(overlap)

    def _primitive_is_collision_free(
        self,
        grid,
        start,
        curvature,
        distance,
        check_start,
    ):
        if check_start and not self._pose_is_collision_free(grid, start):
            return False
        if self._primitive_region_is_empty(grid, start, curvature, distance):
            return True
        resolution = collision_resolution(grid)
        sample_step = min(self.collision_check_step, 0.5 * resolution)
        if resolution < grid.resolution - 1e-12:
            radius = math.hypot(
                max(self.footprint.front, self.footprint.rear) + self.footprint.padding,
                self.footprint.half_width + self.footprint.padding,
            )
            sample_step /= 1.0 + abs(curvature) * radius
        return self._sampled_poses_are_collision_free(
            grid, self._motion_samples(start, curvature, distance, sample_step))

    def primitive_is_collision_free(
        self,
        grid,
        start,
        curvature,
        distance,
    ):
        """Check the complete swept primitive, including both endpoints."""
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        start = self._coerce_pose(start)
        distance = float(distance)
        if distance < 0.0:
            raise ValueError("forward primitive distance cannot be negative")
        return self._primitive_is_collision_free(
            grid,
            start,
            float(curvature),
            distance,
            check_start=True,
        )
