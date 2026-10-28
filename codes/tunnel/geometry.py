"""Occupancy grids and swept rectangular-body collision checks."""

from dataclasses import dataclass
import heapq
import math

import numpy as np

_ANGLE_EPSILON = 1e-12

def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class RectangularFootprint:
    """Robot bounds relative to its base pose.

    ``front`` and ``rear`` are intentionally separate because the TurtleBot3
    base frame is not at the geometric centre of its footprint. ``padding`` is
    applied on all four sides before every collision check.
    """

    front: float
    rear: float
    half_width: float
    padding: float = 0.0

    def __post_init__(self):
        if self.front <= 0.0 or self.rear <= 0.0:
            raise ValueError("footprint front and rear must be positive")
        if self.half_width <= 0.0:
            raise ValueError("footprint half_width must be positive")
        if self.padding < 0.0:
            raise ValueError("footprint padding cannot be negative")


class OccupancyGrid:
    """Immutable-shape occupancy-grid snapshot with cached distance fields."""

    def __init__(
        self,
        data,
        resolution,
        origin_x=0.0,
        origin_y=0.0,
        occupied_threshold=50,
        unknown_is_occupied=True,
        soft_cost_data=None,
        dynamic_cell_rectangles=None,
        static_cell_rectangles=None,
    ):
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError("occupancy data must be a two-dimensional array")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("occupancy data cannot be empty")
        if not math.isfinite(float(resolution)) or resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if occupied_threshold < 0:
            raise ValueError("occupied_threshold cannot be negative")

        self.data = values.copy()
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.occupied_threshold = int(occupied_threshold)
        self.unknown_is_occupied = bool(unknown_is_occupied)
        if self.data.dtype == np.bool_:
            occupied = self.data.copy()
        else:
            occupied = self.data >= self.occupied_threshold
        if self.unknown_is_occupied:
            occupied = occupied | (self.data < 0)
        self._occupied = np.asarray(occupied, dtype=bool)
        if soft_cost_data is None:
            soft_cost_values = np.zeros(self.data.shape, dtype=np.float64)
        else:
            soft_cost_values = np.asarray(soft_cost_data, dtype=np.float64)
            if soft_cost_values.shape != self.data.shape:
                raise ValueError(
                    "soft cost data must match occupancy data shape"
                )
            if (
                not np.all(np.isfinite(soft_cost_values))
                or np.any(soft_cost_values < 0.0)
                or np.any(soft_cost_values >= self.occupied_threshold)
            ):
                raise ValueError(
                    "soft cost values must be finite and in "
                    "[0, occupied_threshold)"
                )
        self._soft_cost_values = soft_cost_values.copy()
        self._distance_field = None
        self._inflated_masks = {0.0: self._occupied}
        self._integral_masks = {}
        self._refined_cell_geometry = {}
        self._occupied_geometry_cache = None
        self.collision_resolution = self.resolution
        self._refined_cells = np.zeros(self.data.size, dtype=bool)
        for source, cell_rectangles in (
                ("dynamic", dynamic_cell_rectangles),
                ("static", static_cell_rectangles)):
            for raw_index, rectangles in (cell_rectangles or {}).items():
                index = int(raw_index)
                if index != raw_index or not 0 <= index < self.data.size:
                    raise ValueError("refined {} cell index is outside the grid".format(source))
                if index in self._refined_cell_geometry:
                    raise ValueError("a cell cannot have both static and dynamic refinements")
                row, column = divmod(index, self.width)
                boxes = np.asarray(rectangles, dtype=np.float64)
                if (boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0
                        or not np.all(np.isfinite(boxes))
                        or not self._occupied[row, column]
                        or self.data[row, column] < 0):
                    raise ValueError("refined {} cells require nonempty finite known occupied rectangles".format(source))
                minimum = np.asarray((self.origin_x + column * self.resolution,
                                      self.origin_y + row * self.resolution))
                if (np.any(boxes[:, 2:] <= boxes[:, :2])
                        or np.any(boxes[:, :2] < minimum - 1e-9)
                        or np.any(boxes[:, 2:] > minimum + self.resolution + 1e-9)):
                    raise ValueError("refined rectangles must stay inside their source cell")
                centres = 0.5 * (boxes[:, :2] + boxes[:, 2:])
                halves = 0.5 * (boxes[:, 2:] - boxes[:, :2])
                self._refined_cell_geometry[index] = np.column_stack((centres, halves))
                self._refined_cells[index] = True
                self.collision_resolution = min(self.collision_resolution, float(np.min(2.0 * halves)))

    def occupied_cell_geometry(self, minimum_x, maximum_x, minimum_y, maximum_y,
                               inflation_radius=0.0):
        """Return exact hard rectangles behind the conservative coarse index.

        Only explicitly refined known occupied cells use exact rectangles;
        other static/dynamic cells and unknown cells keep their full extent.
        Positive coarse-grid inflation uses full-cell geometry.
        """
        key = (minimum_x, maximum_x, minimum_y, maximum_y, inflation_radius)
        # A grid is a fixed snapshot. Adjacent swept poses often query the
        # same coarse AABB; retain just that last exact rectangle array.
        # Read/write the key and value together so concurrent planner/control
        # readers cannot pair one query's bounds with another query's cells.
        cached = self._occupied_geometry_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        occupied = self.inflated_mask(inflation_radius)[
            minimum_y:maximum_y + 1, minimum_x:maximum_x + 1]
        rows, columns = np.nonzero(occupied)
        columns, rows = columns + minimum_x, rows + minimum_y
        indices = rows * self.width + columns
        refined = self._refined_cells[indices] if inflation_radius <= 0.0 else np.zeros(indices.shape, dtype=bool)
        coarse = np.column_stack((
            self.origin_x + (columns[~refined] + 0.5) * self.resolution,
            self.origin_y + (rows[~refined] + 0.5) * self.resolution,
            np.full(np.count_nonzero(~refined), 0.5 * self.resolution),
            np.full(np.count_nonzero(~refined), 0.5 * self.resolution),
        ))
        geometry = (np.concatenate([coarse] + [self._refined_cell_geometry[int(index)] for index in indices[refined]])
                    if np.any(refined) else coarse)
        geometry.setflags(write=False)
        self._occupied_geometry_cache = (key, geometry)
        return geometry

    @property
    def width(self):
        return int(self.data.shape[1])

    @property
    def height(self):
        return int(self.data.shape[0])

    @property
    def maximum_x(self):
        return self.origin_x + self.width * self.resolution

    @property
    def maximum_y(self):
        return self.origin_y + self.height * self.resolution

    def world_to_grid(self, x, y):
        """Return integer ``(grid_x, grid_y)`` for a world coordinate."""
        grid_x = int(math.floor((float(x) - self.origin_x) / self.resolution))
        grid_y = int(math.floor((float(y) - self.origin_y) / self.resolution))
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Return the world coordinate at a cell centre."""
        return (
            self.origin_x + (float(grid_x) + 0.5) * self.resolution,
            self.origin_y + (float(grid_y) + 0.5) * self.resolution,
        )

    def contains_cell(self, grid_x, grid_y):
        return (
            0 <= int(grid_x) < self.width
            and 0 <= int(grid_y) < self.height
        )

    def _compute_distance_field(self):
        """Build an 8-neighbour approximation of obstacle distance."""
        distances = np.full(self._occupied.shape, math.inf, dtype=np.float64)
        occupied_y, occupied_x = np.nonzero(self._occupied)
        if occupied_x.size == 0:
            self._distance_field = distances
            return

        queue = []
        for grid_x, grid_y in zip(occupied_x.tolist(), occupied_y.tolist()):
            distances[grid_y, grid_x] = 0.0
            heapq.heappush(queue, (0.0, grid_x, grid_y))

        diagonal = math.sqrt(2.0)
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )
        width, height = self.width, self.height
        while queue:
            distance, grid_x, grid_y = heapq.heappop(queue)
            if distance > distances[grid_y, grid_x] + 1e-12:
                continue
            for offset_x, offset_y, edge_cost in neighbours:
                next_x = grid_x + offset_x
                next_y = grid_y + offset_y
                if not (0 <= next_x < width and 0 <= next_y < height):
                    continue
                candidate = distance + edge_cost
                if candidate + 1e-12 < distances[next_y, next_x]:
                    distances[next_y, next_x] = candidate
                    heapq.heappush(queue, (candidate, next_x, next_y))
        self._distance_field = distances * self.resolution

    @property
    def distance_field(self):
        """Cell-centre distance to the closest occupied cell centre."""
        if self._distance_field is None:
            self._compute_distance_field()
        return self._distance_field

    def inflated_mask(self, inflation_radius=0.0):
        """Return a conservative raster inflation of occupied cells."""
        radius = max(0.0, float(inflation_radius))
        if radius <= 1e-12:
            return self._occupied
        key = round(radius, 12)
        if key not in self._inflated_masks:
            # A cell is marked if its square can intersect the requested
            # inflation disk.  The footprint SAT check then uses its full area.
            cell_half_diagonal = self.resolution / math.sqrt(2.0)
            self._inflated_masks[key] = (
                self.distance_field <= radius + cell_half_diagonal
            )
        return self._inflated_masks[key]

    def occupied_integral(self, inflation_radius=0.0):
        """Return a summed-area table for fast empty rectangle queries."""
        radius = max(0.0, float(inflation_radius))
        key = 0.0 if radius <= 1e-12 else round(radius, 12)
        if key not in self._integral_masks:
            mask = self.inflated_mask(radius)
            integral = np.zeros(
                (self.height + 1, self.width + 1),
                dtype=np.int32,
            )
            integral[1:, 1:] = np.cumsum(
                np.cumsum(mask, axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            )
            self._integral_masks[key] = integral
        return self._integral_masks[key]

    def occupied_count(
        self,
        minimum_x,
        maximum_x,
        minimum_y,
        maximum_y,
        inflation_radius=0.0,
    ):
        """Count occupied cells in an inclusive in-bounds rectangle."""
        integral = self.occupied_integral(inflation_radius)
        first_x = int(minimum_x)
        last_x = int(maximum_x) + 1
        first_y = int(minimum_y)
        last_y = int(maximum_y) + 1
        return int(
            integral[last_y, last_x]
            - integral[first_y, last_x]
            - integral[last_y, first_x]
            + integral[first_y, first_x]
        )


class CollisionGeometry:
    """A rectangular body, turning bound and full swept collision checks."""

    def __init__(self, footprint, minimum_turning_radius=.30,
                 inflation_radius=0., collision_check_step=.025,
                 collision_check_angle=math.radians(3.), keep_in_rectangles=None):
        if not isinstance(footprint, RectangularFootprint):
            footprint = RectangularFootprint(**footprint)
        if minimum_turning_radius <= 0.:
            raise ValueError('minimum_turning_radius must be positive')
        if inflation_radius < 0.:
            raise ValueError('inflation_radius cannot be negative')
        if collision_check_step <= 0. or collision_check_angle <= 0.:
            raise ValueError('collision sampling intervals must be positive')
        self.footprint = footprint
        self.minimum_turning_radius = float(minimum_turning_radius)
        self.inflation_radius = float(inflation_radius)
        self.collision_check_step = float(collision_check_step)
        self.collision_check_angle = float(collision_check_angle)
        self.keep_in_rectangles = self._validated_keep_in_rectangles(keep_in_rectangles)

    @staticmethod
    def _coerce_pose(pose):
        if isinstance(pose, Pose2D):
            return Pose2D(
                float(pose.x),
                float(pose.y),
                normalize_angle(pose.yaw),
            )
        if len(pose) != 3:
            raise ValueError("a pose must contain x, y and yaw")
        return Pose2D(float(pose[0]), float(pose[1]), normalize_angle(pose[2]))

    @staticmethod
    def _validated_keep_in_rectangles(rectangles):
        if rectangles is None:
            return ()
        try:
            rectangles = list(rectangles)
        except TypeError as error:
            raise ValueError("keep_in_rectangles must be a sequence") from error

        validated = []
        for rectangle in rectangles:
            try:
                values = tuple(rectangle)
            except TypeError as error:
                raise ValueError(
                    "each keep-in rectangle must contain four values"
                ) from error
            if len(values) != 4:
                raise ValueError(
                    "each keep-in rectangle must contain four values"
                )
            try:
                values = tuple(float(value) for value in values)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "keep-in rectangle coordinates must be finite"
                ) from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError("keep-in rectangle coordinates must be finite")
            minimum_x, minimum_y, maximum_x, maximum_y = values
            if minimum_x >= maximum_x or minimum_y >= maximum_y:
                raise ValueError(
                    "keep-in rectangle minimums must be below maximums"
                )
            validated.append(values)
        return tuple(validated)

    @staticmethod
    def propagate(start, curvature, distance):
        distance = float(distance)
        curvature = float(curvature)
        if abs(curvature) <= _ANGLE_EPSILON:
            return Pose2D(
                start.x + distance * math.cos(start.yaw),
                start.y + distance * math.sin(start.yaw),
                start.yaw,
            )
        end_yaw_unwrapped = start.yaw + distance * curvature
        return Pose2D(
            start.x
            + (math.sin(end_yaw_unwrapped) - math.sin(start.yaw))
            / curvature,
            start.y
            + (-math.cos(end_yaw_unwrapped) + math.cos(start.yaw))
            / curvature,
            normalize_angle(end_yaw_unwrapped),
        )

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

    def footprint_geometry(self, pose):
        front = self.footprint.front + self.footprint.padding
        rear = self.footprint.rear + self.footprint.padding
        half_width = self.footprint.half_width + self.footprint.padding
        half_length = 0.5 * (front + rear)
        centre_offset = 0.5 * (front - rear)
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        centre_x = pose.x + centre_offset * cosine
        centre_y = pose.y + centre_offset * sine
        return (
            centre_x,
            centre_y,
            cosine,
            sine,
            half_length,
            half_width,
        )

    def _pose_collision_query(self, grid, pose):
        """Return the footprint rectangle and its coarse grid bounds."""
        if not self._pose_is_within_keep_in(pose):
            return None

        epsilon = 1e-10
        (
            centre_x,
            centre_y,
            cosine,
            sine,
            half_length,
            half_width,
        ) = self.footprint_geometry(pose)

        absolute_cosine = abs(cosine)
        absolute_sine = abs(sine)
        extent_x = (
            absolute_cosine * half_length + absolute_sine * half_width
        )
        extent_y = (
            absolute_sine * half_length + absolute_cosine * half_width
        )
        footprint_minimum_x = centre_x - extent_x
        footprint_maximum_x = centre_x + extent_x
        footprint_minimum_y = centre_y - extent_y
        footprint_maximum_y = centre_y + extent_y
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
        return ((centre_x, centre_y, cosine, sine, half_length, half_width),
                (minimum_x, maximum_x, minimum_y, maximum_y))

    def _pose_is_collision_free(self, grid, pose):
        query = self._pose_collision_query(grid, pose)
        if query is None:
            return False
        footprint, bounds = query
        return self._footprints_are_collision_free(grid, bounds, [footprint])

    def _footprints_are_collision_free(self, grid, bounds, footprints):
        """One SAT implementation for up to 32 poses with identical cell bounds."""
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
        # Limit broadcast SAT matrices to at most 32 poses by 256 cells.
        fields = np.asarray(footprints, dtype=np.float64).T[:, :, None]
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
            footprint, current_bounds = query
            if footprints and (current_bounds != bounds or len(footprints) == 32):
                if not self._footprints_are_collision_free(grid, bounds, footprints):
                    return False
                footprints = []
            bounds = current_bounds
            footprints.append(footprint)
        return not footprints or self._footprints_are_collision_free(grid, bounds, footprints)

    def pose_is_collision_free(self, grid, pose):
        """Check an oriented rectangle against bounds and occupied cells."""
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
        sample_step = min(self.collision_check_step, 0.5 * grid.collision_resolution)
        if grid.collision_resolution < grid.resolution - 1e-12:
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
