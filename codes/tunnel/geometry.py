"""Occupancy grids and body clearance scores for local path selection."""

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
    """Robot envelope and optional component rectangles in the base frame.

    Each part is ``(xmin, ymin, xmax, ymax)`` inside the envelope. Without
    parts, the envelope itself is checked. Padding expands each checked
    rectangle on all four sides.
    """

    front: float
    rear: float
    half_width: float
    padding: float = 0.0
    parts: tuple = ()

    def __post_init__(self):
        if not all(math.isfinite(v) for v in
                   (self.front, self.rear, self.half_width, self.padding)):
            raise ValueError("footprint dimensions must be finite")
        if self.front <= 0.0 or self.rear <= 0.0:
            raise ValueError("footprint front and rear must be positive")
        if self.half_width <= 0.0:
            raise ValueError("footprint half_width must be positive")
        if self.padding < 0.0:
            raise ValueError("footprint padding cannot be negative")
        try:
            parts = tuple(tuple(float(v) for v in part) for part in self.parts)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("footprint parts must contain four finite coordinates") from error
        for part in parts:
            if len(part) != 4 or not all(math.isfinite(v) for v in part):
                raise ValueError("footprint parts must contain four finite coordinates")
            xmin, ymin, xmax, ymax = part
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("footprint parts must have positive length and width")
            if (xmin < -self.rear - 1e-10 or xmax > self.front + 1e-10
                    or ymin < -self.half_width - 1e-10
                    or ymax > self.half_width + 1e-10):
                raise ValueError("footprint parts must stay inside the body envelope")
        object.__setattr__(self, 'parts', parts)


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
    """Body components, turning bound and sampled path clearance scores."""

    def __init__(self, footprint, minimum_turning_radius=.30,
                 inflation_radius=0., keep_in_rectangles=None):
        if not isinstance(footprint, RectangularFootprint):
            footprint = RectangularFootprint(**footprint)
        if minimum_turning_radius <= 0.:
            raise ValueError('minimum_turning_radius must be positive')
        if inflation_radius < 0.:
            raise ValueError('inflation_radius cannot be negative')
        self.footprint = footprint
        parts = footprint.parts or ((-footprint.rear, -footprint.half_width,
                                     footprint.front, footprint.half_width),)
        self._body_rectangles = tuple(
            ((x0+x1)/2., (y0+y1)/2., (x1-x0)/2.+footprint.padding,
             (y1-y0)/2.+footprint.padding)
            for x0, y0, x1, y1 in parts)
        self.minimum_turning_radius = float(minimum_turning_radius)
        self.inflation_radius = float(inflation_radius)
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

    def footprint_geometry(self, pose):
        """Return the padded envelope used for broad-phase and exit checks."""
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

    @staticmethod
    def _rectangle_clearances(footprints, obstacles, limits):
        """Exact OBB/AABB distances, capped by each footprint's current limit."""
        squared = np.square(limits)
        for first in range(0, len(footprints), 32):
            body = footprints[first:first + 32]
            cx, cy, c, s, hx, hy = body.T
            ex, ey = abs(c)*hx + abs(s)*hy, abs(s)*hx + abs(c)*hy
            current = squared[first:first + len(body)]
            for offset in range(0, len(obstacles), 256):
                cells = obstacles[offset:offset + 256]
                dx = cells[None, :, 0] - cx[:, None]
                dy = cells[None, :, 1] - cy[:, None]
                gx = np.maximum(abs(dx)-ex[:, None]-cells[None, :, 2], 0.)
                gy = np.maximum(abs(dy)-ey[:, None]-cells[None, :, 3], 0.)
                i, j = np.nonzero(gx*gx + gy*gy < current[:, None])
                if not len(i):
                    continue
                dx, dy = dx[i, j], dy[i, j]
                cosine, sine, half_x, half_y = c[i], s[i], hx[i], hy[i]
                cell_x, cell_y = cells[j, 2], cells[j, 3]
                overlap = (
                    (abs(dx*cosine+dy*sine) <= half_x+cell_x*abs(cosine)+cell_y*abs(sine)+1e-10)
                    & (abs(-dx*sine+dy*cosine) <= half_y+cell_x*abs(sine)+cell_y*abs(cosine)+1e-10)
                    & (abs(dx) <= ex[i]+cell_x+1e-10)
                    & (abs(dy) <= ey[i]+cell_y+1e-10))
                nearest = np.full(len(i), np.inf)
                # For disjoint rectangles, a closest pair includes a vertex.
                for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                    x = sign_x*half_x*cosine-sign_y*half_y*sine-dx
                    y = sign_x*half_x*sine+sign_y*half_y*cosine-dy
                    gx, gy = np.maximum(abs(x)-cell_x, 0.), np.maximum(abs(y)-cell_y, 0.)
                    np.minimum(nearest, gx*gx+gy*gy, out=nearest)
                    x, y = dx+sign_x*cell_x, dy+sign_y*cell_y
                    gx = np.maximum(abs(x*cosine+y*sine)-half_x, 0.)
                    gy = np.maximum(abs(-x*sine+y*cosine)-half_y, 0.)
                    np.minimum(nearest, gx*gx+gy*gy, out=nearest)
                nearest[overlap] = 0.
                np.minimum.at(current, i, nearest)
        return np.sqrt(squared)

    def primitive_clearances(self, grid, start, curvature, distance, maximum_clearance, samples=9):
        """Return capped body clearances at the start and arc samples, for scoring.

        Samples are uniformly spaced along the arc. Distances include padded
        components, exact occupied rectangles and the map exterior. Overlap
        has zero clearance; this score does not reject a path candidate.
        """
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        start = self._coerce_pose(start)
        curvature, distance, maximum_clearance = map(float, (curvature, distance, maximum_clearance))
        if (not all(math.isfinite(v) for v in (start.x, start.y, start.yaw,
                                              curvature, distance, maximum_clearance))
                or distance < 0. or maximum_clearance < 0.):
            raise ValueError("clearance inputs must be finite with nonnegative distances")
        try:
            sample_count = float(samples)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("clearance samples must be a finite integer of at least 2") from error
        if not math.isfinite(sample_count) or sample_count < 2 or not sample_count.is_integer():
            raise ValueError("clearance samples must be a finite integer of at least 2")
        poses = [start]
        if distance > 0.:
            poses.extend(self.propagate(start, curvature, value)
                         for value in np.linspace(0., distance, int(sample_count))[1:])
        values = np.asarray([(p.x, p.y, p.yaw) for p in poses])
        parts = np.asarray(self._body_rectangles)
        c, s = np.cos(values[:, 2, None]), np.sin(values[:, 2, None])
        cx = values[:, 0, None] + c*parts[None, :, 0] - s*parts[None, :, 1]
        cy = values[:, 1, None] + s*parts[None, :, 0] + c*parts[None, :, 1]
        hx, hy = np.broadcast_to(parts[:, 2], cx.shape), np.broadcast_to(parts[:, 3], cx.shape)
        ex, ey = abs(c)*hx + abs(s)*hy, abs(s)*hx + abs(c)*hy
        boundary = np.minimum.reduce((cx-ex-grid.origin_x, grid.maximum_x-cx-ex,
                                      cy-ey-grid.origin_y, grid.maximum_y-cy-ey))
        limits = np.clip(boundary, 0., maximum_clearance).ravel()
        active = limits > 0.
        if np.any(active):
            footprints = np.stack((cx, cy, np.broadcast_to(c, cx.shape),
                                   np.broadcast_to(s, cx.shape), hx, hy), axis=-1).reshape(-1, 6)
            x0 = float(np.min((cx-ex).ravel()[active]))-maximum_clearance
            y0 = float(np.min((cy-ey).ravel()[active]))-maximum_clearance
            x1 = float(np.max((cx+ex).ravel()[active]))+maximum_clearance
            y1 = float(np.max((cy+ey).ravel()[active]))+maximum_clearance
            c0, r0 = grid.world_to_grid(x0, y0)
            c1, r1 = grid.world_to_grid(x1, y1)
            bounds = (max(0, c0), min(grid.width-1, c1), max(0, r0), min(grid.height-1, r1))
            if grid.occupied_count(*bounds, self.inflation_radius):
                obstacles = grid.occupied_cell_geometry(*bounds, self.inflation_radius)
                limits[active] = self._rectangle_clearances(footprints[active], obstacles, limits[active])
        return np.min(limits.reshape(cx.shape), axis=1)
