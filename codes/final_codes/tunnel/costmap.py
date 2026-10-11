#!/usr/bin/env python3
# Ported unchanged from custom_autorace_bringup/tunnel_costmap.py
# Source: /home/sj/tb3_autorace_noetic_ws working tree, 2026-10-08.
"""ROS-independent dynamic occupancy grid for the tunnel mission.

The static map is supplied in ROS ``OccupancyGrid`` row-major order.  A
planning window may be selected with world-coordinate bounds, after which
LaserScan-like measurements can mark and clear dynamic obstacle cells.  This
module deliberately has no ROS imports; transforming a scan into the map frame
and ordering scans by timestamp remain responsibilities of the caller.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScanUpdate:
    """Summary of one :meth:`TunnelCostmap.update_scan` call.

    Cell counts describe state transitions, not the number of laser beams.
    ``hit_beams`` includes finite, in-range returns whose endpoint may lie
    outside the cropped planning grid.
    ``newly_observed_cells`` counts new cumulative free-evidence bits, not
    obstacle changes; an otherwise unchanged scan can still add evidence.
    """

    beam_count: int
    hit_beams: int
    clearing_beams: int
    marked_cells: int
    cleared_cells: int
    decayed_cells: int
    refined_cells: int = 0
    newly_observed_cells: int = 0


class TunnelCostmap:
    """Combine a cropped static grid with LaserScan-derived obstacles.

    Args:
        static_data: Flat, row-major occupancy values, or ``height`` rows of
            ``width`` values.  Values use ROS conventions (-1 unknown, 0 free,
            100 occupied).
        width: Width of the source static grid in cells.
        height: Height of the source static grid in cells.
        resolution: Source grid cell size in metres.
        origin_x: World x coordinate of the source grid's lower-left corner.
        origin_y: World y coordinate of the source grid's lower-left corner.
        planning_bounds: Optional ``(min_x, min_y, max_x, max_y)`` world AABB.
            The crop contains every source cell intersecting this half-open
            box and is clipped to the source map.
        static_occupied_threshold: Static values at or above this value are
            immutable obstacles.
        mark_observations: Consecutive scan updates containing a hit in a cell
            before it becomes a dynamic obstacle.
        clear_observations: Consecutive scan updates ray-clearing a dynamic
            obstacle before it is removed.
        decay_updates: Number of consecutive scan updates in which a confirmed
            dynamic obstacle is not observed before it is removed.  Zero
            disables decay.
        dynamic_occupied_value: Value exported for dynamic obstacles.
        dynamic_inflation_value: Maximum value exported for cells in the
            dynamic obstacle clearance band.  When omitted it equals
            ``dynamic_occupied_value`` for backwards-compatible hard
            inflation.  A value below the planner's occupied threshold keeps
            confirmed endpoints lethal while exposing a linearly graded
            traversal cost, one level per grid-cell radius.
        static_hit_exclusion_radius: Do not mark a finite endpoint whose
            distance to any static occupied cell is at most this many metres.
            Ray clearing before the endpoint is still applied.  Zero disables
            exclusion outside the static occupied cell itself.
        endpoint_clear_guard_radius: For each accepted dynamic endpoint, keep
            cells whose centres are within this radius out of the same scan's
            clearing set.  Only the endpoint itself is marked.  Zero preserves
            the ordinary endpoint-only clearing precedence.
        dynamic_inflation_radius: Inflate confirmed dynamic obstacle cells by
            this radius when exporting planner occupancy.  Distances are
            between cell centres; the raw dynamic layer is unchanged.  Zero
            exports only the confirmed cells themselves.
        dynamic_cell_size: Maximum side of supported sensor-surface subcells.
            Zero retains source-grid geometry. Confirmation and clearing still
            use source cells; adjacent accepted returns within one source-cell
            distance define a surface segment. Isolated returns stay coarse.

    A cell is counted at most once per scan, so ``mark_observations`` and
    ``clear_observations`` are counts of scans rather than counts of beams.
    Marking wins when a cell is both traversed and hit during the same scan.
    """

    def __init__(
        self,
        static_data,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        planning_bounds=None,
        static_occupied_threshold=65,
        mark_observations=1,
        clear_observations=1,
        decay_updates=0,
        dynamic_occupied_value=100,
        dynamic_inflation_value=None,
        static_hit_exclusion_radius=0.0,
        endpoint_clear_guard_radius=0.0,
        dynamic_inflation_radius=0.0,
        dynamic_cell_size=0.0,
    ):
        source_width = self._positive_integer("width", width)
        source_height = self._positive_integer("height", height)
        resolution = self._finite_float("resolution", resolution)
        origin_x = self._finite_float("origin_x", origin_x)
        origin_y = self._finite_float("origin_y", origin_y)
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")

        static_occupied_threshold = self._integer_in_range(
            "static_occupied_threshold", static_occupied_threshold, 0, 100
        )
        mark_observations = self._positive_integer(
            "mark_observations", mark_observations
        )
        clear_observations = self._positive_integer(
            "clear_observations", clear_observations
        )
        decay_updates = self._integer_in_range(
            "decay_updates", decay_updates, 0, 2 ** 31 - 1
        )
        dynamic_occupied_value = self._integer_in_range(
            "dynamic_occupied_value", dynamic_occupied_value, 1, 100
        )
        if dynamic_inflation_value is None:
            dynamic_inflation_value = dynamic_occupied_value
        dynamic_inflation_value = self._integer_in_range(
            "dynamic_inflation_value", dynamic_inflation_value, 1, 100
        )
        if dynamic_inflation_value > dynamic_occupied_value:
            raise ValueError(
                "dynamic_inflation_value cannot exceed "
                "dynamic_occupied_value"
            )
        static_hit_exclusion_radius = self._finite_float(
            "static_hit_exclusion_radius", static_hit_exclusion_radius
        )
        if static_hit_exclusion_radius < 0.0:
            raise ValueError("static_hit_exclusion_radius must be non-negative")
        endpoint_clear_guard_radius = self._finite_float(
            "endpoint_clear_guard_radius", endpoint_clear_guard_radius
        )
        if endpoint_clear_guard_radius < 0.0:
            raise ValueError("endpoint_clear_guard_radius must be non-negative")
        dynamic_inflation_radius = self._finite_float(
            "dynamic_inflation_radius", dynamic_inflation_radius
        )
        if dynamic_inflation_radius < 0.0:
            raise ValueError("dynamic_inflation_radius must be non-negative")
        dynamic_cell_size = self._finite_float("dynamic_cell_size", dynamic_cell_size)
        if dynamic_cell_size < 0.0:
            raise ValueError("dynamic_cell_size must be non-negative")

        source_data = self._flatten_static_data(
            static_data, source_width, source_height
        )
        column_start, column_stop, row_start, row_stop = self._crop_window(
            source_width,
            source_height,
            resolution,
            origin_x,
            origin_y,
            planning_bounds,
        )

        self.width = column_stop - column_start
        self.height = row_stop - row_start
        self.resolution = resolution
        self.origin_x = origin_x + column_start * resolution
        self.origin_y = origin_y + row_start * resolution
        self.source_column = column_start
        self.source_row = row_start
        self.static_occupied_threshold = static_occupied_threshold
        self.mark_observations = mark_observations
        self.clear_observations = clear_observations
        self.decay_updates = decay_updates
        self.dynamic_occupied_value = dynamic_occupied_value
        self.dynamic_inflation_value = dynamic_inflation_value
        self.static_hit_exclusion_radius = static_hit_exclusion_radius
        self.endpoint_clear_guard_radius = endpoint_clear_guard_radius
        self.dynamic_inflation_radius = dynamic_inflation_radius
        # Observation/clearing counters retain their source-map cell identity.
        # Only supported sensor surfaces acquire smaller collision cells.
        self.dynamic_subdivisions = (
            max(1, int(math.ceil(resolution / dynamic_cell_size - 1e-6)))
            if dynamic_cell_size > 0.0 else 1
        )
        self.dynamic_cell_size = resolution / self.dynamic_subdivisions
        self._dynamic_subcells = {}

        # Keep the source mask, rather than only the crop, so an endpoint near
        # a planning-window edge can still be matched to a wall immediately
        # outside that window.
        self._source_width = source_width
        self._source_height = source_height
        self._source_origin_x = origin_x
        self._source_origin_y = origin_y
        self._source_static_occupied = bytes(
            value >= static_occupied_threshold for value in source_data
        )

        cropped_static_data = []
        for row in range(row_start, row_stop):
            first = row * source_width + column_start
            last = row * source_width + column_stop
            cropped_static_data.extend(source_data[first:last])
        self._static_data = tuple(cropped_static_data)

        cell_count = self.width * self.height
        self._static_occupied = bytes(
            value >= static_occupied_threshold for value in self._static_data
        )
        self._dynamic_occupied = bytearray(cell_count)
        self._hit_counts = [0] * cell_count
        self._clear_counts = [0] * cell_count
        self._unobserved_updates = [0] * cell_count
        self._observed_free = bytearray(cell_count)

    def copy_for_update(self):
        """Copy the mutable scan history, sharing only immutable static data.

        The caller can update or reset this candidate without changing a
        costmap currently held by a planner/control snapshot. Configuration
        scalars and the tuple/bytes static layers need no per-scan deepcopy.
        """
        candidate = object.__new__(type(self))
        candidate.__dict__ = self.__dict__.copy()
        candidate._dynamic_occupied = self._dynamic_occupied.copy()
        candidate._hit_counts = self._hit_counts.copy()
        candidate._clear_counts = self._clear_counts.copy()
        candidate._unobserved_updates = self._unobserved_updates.copy()
        candidate._observed_free = self._observed_free.copy()
        candidate._dynamic_subcells = {
            index: None if cells is None else cells.copy()
            for index, cells in self._dynamic_subcells.items()
        }
        return candidate

    @staticmethod
    def _finite_float(name, value):
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("{} must be a finite number".format(name)) from error
        if not math.isfinite(converted):
            raise ValueError("{} must be a finite number".format(name))
        return converted

    @staticmethod
    def _positive_integer(name, value):
        if isinstance(value, bool):
            raise ValueError("{} must be a positive integer".format(name))
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("{} must be a positive integer".format(name)) from error
        if converted != value or converted <= 0:
            raise ValueError("{} must be a positive integer".format(name))
        return converted

    @staticmethod
    def _integer_in_range(name, value, minimum, maximum):
        if isinstance(value, bool):
            raise ValueError("{} must be an integer".format(name))
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("{} must be an integer".format(name)) from error
        if converted != value or converted < minimum or converted > maximum:
            raise ValueError(
                "{} must be an integer in [{}, {}]".format(
                    name, minimum, maximum
                )
            )
        return converted

    @classmethod
    def _flatten_static_data(cls, data, width, height):
        try:
            outer = list(data)
        except TypeError as error:
            raise ValueError("static_data must be an iterable") from error

        flat = outer
        if len(outer) == height and outer:
            nested = []
            rows_are_sequences = True
            for row in outer:
                if isinstance(row, (str, bytes, bytearray)):
                    rows_are_sequences = False
                    break
                try:
                    values = list(row)
                except TypeError:
                    rows_are_sequences = False
                    break
                if len(values) != width:
                    raise ValueError("each static_data row must contain width values")
                nested.extend(values)
            if rows_are_sequences:
                flat = nested

        expected = width * height
        if len(flat) != expected:
            raise ValueError(
                "static_data has {} values; expected {}".format(
                    len(flat), expected
                )
            )

        result = []
        for value in flat:
            if isinstance(value, bool):
                raise ValueError("static occupancy values must be integers")
            try:
                converted = int(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "static occupancy values must be integers"
                ) from error
            if converted != value or converted < -1 or converted > 100:
                raise ValueError("static occupancy values must be in [-1, 100]")
            result.append(converted)
        return result

    @classmethod
    def _crop_window(
        cls,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        planning_bounds,
    ):
        if planning_bounds is None:
            return 0, width, 0, height
        try:
            bounds = tuple(planning_bounds)
        except TypeError as error:
            raise ValueError(
                "planning_bounds must be (min_x, min_y, max_x, max_y)"
            ) from error
        if len(bounds) != 4:
            raise ValueError(
                "planning_bounds must be (min_x, min_y, max_x, max_y)"
            )
        minimum_x, minimum_y, maximum_x, maximum_y = (
            cls._finite_float("planning bound", value) for value in bounds
        )
        if minimum_x >= maximum_x or minimum_y >= maximum_y:
            raise ValueError("planning_bounds minimums must be below maximums")

        # A tiny tolerance prevents an exactly grid-aligned decimal boundary
        # (for example 0.3 / 0.1) from selecting an adjacent cell due solely
        # to binary floating-point representation.
        tolerance = 1e-12
        column_start = math.floor(
            (minimum_x - origin_x) / resolution + tolerance
        )
        column_stop = math.ceil(
            (maximum_x - origin_x) / resolution - tolerance
        )
        row_start = math.floor(
            (minimum_y - origin_y) / resolution + tolerance
        )
        row_stop = math.ceil(
            (maximum_y - origin_y) / resolution - tolerance
        )
        column_start = max(0, min(width, column_start))
        column_stop = max(0, min(width, column_stop))
        row_start = max(0, min(height, row_start))
        row_stop = max(0, min(height, row_stop))
        if column_start >= column_stop or row_start >= row_stop:
            raise ValueError("planning_bounds do not overlap the static grid")
        return column_start, column_stop, row_start, row_stop

    @property
    def shape(self):
        return self.height, self.width

    @property
    def bounds(self):
        """Return the effective, grid-aligned half-open world bounds."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width * self.resolution,
            self.origin_y + self.height * self.resolution,
        )

    @property
    def static_data(self):
        """Return a copy of the cropped static occupancy data."""
        return list(self._static_data)

    @property
    def observed_free_mask(self):
        """Immutable row-major evidence of cells previously ray-observed free.

        This cumulative layer does not override current static/dynamic
        occupancy. It distinguishes measured space from merely static-map
        free space; it is not a continuous visibility proof for a whole cell.
        Cells intersecting the sensor's minimum-range blind disk are not
        newly certified by that scan. No robot-footprint exemption is made.
        """
        return bytes(self._observed_free)

    def reset_dynamic(self):
        """Clear dynamic and free-evidence history, leaving static data intact."""
        cell_count = self.width * self.height
        self._dynamic_occupied = bytearray(cell_count)
        self._hit_counts = [0] * cell_count
        self._clear_counts = [0] * cell_count
        self._unobserved_updates = [0] * cell_count
        self._observed_free = bytearray(cell_count)
        self._dynamic_subcells.clear()

    def refined_dynamic_cells(self):
        """Confirmed sensor-cell rectangles; other cells keep their full area.

        A missing/None surface stays coarse, including an isolated return.
        Static occupied and unknown cells can never be reduced by refinement.
        Geometry persists until the same coarse clear/decay contract removes
        the observation, so looking elsewhere cannot erase a known surface.
        """
        rectangles = {}
        subdivisions = self.dynamic_subdivisions
        if (subdivisions == 1 or (self.dynamic_inflation_radius > 0.0
                and self.dynamic_inflation_value >= self.static_occupied_threshold)):
            return rectangles
        step = self.dynamic_cell_size
        for index, subcells in self._dynamic_subcells.items():
            if (not self._dynamic_occupied[index] or not subcells
                    or self._static_occupied[index] or self._static_data[index] < 0):
                continue
            row, column = divmod(index, self.width)
            cells = []
            for subcell in sorted(subcells):
                subrow, subcolumn = divmod(subcell, subdivisions)
                x = self.origin_x + (column * subdivisions + subcolumn) * step
                y = self.origin_y + (row * subdivisions + subrow) * step
                cells.append((x, y, x + step, y + step))
            rectangles[index] = tuple(cells)
        return rectangles

    def to_collision_occupancy_data(self):
        """Export the same hard geometry for the existing costmap topic.

        Static/source cells are subdivided without shrinking their extent;
        only supported dynamic cells have a more precise occupied subset.
        The planner need not allocate this expanded visualization grid.
        """
        coarse = self.to_occupancy_data()
        subdivisions = self.dynamic_subdivisions
        if subdivisions == 1:
            return self.width, self.height, self.resolution, coarse
        refined = self.refined_dynamic_cells()
        width = self.width * subdivisions
        height = self.height * subdivisions
        values = []
        for row in range(self.height):
            expanded = []
            for value in coarse[row * self.width:(row + 1) * self.width]:
                expanded.extend([value] * subdivisions)
            values.extend(expanded * subdivisions)
        for index in refined:
            row, column = divmod(index, self.width)
            base = self._static_data[index]
            if self.dynamic_inflation_radius > 0.0:
                base = max(base, self.dynamic_inflation_value)
            for subrow in range(subdivisions):
                first = (row * subdivisions + subrow) * width + column * subdivisions
                values[first:first + subdivisions] = [
                    coarse[index] if subrow * subdivisions + subcolumn in self._dynamic_subcells[index]
                    else base for subcolumn in range(subdivisions)
                ]
        return width, height, self.dynamic_cell_size, values

    def to_observed_collision_occupancy_data(self):
        """Export collision geometry with unobserved free space left unknown.

        This separate view never changes the full-route grid. Confirmed hard
        cells retain their original values, and static unknown stays unknown
        even when a ray crossed it. In an unobserved refined source cell, only
        the occupied subcells remain known; its apparent free gaps are unknown.
        Free evidence is cumulative, not a replacement for scan freshness.
        """
        width, height, resolution, values = self.to_collision_occupancy_data()
        subdivisions = self.dynamic_subdivisions
        for row in range(height):
            coarse_row = (row // subdivisions) * self.width
            first = row * width
            for column in range(width):
                index = first + column
                if (not self._observed_free[coarse_row + column // subdivisions]
                        and values[index] < self.static_occupied_threshold):
                    values[index] = -1
        return width, height, resolution, values

    def world_to_cell(self, x, y):
        """Return ``(column, row)`` or ``None`` when outside this crop."""
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        maximum_x = self.origin_x + self.width * self.resolution
        maximum_y = self.origin_y + self.height * self.resolution
        if (
            x < self.origin_x
            or x >= maximum_x
            or y < self.origin_y
            or y >= maximum_y
        ):
            return None
        column = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if 0 <= column < self.width and 0 <= row < self.height:
            return column, row
        return None

    @staticmethod
    def _cell_coordinate(name, value, limit):
        if isinstance(value, bool):
            raise IndexError("{} is outside the grid".format(name))
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise IndexError("{} is outside the grid".format(name)) from error
        if converted != value or converted < 0 or converted >= limit:
            raise IndexError("{} is outside the grid".format(name))
        return converted

    def is_static_occupied(self, column, row):
        column = self._cell_coordinate("column", column, self.width)
        row = self._cell_coordinate("row", row, self.height)
        return bool(self._static_occupied[row * self.width + column])

    def is_dynamic_occupied(self, column, row):
        column = self._cell_coordinate("column", column, self.width)
        row = self._cell_coordinate("row", row, self.height)
        return bool(self._dynamic_occupied[row * self.width + column])

    def _inflation_value_for_offset(
        self, row_offset, column_offset, maximum_offset
    ):
        value = self.dynamic_inflation_value
        if value >= self.dynamic_occupied_value or maximum_offset <= 0:
            return value
        radial_cells = math.hypot(row_offset, column_offset)
        band = max(1, int(math.ceil(radial_cells - 1e-12)))
        remaining_bands = max(1, maximum_offset - band + 1)
        return max(
            1,
            int(round(value * remaining_bands / maximum_offset)),
        )

    def to_occupancy_data(self):
        """Return static, raw-dynamic and clearance-band occupancy data."""
        combined = list(self._static_data)
        maximum_offset = int(
            math.floor(
                self.dynamic_inflation_radius / self.resolution + 1e-12
            )
        )
        squared_radius = (
            self.dynamic_inflation_radius * self.dynamic_inflation_radius
        )
        tolerance = max(1.0, squared_radius) * 1e-15
        for index, occupied in enumerate(self._dynamic_occupied):
            if not occupied:
                continue
            if not self._static_occupied[index]:
                combined[index] = self.dynamic_occupied_value
            source_row, source_column = divmod(index, self.width)
            for row_offset in range(-maximum_offset, maximum_offset + 1):
                row = source_row + row_offset
                if row < 0 or row >= self.height:
                    continue
                distance_y = row_offset * self.resolution
                for column_offset in range(
                    -maximum_offset, maximum_offset + 1
                ):
                    column = source_column + column_offset
                    if column < 0 or column >= self.width:
                        continue
                    distance_x = column_offset * self.resolution
                    if (
                        distance_x * distance_x + distance_y * distance_y
                        > squared_radius + tolerance
                    ):
                        continue
                    inflated_index = row * self.width + column
                    if (
                        not self._static_occupied[inflated_index]
                        and combined[inflated_index] >= 0
                    ):
                        inflation_value = self._inflation_value_for_offset(
                            row_offset, column_offset, maximum_offset
                        )
                        combined[inflated_index] = max(
                            combined[inflated_index],
                            inflation_value,
                        )
        return combined

    def to_soft_cost_data(self):
        """Return only the dynamic clearance band as graded row-major data."""
        costs = [0] * (self.width * self.height)
        if self.dynamic_inflation_value >= self.dynamic_occupied_value:
            return costs
        maximum_offset = int(
            math.floor(
                self.dynamic_inflation_radius / self.resolution + 1e-12
            )
        )
        if maximum_offset <= 0:
            return costs
        squared_radius = (
            self.dynamic_inflation_radius * self.dynamic_inflation_radius
        )
        tolerance = max(1.0, squared_radius) * 1e-15
        for index, occupied in enumerate(self._dynamic_occupied):
            if not occupied:
                continue
            source_row, source_column = divmod(index, self.width)
            for row_offset in range(-maximum_offset, maximum_offset + 1):
                row = source_row + row_offset
                if row < 0 or row >= self.height:
                    continue
                distance_y = row_offset * self.resolution
                for column_offset in range(
                    -maximum_offset, maximum_offset + 1
                ):
                    column = source_column + column_offset
                    if column < 0 or column >= self.width:
                        continue
                    distance_x = column_offset * self.resolution
                    if (
                        distance_x * distance_x + distance_y * distance_y
                        > squared_radius + tolerance
                    ):
                        continue
                    inflated_index = row * self.width + column
                    if (
                        self._static_data[inflated_index] < 0
                        or self._static_occupied[inflated_index]
                        or self._dynamic_occupied[inflated_index]
                    ):
                        continue
                    costs[inflated_index] = max(
                        costs[inflated_index],
                        self._inflation_value_for_offset(
                            row_offset, column_offset, maximum_offset
                        ),
                    )
        return costs

    def to_raw_occupancy_data(self):
        """Return static data plus confirmed endpoints, without inflation."""
        combined = list(self._static_data)
        for index, occupied in enumerate(self._dynamic_occupied):
            if occupied and not self._static_occupied[index]:
                combined[index] = self.dynamic_occupied_value
        return combined

    def update_scan(
        self,
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sensor_pose,
    ):
        """Apply one LaserScan-like observation in the map frame.

        ``sensor_pose`` is exactly ``(x, y, yaw)`` in the same world frame as
        the static map.  A finite return in ``[range_min, range_max]`` clears
        cells before its endpoint and marks the endpoint.  Positive infinity,
        or a finite value above ``range_max``, clears up to ``range_max`` and
        does not mark.  NaN, negative infinity, and values below ``range_min``
        are ignored.

        This method intentionally accepts no timestamp.  The caller must only
        pass scans after placing them in the desired timestamp order.
        """
        angle_min = self._finite_float("angle_min", angle_min)
        angle_increment = self._finite_float(
            "angle_increment", angle_increment
        )
        range_min = self._finite_float("range_min", range_min)
        range_max = self._finite_float("range_max", range_max)
        if range_min < 0.0 or range_max <= range_min:
            raise ValueError(
                "range bounds must satisfy 0 <= range_min < range_max"
            )
        try:
            pose = tuple(sensor_pose)
        except TypeError as error:
            raise ValueError("sensor_pose must be (x, y, yaw)") from error
        if len(pose) != 3:
            raise ValueError("sensor_pose must be (x, y, yaw)")
        sensor_x, sensor_y, sensor_yaw = (
            self._finite_float("sensor pose value", value) for value in pose
        )
        try:
            values = list(ranges)
        except TypeError as error:
            raise ValueError("ranges must be an iterable") from error

        hit_cells = set()
        measured_hit_cells = set()
        clear_cells = set()
        hit_beams = 0
        clearing_beams = 0
        surface_points = {}
        for beam_index, raw_distance in enumerate(values):
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isnan(distance) or distance == -math.inf:
                continue
            if distance < range_min:
                continue

            has_hit = math.isfinite(distance) and distance <= range_max
            ray_length = distance if has_hit else range_max
            if not math.isfinite(ray_length):
                # range_max is validated finite, so this only protects future
                # changes to the accepted metadata contract.
                continue
            angle = angle_min + beam_index * angle_increment + sensor_yaw
            if not math.isfinite(angle):
                continue
            endpoint_x = sensor_x + ray_length * math.cos(angle)
            endpoint_y = sensor_y + ray_length * math.sin(angle)
            endpoint_cell = self.world_to_cell(endpoint_x, endpoint_y)
            ray_cells = self._ray_cells(
                sensor_x, sensor_y, endpoint_x, endpoint_y
            )
            if has_hit:
                hit_beams += 1
                if endpoint_cell is not None:
                    endpoint_index = (
                        endpoint_cell[1] * self.width + endpoint_cell[0]
                    )
                    measured_hit_cells.add(endpoint_index)
                    if not self._endpoint_near_static_obstacle(
                        endpoint_x, endpoint_y
                    ):
                        hit_cells.add(endpoint_index)
                        if self.dynamic_subdivisions > 1:
                            surface_points[beam_index] = (
                                endpoint_x, endpoint_y, endpoint_index
                            )
                        # Only an accepted dynamic hit has endpoint precedence
                        # over clearing.  A static-wall-suppressed endpoint is
                        # a clearing observation, allowing an earlier dynamic
                        # ghost in that free cell to disappear normally.
                        ray_cells.discard(endpoint_index)
            else:
                clearing_beams += 1
            clear_cells.update(ray_cells)

        surface_cells = None
        if self.dynamic_subdivisions > 1:
            surface_cells = {}
            supported_beams = set()
            for beam, first in surface_points.items():
                second = surface_points.get(beam + 1)
                # Connect only adjacent accepted returns separated by no more
                # than one original cell. This fills the observed local surface
                # rather than treating smaller endpoint pixels as isolated dots.
                gap = math.inf if second is None else math.hypot(
                    second[0] - first[0], second[1] - first[1])
                if gap <= 1e-9 or gap > self.resolution:
                    continue
                supported_beams.update((beam, beam + 1))
                for index, subcells in self._surface_subcells(first, second).items():
                    if self._static_occupied[index]:
                        continue
                    surface_cells.setdefault(index, set()).update(subcells)
                    hit_cells.add(index)
            for beam, point in surface_points.items():
                if beam not in supported_beams:
                    # No supported surface extent: retain the complete source
                    # cell, not a point-only obstacle with an invented extent.
                    surface_cells[point[2]] = None

        # An endpoint observed by any beam is occupied even if a neighbouring
        # beam traverses the same raster cell.
        clear_cells.difference_update(hit_cells)
        if self.endpoint_clear_guard_radius > 0.0:
            clear_cells.difference_update(
                self._endpoint_clear_guard_cells(hit_cells)
            )
        marked_cells, cleared_cells, decayed_cells, refined_cells = self._apply_observations(
            hit_cells, clear_cells, surface_cells
        )
        # Reuse the existing ray traversal, without changing its obstacle
        # clearing contract. Even a wall-suppressed return is not evidence
        # that its endpoint cell is free. Range-min blind cells are excluded
        # only from this separate evidence layer, never from hard geometry.
        observed_cells = clear_cells.difference(
            measured_hit_cells,
            self._minimum_range_cells(sensor_x, sensor_y, range_min),
        )
        newly_observed_cells = 0
        for index in observed_cells:
            if not self._static_occupied[index] and not self._observed_free[index]:
                self._observed_free[index] = 1
                newly_observed_cells += 1
        return ScanUpdate(
            beam_count=len(values),
            hit_beams=hit_beams,
            clearing_beams=clearing_beams,
            marked_cells=marked_cells,
            cleared_cells=cleared_cells,
            decayed_cells=decayed_cells,
            refined_cells=refined_cells,
            newly_observed_cells=newly_observed_cells,
        )

    def _minimum_range_cells(self, sensor_x, sensor_y, range_min):
        """Cells intersecting the unmeasured open minimum-range disk."""
        if range_min <= 0.0:
            return set()
        resolution = self.resolution
        first_column = max(0, math.floor((sensor_x - range_min - self.origin_x) / resolution))
        last_column = min(self.width - 1, math.floor((sensor_x + range_min - self.origin_x) / resolution))
        first_row = max(0, math.floor((sensor_y - range_min - self.origin_y) / resolution))
        last_row = min(self.height - 1, math.floor((sensor_y + range_min - self.origin_y) / resolution))
        cells = set()
        for row in range(first_row, last_row + 1):
            lower_y = self.origin_y + row * resolution
            dy = max(lower_y - sensor_y, 0.0, sensor_y - lower_y - resolution)
            for column in range(first_column, last_column + 1):
                lower_x = self.origin_x + column * resolution
                dx = max(lower_x - sensor_x, 0.0, sensor_x - lower_x - resolution)
                if dx * dx + dy * dy < range_min * range_min:
                    cells.add(row * self.width + column)
        return cells

    def _endpoint_clear_guard_cells(self, hit_cells):
        """Return cells protected only from clearing near accepted hits.

        Distances are between OccupancyGrid cell centres.  The returned cells
        are deliberately not added to ``hit_cells``: neighbouring cells retain
        existing dynamic state for this scan but never gain an observation.
        """
        radius = self.endpoint_clear_guard_radius
        maximum_offset = int(math.floor(radius / self.resolution + 1e-12))
        squared_radius = radius * radius
        tolerance = max(1.0, squared_radius) * 1e-15
        guarded_cells = set()
        for hit_index in hit_cells:
            hit_row, hit_column = divmod(hit_index, self.width)
            for row_offset in range(-maximum_offset, maximum_offset + 1):
                row = hit_row + row_offset
                if row < 0 or row >= self.height:
                    continue
                distance_y = row_offset * self.resolution
                for column_offset in range(
                    -maximum_offset, maximum_offset + 1
                ):
                    column = hit_column + column_offset
                    if column < 0 or column >= self.width:
                        continue
                    distance_x = column_offset * self.resolution
                    if (
                        distance_x * distance_x + distance_y * distance_y
                        <= squared_radius + tolerance
                    ):
                        guarded_cells.add(row * self.width + column)
        return guarded_cells

    def _endpoint_near_static_obstacle(self, x, y):
        radius = self.static_hit_exclusion_radius
        resolution = self.resolution
        origin_x = self._source_origin_x
        origin_y = self._source_origin_y

        # ``ceil(...) - 1`` includes the cell whose maximum edge is exactly
        # one radius away from the endpoint.
        minimum_column = (
            int(math.ceil((x - radius - origin_x) / resolution)) - 1
        )
        maximum_column = int(math.floor((x + radius - origin_x) / resolution))
        minimum_row = (
            int(math.ceil((y - radius - origin_y) / resolution)) - 1
        )
        maximum_row = int(math.floor((y + radius - origin_y) / resolution))
        minimum_column = max(0, minimum_column)
        minimum_row = max(0, minimum_row)
        maximum_column = min(self._source_width - 1, maximum_column)
        maximum_row = min(self._source_height - 1, maximum_row)
        if minimum_column > maximum_column or minimum_row > maximum_row:
            return False

        squared_radius = radius * radius
        tolerance = 1e-15
        for row in range(minimum_row, maximum_row + 1):
            cell_minimum_y = origin_y + row * resolution
            cell_maximum_y = cell_minimum_y + resolution
            distance_y = max(cell_minimum_y - y, 0.0, y - cell_maximum_y)
            for column in range(minimum_column, maximum_column + 1):
                source_index = row * self._source_width + column
                if not self._source_static_occupied[source_index]:
                    continue
                cell_minimum_x = origin_x + column * resolution
                cell_maximum_x = cell_minimum_x + resolution
                distance_x = max(
                    cell_minimum_x - x, 0.0, x - cell_maximum_x
                )
                if (
                    distance_x * distance_x + distance_y * distance_y
                    <= squared_radius + tolerance
                ):
                    return True
        return False

    def _surface_subcells(self, first, second):
        """Return closed small rectangles that cover the entire segment."""
        step = self.dynamic_cell_size
        subdivisions = self.dynamic_subdivisions
        x0, y0, x1, y1 = first[0], first[1], second[0], second[1]
        low_x = max(0, int(math.floor((min(x0, x1) - self.origin_x) / step)))
        high_x = min(self.width * subdivisions - 1,
                     int(math.floor((max(x0, x1) - self.origin_x) / step)))
        low_y = max(0, int(math.floor((min(y0, y1) - self.origin_y) / step)))
        high_y = min(self.height * subdivisions - 1,
                     int(math.floor((max(y0, y1) - self.origin_y) / step)))
        result = {}
        for row in range(low_y, high_y + 1):
            for column in range(low_x, high_x + 1):
                xmin, ymin = self.origin_x + column * step, self.origin_y + row * step
                lower, upper = 0.0, 1.0
                for start, delta, minimum in ((x0, x1 - x0, xmin), (y0, y1 - y0, ymin)):
                    if abs(delta) <= 1e-15:
                        if not minimum - 1e-12 <= start <= minimum + step + 1e-12:
                            upper = -1.0
                            break
                    else:
                        a, b = (minimum - start) / delta, (minimum + step - start) / delta
                        lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
                if lower > upper + 1e-12:
                    continue
                index = (row // subdivisions) * self.width + column // subdivisions
                subcell = (row % subdivisions) * subdivisions + column % subdivisions
                result.setdefault(index, set()).add(subcell)
        return result

    def _apply_observations(self, hit_cells, clear_cells, surface_cells=None):
        marked_cells = 0
        cleared_cells = 0
        decayed_cells = 0
        refined_cells = 0
        for index in range(self.width * self.height):
            if self._static_occupied[index]:
                # Static occupied cells are never represented in, or changed
                # by, the dynamic layer.
                continue

            if index in hit_cells:
                if surface_cells is not None:
                    old = self._dynamic_subcells.get(index)
                    old_size = -1 if old is None else len(old)
                    if not self._dynamic_occupied[index] and self._hit_counts[index] == 0:
                        self._dynamic_subcells.pop(index, None)
                    observed = surface_cells.get(index)
                    if index not in self._dynamic_subcells:
                        self._dynamic_subcells[index] = None if observed is None else set(observed)
                    elif observed is None:
                        self._dynamic_subcells[index] = None
                    elif self._dynamic_subcells[index] is not None:
                        self._dynamic_subcells[index].update(observed)
                    new = self._dynamic_subcells[index]
                    new_size = -1 if new is None else len(new)
                    if self._dynamic_occupied[index] and old_size != new_size:
                        refined_cells += 1
                self._unobserved_updates[index] = 0
                self._clear_counts[index] = 0
                self._hit_counts[index] = min(
                    self.mark_observations, self._hit_counts[index] + 1
                )
                if (
                    not self._dynamic_occupied[index]
                    and self._hit_counts[index] >= self.mark_observations
                ):
                    self._dynamic_occupied[index] = 1
                    marked_cells += 1
                continue

            if index in clear_cells:
                self._unobserved_updates[index] = 0
                self._hit_counts[index] = 0
                if self._dynamic_occupied[index]:
                    self._clear_counts[index] = min(
                        self.clear_observations,
                        self._clear_counts[index] + 1,
                    )
                    if self._clear_counts[index] >= self.clear_observations:
                        self._dynamic_occupied[index] = 0
                        self._dynamic_subcells.pop(index, None)
                        self._clear_counts[index] = 0
                        cleared_cells += 1
                else:
                    self._clear_counts[index] = 0
                continue

            self._clear_counts[index] = 0
            if not self._dynamic_occupied[index]:
                # Candidate hits must be consecutive.  Decay only controls
                # the lifetime of an already-confirmed dynamic obstacle.
                self._hit_counts[index] = 0
                self._unobserved_updates[index] = 0
                self._dynamic_subcells.pop(index, None)
                continue
            if self.decay_updates <= 0:
                self._unobserved_updates[index] = 0
                continue
            self._unobserved_updates[index] = min(
                self.decay_updates, self._unobserved_updates[index] + 1
            )
            if self._unobserved_updates[index] >= self.decay_updates:
                was_occupied = bool(self._dynamic_occupied[index])
                self._dynamic_occupied[index] = 0
                self._dynamic_subcells.pop(index, None)
                self._hit_counts[index] = 0
                self._clear_counts[index] = 0
                self._unobserved_updates[index] = 0
                if was_occupied:
                    decayed_cells += 1
        return marked_cells, cleared_cells, decayed_cells, refined_cells

    def _ray_cells(self, start_x, start_y, end_x, end_y):
        clipped = self._clip_segment(start_x, start_y, end_x, end_y)
        if clipped is None:
            return set()
        clipped_start_x, clipped_start_y, clipped_end_x, clipped_end_y = clipped
        start = self.world_to_cell(clipped_start_x, clipped_start_y)
        end = self.world_to_cell(clipped_end_x, clipped_end_y)
        if start is None or end is None:
            return set()
        return {
            row * self.width + column
            for column, row in self._bresenham_cells(start, end)
        }

    def _clip_segment(self, start_x, start_y, end_x, end_y):
        """Liang-Barsky clip against the grid's half-open world bounds."""
        minimum_x, minimum_y, maximum_x, maximum_y = self.bounds
        # nextafter makes the closed Liang-Barsky box match OccupancyGrid's
        # half-open maximum edges and guarantees world_to_cell succeeds.
        # Python 3.8 in ROS Noetic does not provide ``math.nextafter``. Move
        # the maximum edges inward by a scale-aware amount far below any map
        # resolution so the closed clip box matches OccupancyGrid's half-open
        # bounds on every supported runtime.
        maximum_x -= max(1.0, abs(maximum_x), abs(minimum_x)) * 1e-12
        maximum_y -= max(1.0, abs(maximum_y), abs(minimum_y)) * 1e-12
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        entering = 0.0
        leaving = 1.0
        for direction, distance in (
            (-delta_x, start_x - minimum_x),
            (delta_x, maximum_x - start_x),
            (-delta_y, start_y - minimum_y),
            (delta_y, maximum_y - start_y),
        ):
            if direction == 0.0:
                if distance < 0.0:
                    return None
                continue
            ratio = distance / direction
            if direction < 0.0:
                if ratio > leaving:
                    return None
                entering = max(entering, ratio)
            else:
                if ratio < entering:
                    return None
                leaving = min(leaving, ratio)
        if entering > leaving:
            return None
        clipped_start_x = start_x + entering * delta_x
        clipped_start_y = start_y + entering * delta_y
        clipped_end_x = start_x + leaving * delta_x
        clipped_end_y = start_y + leaving * delta_y
        # The interpolation above can round a mathematically interior value
        # back onto a maximum boundary.  Clamp once more to the exact box used
        # for clipping before converting to cells.
        return (
            min(max(clipped_start_x, minimum_x), maximum_x),
            min(max(clipped_start_y, minimum_y), maximum_y),
            min(max(clipped_end_x, minimum_x), maximum_x),
            min(max(clipped_end_y, minimum_y), maximum_y),
        )

    @staticmethod
    def _bresenham_cells(start, end):
        """Yield an inclusive conventional Bresenham raster line."""
        column, row = start
        end_column, end_row = end
        delta_column = abs(end_column - column)
        step_column = 1 if column < end_column else -1
        delta_row = -abs(end_row - row)
        step_row = 1 if row < end_row else -1
        error = delta_column + delta_row
        while True:
            yield column, row
            if column == end_column and row == end_row:
                break
            doubled_error = 2 * error
            if doubled_error >= delta_row:
                error += delta_row
                column += step_column
            if doubled_error <= delta_column:
                error += delta_column
                row += step_row


__all__ = ["ScanUpdate", "TunnelCostmap"]
