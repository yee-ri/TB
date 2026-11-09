"""XY global routing and short curved-path control from the current pose."""

from dataclasses import dataclass, field, replace
import heapq
import math
from pathlib import Path

import numpy as np
import yaml

from .geometry import CollisionGeometry, OccupancyGrid, Pose2D, normalize_angle
from .tracking import LimitedCommand, limit_tracking_command


def load_navigation_settings(overrides=None):
    """Load navigation defaults and apply configuration overrides."""
    with Path(__file__).with_name('navigation.yaml').open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    settings.update(overrides or {})
    return settings


@dataclass(frozen=True)
class GlobalRoute:
    points: np.ndarray
    edges: np.ndarray = field(init=False, repr=False)
    lengths: np.ndarray = field(init=False, repr=False)
    station: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float).copy()
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                or not np.all(np.isfinite(points))):
            raise ValueError('route points must be a nonempty finite Nx2 array')
        if len(points) > 1:
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
            points = points[keep]
        edges = np.diff(points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        station = np.r_[0., np.cumsum(lengths)]
        for name, value in (('points', points), ('edges', edges),
                            ('lengths', lengths), ('station', station)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class GlobalLocalPlanner:
    def __init__(self, geometry, settings=None):
        if not isinstance(geometry, CollisionGeometry):
            raise TypeError('geometry must be CollisionGeometry')
        self.geometry = geometry
        defaults = load_navigation_settings(settings)
        defaults.setdefault('global_clearance', geometry.footprint.half_width + geometry.footprint.padding)
        for key in ('global_clearance', 'global_clearance_distance',
                    'global_clearance_weight', 'global_soft_cost_weight',
                    'local_lookahead', 'local_horizon', 'local_heading_weight',
                    'local_steering_weight', 'local_clearance_distance',
                    'local_clearance_weight', 'local_rotation_gain'):
            value = defaults[key]
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError('%s must be finite and nonnegative' % key)
            setattr(self, key, float(value))
        for key in ('global_maximum_iterations', 'local_steering_samples', 'local_clearance_samples'):
            value = defaults[key]
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError('%s must be a positive integer' % key)
            setattr(self, key, int(value))
        if self.local_steering_samples < 3 or self.local_steering_samples % 2 != 1:
            raise ValueError('local_steering_samples must be odd and at least three')
        if self.local_clearance_samples < 2:
            raise ValueError('local_clearance_samples must be at least two')
        if min(self.local_lookahead, self.local_horizon, self.local_rotation_gain) <= 0:
            raise ValueError('local lookahead, horizon and rotation gain must be positive')
        if self.global_clearance_distance < self.global_clearance:
            raise ValueError('global_clearance_distance must cover global_clearance')

    @staticmethod
    def _pose(value):
        try:
            pose = value if isinstance(value, Pose2D) else Pose2D(*map(float, value))
            return pose if all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)) else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _route_costs(self, grid):
        """Bounded, vectorized clearance bands; no full-grid Python Dijkstra.

        Coarse occupied cells remain conservative even where the local checker
        has finer wall rectangles. Unknown policy is inherited from the grid.
        """
        radius = self.global_clearance_distance + grid.resolution / math.sqrt(2.)
        cells = int(math.ceil(radius / grid.resolution))
        source = np.pad(grid._occupied, cells, constant_values=True)
        blocked = grid._occupied.copy()
        proximity = np.zeros(grid.data.shape, dtype=float)
        half_diagonal = grid.resolution / math.sqrt(2.)
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                distance = max(0., math.hypot(dx, dy)*grid.resolution - half_diagonal)
                if distance > self.global_clearance_distance:
                    continue
                occupied = source[cells+dy:cells+dy+grid.height,
                                  cells+dx:cells+dx+grid.width]
                if distance <= self.global_clearance:
                    blocked |= occupied
                if self.global_clearance_distance > 0.:
                    penalty = max(0., 1. - distance/self.global_clearance_distance)
                    np.maximum(proximity, occupied*penalty, out=proximity)
        blocked |= self._outside_keep_in(grid)
        costs = (1. + self.global_clearance_weight*proximity
                 + self.global_soft_cost_weight*grid._soft_cost_values
                 / max(1., grid.occupied_threshold))
        return blocked, costs

    def _outside_keep_in(self, grid):
        if not self.geometry.keep_in_rectangles:
            return np.zeros(grid.data.shape, dtype=bool)
        xs = grid.origin_x + (np.arange(grid.width) + .5)*grid.resolution
        ys = grid.origin_y + (np.arange(grid.height) + .5)*grid.resolution
        outside = np.ones(grid.data.shape, dtype=bool)
        for x0, y0, x1, y1 in self.geometry.keep_in_rectangles:
            outside &= ~((ys[:, None] >= y0) & (ys[:, None] <= y1)
                         & (xs[None, :] >= x0) & (xs[None, :] <= x1))
        return outside

    def _line_clear(self, grid, blocked, first, last):
        distance = float(np.linalg.norm(np.asarray(last)-first))
        count = max(1, int(math.ceil(distance/(grid.resolution*.25))))
        points = np.linspace(first, last, count+1)
        columns = np.floor((points[:, 0]-grid.origin_x)/grid.resolution).astype(int)
        rows = np.floor((points[:, 1]-grid.origin_y)/grid.resolution).astype(int)
        if (np.any(columns < 0) or np.any(columns >= grid.width)
                or np.any(rows < 0) or np.any(rows >= grid.height)
                or np.any(blocked[rows, columns])):
            return False
        # A sampled diagonal must not cut between two inflated blocked cells.
        changed = (np.diff(columns) != 0) & (np.diff(rows) != 0)
        if (np.any(blocked[rows[:-1][changed], columns[1:][changed]])
                or np.any(blocked[rows[1:][changed], columns[:-1][changed]])):
            return False
        if self.geometry.keep_in_rectangles:
            inside = np.zeros(len(points), dtype=bool)
            for x0, y0, x1, y1 in self.geometry.keep_in_rectangles:
                inside |= ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                           & (points[:, 1] >= y0) & (points[:, 1] <= y1))
            if not np.all(inside):
                return False
        return True

    @staticmethod
    def _line_cost(grid, traversal, first, last):
        """Integrate the raster cost exactly between crossed cell boundaries."""
        first, last = np.asarray(first), np.asarray(last)
        delta = last-first
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            return 0.
        cuts = [np.array((0., 1.))]
        for axis, origin in enumerate((grid.origin_x, grid.origin_y)):
            if abs(delta[axis]) <= 1e-12:
                continue
            low, high = sorted((first[axis], last[axis]))
            first_boundary = int(math.floor((low-origin)/grid.resolution))+1
            last_boundary = int(math.ceil((high-origin)/grid.resolution))
            boundaries = origin+np.arange(first_boundary, last_boundary)*grid.resolution
            fractions = (boundaries-first[axis])/delta[axis]
            cuts.append(fractions[(fractions > 0.) & (fractions < 1.)])
        cuts = np.unique(np.concatenate(cuts))
        midpoints = first+((cuts[1:]+cuts[:-1])*.5)[:, None]*delta
        columns = np.floor((midpoints[:, 0]-grid.origin_x)/grid.resolution).astype(int)
        rows = np.floor((midpoints[:, 1]-grid.origin_y)/grid.resolution).astype(int)
        return float(length*np.dot(np.diff(cuts), traversal[rows, columns]))

    def global_route(self, grid, start, goal):
        """Return a complete 8-connected 2-D route, or None; never a stub.

        Position clearance is a routing preference/constraint. The route has
        no body headings; local control scores the full body's clearance.
        """
        if not isinstance(grid, OccupancyGrid):
            raise TypeError('grid must be an OccupancyGrid')
        start, goal = self._pose(start), self._pose(goal)
        if start is None or goal is None:
            return None
        first, last = grid.world_to_grid(start.x, start.y), grid.world_to_grid(goal.x, goal.y)
        if not grid.contains_cell(*first) or not grid.contains_cell(*last):
            return None
        if first == last:
            return GlobalRoute(np.array(((start.x, start.y), (goal.x, goal.y))))
        blocked, traversal = self._route_costs(grid)
        if blocked[last[1], last[0]]:
            return None
        escape_blocked = None
        if blocked[first[1], first[0]]:
            # Accept the measured start; leave its inflation through unoccupied cells.
            escape_blocked = grid._occupied | self._outside_keep_in(grid)
            escape_blocked[first[1], first[0]] = False
        points = self._search_cells(grid, blocked, traversal, first, last, escape_blocked)
        if points is None:
            return None
        points[-1] = (goal.x, goal.y)
        if escape_blocked is None:
            points[0] = (start.x, start.y)
            points = self._simplify_route(grid, blocked, traversal, points)
            return None if points is None else GlobalRoute(points)
        exit_index = next(i for i, point in enumerate(points)
                          if not blocked[grid.world_to_grid(*point)[::-1]])
        # Keep the escape chain: a shortcut could re-enter inflation after leaving it.
        suffix = self._simplify_route(grid, blocked, traversal, points[exit_index:])
        if suffix is None:
            return None
        return GlobalRoute(np.vstack(((start.x, start.y), points[:exit_index], suffix)))

    def _search_cells(self, grid, blocked, traversal, first, last, escape_blocked=None):
        """A* visits only XY cells; return the complete parent chain."""
        width = grid.width
        start_id, goal_id = first[1]*width+first[0], last[1]*width+last[0]
        costs = np.full(grid.data.size, math.inf)
        parents = np.full(grid.data.size, -1, dtype=np.int64)
        costs[start_id] = 0.
        queue = [(math.hypot(last[0]-first[0], last[1]-first[1]), 0., start_id)]
        expanded = 0
        neighbours = ((1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
                      (1, 1, math.sqrt(2.)), (1, -1, math.sqrt(2.)),
                      (-1, 1, math.sqrt(2.)), (-1, -1, math.sqrt(2.)))
        while queue and expanded < self.global_maximum_iterations:
            _, cost, current = heapq.heappop(queue)
            if cost > costs[current] + 1e-12:
                continue
            expanded += 1
            if current == goal_id:
                chain = []
                while current != -1:
                    chain.append(grid.grid_to_world(current % width, current // width))
                    current = int(parents[current])
                return np.asarray(chain[::-1])
            x, y = current % width, current // width
            obstacles = (escape_blocked if escape_blocked is not None and blocked[y, x]
                         else blocked)
            for dx, dy, length in neighbours:
                nx, ny = x+dx, y+dy
                if (not 0 <= nx < width or not 0 <= ny < grid.height or obstacles[ny, nx]
                        or (dx and dy and (obstacles[y, nx] or obstacles[ny, x]))):
                    continue
                next_id = ny*width+nx
                candidate = cost + length*.5*(traversal[y, x]+traversal[ny, nx])
                if candidate + 1e-12 >= costs[next_id]:
                    continue
                costs[next_id], parents[next_id] = candidate, current
                heuristic = math.hypot(last[0]-nx, last[1]-ny)
                heapq.heappush(queue, (candidate+heuristic, candidate, next_id))
        return None

    def _simplify_route(self, grid, blocked, traversal, points):
        """Remove a bend only if both hard clearance and route cost survive."""
        route_cost = np.r_[0., np.cumsum([
            self._line_cost(grid, traversal, a, b)
            for a, b in zip(points[:-1], points[1:])])]
        selected, index = [points[0]], 0
        while index < len(points)-1:
            other = len(points)-1
            while other > index:
                clear = self._line_clear(grid, blocked, points[index], points[other])
                if clear and (other == index+1 or
                        self._line_cost(grid, traversal, points[index], points[other])
                        <= route_cost[other]-route_cost[index]+1e-9):
                    break
                other -= 1
            if other == index:
                return None
            selected.append(points[other])
            index = other
        return np.asarray(selected)

    def select_target(self, route, pose, lookahead):
        """Project the current position onto the route and look forward by metres."""
        pose = self._pose(pose)
        if pose is None or not math.isfinite(float(lookahead)) or lookahead < 0.:
            raise ValueError('target selection requires a finite pose and nonnegative lookahead')
        points = route.points
        if len(points) == 1:
            return float(points[0, 0]), float(points[0, 1]), 0., pose.yaw
        edges, lengths, station = route.edges, route.lengths, route.station
        relative = np.asarray((pose.x, pose.y))-points[:-1]
        fractions = np.clip(np.sum(relative*edges, axis=1)/(lengths*lengths), 0., 1.)
        projections = points[:-1] + fractions[:, None]*edges
        index = int(np.argmin(np.sum((projections-(pose.x, pose.y))**2, axis=1)))
        current = station[index]+fractions[index]*lengths[index]
        target_station = min(station[-1], current+lookahead)
        target_index = min(len(edges)-1, int(np.searchsorted(station, target_station, side='right'))-1)
        fraction = (target_station-station[target_index])/lengths[target_index]
        target = points[target_index]+fraction*edges[target_index]
        tangent = math.atan2(edges[target_index, 1], edges[target_index, 0])
        return float(target[0]), float(target[1]), float(station[-1]-current), tangent

    def route_heading(self, route, pose):
        """Direction to the local target, ignoring tiny grid-connection headings."""
        pose = self._pose(pose)
        tx, ty, _, tangent = self.select_target(route, pose, self.local_lookahead)
        dx, dy = tx-pose.x, ty-pose.y
        return math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else tangent

    def local_command(self, grid, pose, route, control, last_linear, last_angular,
                      elapsed, speed_limit=None):
        """Choose a forward arc by route error and full-body obstacle clearance."""
        pose = self._pose(pose)
        values = (last_linear, last_angular, elapsed)
        if pose is None or not all(math.isfinite(float(v)) for v in values):
            return None
        if speed_limit is not None and (not math.isfinite(speed_limit) or speed_limit <= 0.):
            raise ValueError('local speed limit must be finite and positive')
        elapsed = max(0., min(float(elapsed), control['period']))
        tx, ty, remaining, tangent = self.select_target(route, pose, self.local_lookahead)
        dx, dy = tx-pose.x, ty-pose.y
        distance = min(self.local_horizon, max(remaining, math.hypot(dx, dy)))
        if distance <= 1e-9:
            return LimitedCommand(0., 0., 'at route endpoint')
        maximum = 1./self.geometry.minimum_turning_radius
        desired = 2.*(-math.sin(pose.yaw)*dx+math.cos(pose.yaw)*dy)/max(dx*dx+dy*dy, 1e-12)
        previous = last_angular/last_linear if last_linear > 1e-6 else 0.
        curvatures = np.unique(np.r_[np.linspace(-maximum, maximum, self.local_steering_samples),
                                    np.clip(desired, -maximum, maximum), np.clip(previous, -maximum, maximum)])
        ranked = []
        for curvature in curvatures:
            endpoint = self.geometry.propagate(pose, float(curvature), distance)
            score = (math.hypot(endpoint.x-tx, endpoint.y-ty)
                     + self.local_heading_weight*abs(normalize_angle(tangent-endpoint.yaw))
                     + self.local_steering_weight*abs(curvature-previous)/maximum)
            ranked.append((score, abs(curvature), float(curvature)))
        # Use the normal command ramp for an explicit goal-approach speed target.
        decel = control['linear_deceleration']
        if speed_limit is None:
            decel = min(decel, control['safety_linear_deceleration'])
        target_speed = control['cruise_velocity']
        if speed_limit is not None:
            target_speed = min(target_speed, speed_limit)
        ranked = [(0, score, magnitude, curvature, False)
                  for score, magnitude, curvature in ranked]
        heapq.heapify(ranked)
        while ranked:
            overlaps, score, magnitude, curvature, scored = heapq.heappop(ranked)
            if not scored:
                # Prefer fewer overlapping samples before comparing route cost.
                # A candidate remains selectable even when every arc overlaps.
                overlaps, clearance_cost = self._clearance_score(grid, pose, curvature, distance)
                heapq.heappush(ranked, (overlaps, score+clearance_cost, magnitude, curvature, True))
                continue
            command = limit_tracking_command(
                target_speed, max(target_speed, 1e-6), target_speed*curvature,
                last_linear, last_angular, elapsed, control['linear_acceleration'], decel,
                control['angular_acceleration'], control['maximum_angular_velocity'],
                control['maximum_lateral_acceleration'])
            if command.linear_velocity <= 1e-9:
                continue
            return replace(command, reason='following current-pose local arc')
        return None

    def _clearance_score(self, grid, pose, curvature, distance):
        """Rank overlap samples first, then clearance around all body parts."""
        if self.local_clearance_weight == 0. or self.local_clearance_distance == 0.:
            return 0, 0.
        clearances = self.geometry.primitive_clearances(
            grid, pose, curvature, distance, self.local_clearance_distance,
            samples=self.local_clearance_samples)
        shortfall = 1. - clearances / self.local_clearance_distance
        # Averaging the whole arc rewards moving away even when every
        # candidate starts at the same small clearance.
        return (int(np.count_nonzero(clearances == 0.)),
                self.local_clearance_weight * float(np.mean(shortfall * shortfall)))
