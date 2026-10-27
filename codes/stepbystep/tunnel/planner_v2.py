"""Explicit 2-D routing and current-pose local steering, without SE(2) search.

The route is guidance, not an executable vehicle trajectory. Every command is
selected from the measured pose and checked with V1's oriented rectangular
footprint. No previously selected future endpoint is a planning constraint.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .planner import OccupancyGrid, Pose2D, normalize_angle
from .tracking import limit_tracking_command


@dataclass(frozen=True)
class GlobalRoute:
    points: np.ndarray
    expanded_nodes: int = 0

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float).copy()
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                or not np.all(np.isfinite(points))):
            raise ValueError('route points must be a nonempty finite Nx2 array')
        if len(points) > 1:
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
            points = points[keep]
        points.setflags(write=False)
        object.__setattr__(self, 'points', points)

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass(frozen=True)
class LocalCommand:
    linear_velocity: float
    angular_velocity: float
    reason: str
    safe_distance: float
    target: tuple


class GlobalLocalPlanner:
    def __init__(self, planner, settings=None):
        self.planner = planner  # Geometry and physical limits only; never plan().
        defaults = {
            'global_clearance': planner.footprint.half_width + planner.footprint.padding,
            'global_clearance_distance': .25,
            'global_clearance_weight': 1.,
            'global_soft_cost_weight': 1.,
            'global_maximum_iterations': 20000,
            'local_lookahead': .30,
            'local_horizon': .30,
            'local_steering_samples': 15,
            'local_heading_weight': .12,
            'local_steering_weight': .015,
            'local_rotation_gain': 2.,
        }
        defaults.update(settings or {})
        for key in ('global_clearance', 'global_clearance_distance',
                    'global_clearance_weight', 'global_soft_cost_weight',
                    'local_lookahead', 'local_horizon', 'local_heading_weight',
                    'local_steering_weight', 'local_rotation_gain'):
            value = defaults[key]
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError('%s must be finite and nonnegative' % key)
            setattr(self, key, float(value))
        for key in ('global_maximum_iterations', 'local_steering_samples'):
            value = defaults[key]
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError('%s must be a positive integer' % key)
            setattr(self, key, int(value))
        if self.local_steering_samples < 3 or self.local_steering_samples % 2 != 1:
            raise ValueError('local_steering_samples must be odd and at least three')
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
        if self.planner.keep_in_rectangles:
            xs = grid.origin_x + (np.arange(grid.width) + .5)*grid.resolution
            ys = grid.origin_y + (np.arange(grid.height) + .5)*grid.resolution
            inside = np.zeros(grid.data.shape, dtype=bool)
            for x0, y0, x1, y1 in self.planner.keep_in_rectangles:
                inside |= ((ys[:, None] >= y0) & (ys[:, None] <= y1)
                           & (xs[None, :] >= x0) & (xs[None, :] <= x1))
            blocked |= ~inside
        costs = (1. + self.global_clearance_weight*proximity
                 + self.global_soft_cost_weight*grid._soft_cost_values
                 / max(1., grid.occupied_threshold))
        return blocked, costs

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
        if self.planner.keep_in_rectangles:
            inside = np.zeros(len(points), dtype=bool)
            for x0, y0, x1, y1 in self.planner.keep_in_rectangles:
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
        no body headings; local full-footprint checks decide executability.
        """
        if not isinstance(grid, OccupancyGrid):
            raise TypeError('grid must be an OccupancyGrid')
        start, goal = self._pose(start), self._pose(goal)
        if start is None or goal is None:
            return None
        if (not self.planner.pose_is_collision_free(grid, start)
                or not self.planner.pose_is_collision_free(grid, goal)):
            return None
        first, last = grid.world_to_grid(start.x, start.y), grid.world_to_grid(goal.x, goal.y)
        if not grid.contains_cell(*first) or not grid.contains_cell(*last):
            return None
        blocked, traversal = self._route_costs(grid)
        if blocked[first[1], first[0]] or blocked[last[1], last[0]]:
            return None
        if first == last:
            return GlobalRoute(np.array(((start.x, start.y), (goal.x, goal.y))))
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
                points = np.asarray(chain[::-1])
                points[0], points[-1] = (start.x, start.y), (goal.x, goal.y)
                route_cost = np.r_[0., np.cumsum([
                    self._line_cost(grid, traversal, a, b)
                    for a, b in zip(points[:-1], points[1:])])]
                # A geometric shortcut must preserve the A* clearance and
                # soft-obstacle cost, not undo its chosen obstacle avoidance.
                selected, index = [points[0]], 0
                while index < len(points)-1:
                    other = len(points)-1
                    while other > index+1:
                        if (self._line_clear(grid, blocked, points[index], points[other])
                                and self._line_cost(grid, traversal, points[index], points[other])
                                <= route_cost[other]-route_cost[index]+1e-9):
                            break
                        other -= 1
                    if not self._line_clear(grid, blocked, points[index], points[other]):
                        return None
                    selected.append(points[other])
                    index = other
                return GlobalRoute(np.asarray(selected), expanded)
            x, y = current % width, current // width
            for dx, dy, length in neighbours:
                nx, ny = x+dx, y+dy
                if (not 0 <= nx < width or not 0 <= ny < grid.height or blocked[ny, nx]
                        or (dx and dy and (blocked[y, nx] or blocked[ny, x]))):
                    continue
                next_id = ny*width+nx
                candidate = cost + length*.5*(traversal[y, x]+traversal[ny, nx])
                if candidate + 1e-12 >= costs[next_id]:
                    continue
                costs[next_id], parents[next_id] = candidate, current
                heuristic = math.hypot(last[0]-nx, last[1]-ny)
                heapq.heappush(queue, (candidate+heuristic, candidate, next_id))
        return None

    def select_target(self, route, pose, lookahead):
        """Project the current position onto the route and look forward by metres."""
        pose = self._pose(pose)
        if pose is None or not math.isfinite(float(lookahead)) or lookahead < 0.:
            raise ValueError('target selection requires a finite pose and nonnegative lookahead')
        points = route.points
        if len(points) == 1:
            return float(points[0, 0]), float(points[0, 1]), 0., pose.yaw
        edges = np.diff(points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        station = np.r_[0., np.cumsum(lengths)]
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

    def command_is_collision_free(self, grid, pose, command, control):
        """Check the limited command's swept body over one control period.

        The candidate arc has its own distance check. This check covers the
        motion changed by command rate limits, without extrapolating a stop.
        """
        pose = self._pose(pose)
        try:
            linear = float(command.linear_velocity)
            angular = float(command.angular_velocity)
            duration = float(control['period'])
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return False
        if (pose is None or linear < 0. or duration <= 0.
                or not all(math.isfinite(v) for v in (linear, angular, duration))):
            return False
        if linear > 1e-9:
            return self.planner.primitive_is_collision_free(
                grid, pose, angular/linear, linear*duration)
        angle = angular*duration
        radius = math.hypot(max(self.planner.footprint.front, self.planner.footprint.rear)
                            + self.planner.footprint.padding,
                            self.planner.footprint.half_width+self.planner.footprint.padding)
        step = min(self.planner.collision_check_angle, .5*grid.collision_resolution/radius)
        count = max(1, int(math.ceil(abs(angle)/step)))
        return all(self.planner.pose_is_collision_free(
            grid, Pose2D(pose.x, pose.y, pose.yaw+angle*i/count)) for i in range(count+1))

    def _rotation_command(self, grid, pose, target, error, control, last_linear,
                          last_angular, elapsed):
        if abs(error) <= self.planner.goal_heading_tolerance:
            return None
        desired = math.copysign(min(control['maximum_angular_velocity'],
                                   self.local_rotation_gain*abs(error)), error)
        # Preserve the existing actual-command deceleration ramp.
        linear = max(0., last_linear-min(control['linear_deceleration'],
                                        control['safety_linear_deceleration'])*elapsed)
        # The transition can still have forward momentum. Limit lateral
        # acceleration before requesting a faster turn, just as for an arc.
        angular_limit = control['maximum_angular_velocity']
        if linear > 1e-9:
            angular_limit = min(angular_limit, control['maximum_lateral_acceleration']/linear)
        desired = float(np.clip(desired, -angular_limit, angular_limit))
        delta = control['angular_acceleration']*elapsed
        angular = float(np.clip(desired, last_angular-delta, last_angular+delta))
        angular = float(np.clip(angular, -angular_limit, angular_limit))
        command = LocalCommand(linear, angular, 'turning toward route', 0., target)
        if self.command_is_collision_free(grid, pose, command, control):
            return command
        return None

    def local_command(self, grid, pose, route, control, last_linear, last_angular,
                      elapsed):
        """Choose a short checked arc from the current pose, or a checked turn.

        Candidate scoring has no path-end commitment. Invalid geometry returns
        None; obstacles and unknown cells are never cleared to manufacture a
        command. The caller owns completion and final goal-heading alignment.
        """
        pose = self._pose(pose)
        values = (last_linear, last_angular, elapsed)
        if (pose is None or not all(math.isfinite(float(v)) for v in values)
                or not self.planner.pose_is_collision_free(grid, pose)):
            return None
        elapsed = max(0., min(float(elapsed), control['period']))
        tx, ty, remaining, tangent = self.select_target(route, pose, self.local_lookahead)
        target = (tx, ty)
        dx, dy = tx-pose.x, ty-pose.y
        bearing_error = normalize_angle(math.atan2(dy, dx)-pose.yaw)
        if abs(bearing_error) >= math.pi/2.:
            return self._rotation_command(grid, pose, target, bearing_error, control,
                last_linear, last_angular, elapsed)
        distance = min(self.local_horizon, max(remaining, math.hypot(dx, dy)))
        if distance <= 1e-9:
            command = LocalCommand(0., 0., 'at route endpoint', 0., target)
            return command if self.command_is_collision_free(grid, pose, command, control) else None
        maximum = 1./self.planner.minimum_turning_radius
        desired = 2.*(-math.sin(pose.yaw)*dx+math.cos(pose.yaw)*dy)/max(dx*dx+dy*dy, 1e-12)
        previous = last_angular/last_linear if last_linear > 1e-6 else 0.
        curvatures = np.unique(np.r_[np.linspace(-maximum, maximum, self.local_steering_samples),
                                    np.clip(desired, -maximum, maximum), np.clip(previous, -maximum, maximum)])
        ranked = []
        for curvature in curvatures:
            endpoint = self.planner._propagate(pose, float(curvature), distance)
            score = (math.hypot(endpoint.x-tx, endpoint.y-ty)
                     + self.local_heading_weight*abs(normalize_angle(tangent-endpoint.yaw))
                     + self.local_steering_weight*abs(curvature-previous)/maximum)
            ranked.append((score, abs(curvature), float(curvature)))
        # Preserve the existing actual-command deceleration ramp.
        decel = min(control['linear_deceleration'], control['safety_linear_deceleration'])
        target_speed = control['cruise_velocity']
        for _, _, curvature in sorted(ranked):
            if not self.planner.primitive_is_collision_free(grid, pose, curvature, distance):
                continue
            command = limit_tracking_command(
                target_speed, max(target_speed, 1e-6), target_speed*curvature,
                last_linear, last_angular, elapsed, control['linear_acceleration'], decel,
                control['angular_acceleration'], control['maximum_angular_velocity'],
                control['maximum_lateral_acceleration'])
            if command.linear_velocity <= 1e-9:
                continue
            if self.command_is_collision_free(grid, pose, command, control):
                return LocalCommand(command.linear_velocity, command.angular_velocity,
                                    'following current-pose local arc', distance, target)
        return self._rotation_command(grid, pose, target, bearing_error, control,
            last_linear, last_angular, elapsed)
