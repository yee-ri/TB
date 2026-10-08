"""TB-owned tunnel mission: observed entry, Hybrid A*, odom tracking.

No ROS publishers, AMCL, global course map or other mission dependencies.
Point clouds are supplied in base coordinates with an acquisition-time odom
pose. One lock serializes mapping and the final collision/command decision;
only the stopped Hybrid A* search runs asynchronously on an immutable grid.
"""

import math
import threading
from pathlib import Path

import numpy as np
import yaml

from .costmap import TunnelCostmap
from .planner import HybridAStarPlanner, OccupancyGrid, Pose2D, RectangularFootprint
from .tracking import (TrackingPath, build_speed_profile, calculate_tracking,
                       limit_tracking_command, nearest_path_index, normalize_angle)


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def relative_pose(pose, anchor):
    dx, dy = pose[0] - anchor[0], pose[1] - anchor[1]
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    return Pose2D(c * dx + s * dy, -s * dx + c * dy,
                  normalize_angle(pose[2] - anchor[2]))


def propagate_twist(pose, linear, angular, duration):
    # Exact integration from the source tunnel controller, without ROS.
    angle = angular * duration
    if abs(angular) <= 1e-9:
        return Pose2D(pose.x + linear * duration * math.cos(pose.yaw),
                      pose.y + linear * duration * math.sin(pose.yaw), pose.yaw)
    radius, yaw = linear / angular, pose.yaw + angle
    return Pose2D(pose.x + radius * (math.sin(yaw) - math.sin(pose.yaw)),
                  pose.y - radius * (math.cos(yaw) - math.cos(pose.yaw)),
                  normalize_angle(yaw))


class TunnelMission:
    def __init__(self, config, clock=None):
        self.config = config
        self.clock = clock
        self.control, self.sensors = config['control'], config['sensors']
        for key in ('period', 'linear_acceleration', 'linear_deceleration',
                    'angular_acceleration', 'safety_linear_deceleration',
                    'safety_angular_deceleration', 'safety_reaction_time'):
            if not math.isfinite(self.control[key]) or self.control[key] <= 0:
                raise ValueError('control/%s must be finite and positive' % key)
        if (int(self.sensors['angular_bins']) != self.sensors['angular_bins']
                or self.sensors['angular_bins'] < 8
                or not 0 <= self.sensors['range_min'] < self.sensors['range_max']):
            raise ValueError('invalid LiDAR angular bins or range limits')
        self.footprint = RectangularFootprint(**config['footprint'])
        self.planner = HybridAStarPlanner(self.footprint, **config['planner'])
        goal = config['goal']['pose']
        self.goal = Pose2D(float(goal[0]), float(goal[1]), math.radians(goal[2]))
        self.lock = threading.RLock()
        self.odom = self.cloud = None
        self.lane_stamp = None
        self.lane_count = 0
        self.generation = 0
        self.worker = None
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.state, self.reason = 'ENTRY', 'waiting for sensors'
            self.anchor = self.path = self.costmap = self.grid = None
            self.observed_grid = self.plan_result = None
            self.cloud_stamp = self.last_time = None
            self.version = self.path_version = self.index = self.plans = 0
            self.last_linear = self.last_angular = 0.0
            self.entry_stop = None
            self.lane_count = 0
            self.completion_started = None

    def close(self):
        self.reset()

    def update_odometry(self, pose, linear, angular, stamp):
        values = tuple(float(x) for x in pose) + (float(linear), float(angular), float(stamp))
        if len(values) != 6 or not all(math.isfinite(x) for x in values):
            return False
        with self.lock:
            if self.odom is not None and stamp <= self.odom[-1]:
                return False
            self.odom = values
        return True

    def update_lane(self, visible, stamp):
        with self.lock:
            if not math.isfinite(stamp) or (self.lane_stamp is not None and stamp <= self.lane_stamp):
                return
            gap = math.inf if self.lane_stamp is None else stamp - self.lane_stamp
            self.lane_count = ((self.lane_count if gap <= self.sensors['lane_maximum_age'] else 0) + 1
                               if visible else 0)
            self.lane_stamp = stamp

    def update_cloud(self, points, sensor_pose, odom_pose, stamp):
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points) or not all(math.isfinite(v) for v in (*sensor_pose, *odom_pose, stamp)):
            return False
        ranges = np.linalg.norm(points - np.asarray(sensor_pose[:2]), axis=1)
        points = points[(ranges >= self.sensors['range_min']) & (ranges <= self.sensors['range_max'])]
        if not len(points):
            return False
        with self.lock:
            if self.cloud is not None and stamp <= self.cloud[-1]:
                return False
            self.cloud = (points.copy(), tuple(sensor_pose), tuple(odom_pose), float(stamp))
            if self.anchor is not None:
                self._map_cloud()
        return True

    def _map_cloud(self):
        points, sensor, odom_pose, stamp = self.cloud
        if self.cloud_stamp is not None and stamp <= self.cloud_stamp:
            return
        # Missing angular bins are NaN, never fabricated infinity/free rays.
        delta = points - np.asarray(sensor[:2])
        ranges = np.linalg.norm(delta, axis=1)
        angles = np.arctan2(delta[:, 1], delta[:, 0]) - sensor[2]
        bins = int(self.sensors['angular_bins'])
        angle_step = 2 * math.pi / bins
        indices = np.floor(((angles + math.pi) % (2 * math.pi)) / angle_step).astype(int)
        valid = ((ranges >= self.sensors['range_min']) & (ranges <= self.sensors['range_max']))
        scan = np.full(bins, np.inf)
        np.minimum.at(scan, indices[valid], ranges[valid])
        scan[~np.isfinite(scan)] = np.nan
        pose = relative_pose(odom_pose, self.anchor)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        sensor_local = (pose.x + c * sensor[0] - s * sensor[1],
                        pose.y + s * sensor[0] + c * sensor[1], pose.yaw + sensor[2])
        self.costmap.update_scan(scan, -math.pi + .5 * angle_step, angle_step,
                                 self.sensors['range_min'], self.sensors['range_max'], sensor_local)
        self.cloud_stamp = stamp
        self.version += 1
        self.grid = self.observed_grid = None

    def _initialize(self):
        self.anchor = self.odom[:3]
        cfg = self.config['grid']
        x0, y0, x1, y1 = cfg['bounds']
        resolution = cfg['resolution']
        width, height = int(math.ceil((x1-x0)/resolution)), int(math.ceil((y1-y0)/resolution))
        self.costmap = TunnelCostmap(
            [0] * (width * height), width, height, resolution, x0, y0,
            **{k: v for k, v in cfg.items() if k not in ('bounds', 'resolution')})
        self._map_cloud()

    def _grids(self, pose):
        if self.grid is None:
            c = self.costmap
            # Unseen space can be proposed by search, but does NOT authorize
            # entry or a command: those use observed_grid with unknown blocked.
            values = np.asarray(c.to_occupancy_data()).reshape(c.height, c.width)
            self.grid = OccupancyGrid(values, c.resolution, c.origin_x, c.origin_y,
                                      soft_cost_data=np.asarray(c.to_soft_cost_data()).reshape(c.height, c.width))
            w, h, r, data = c.to_observed_collision_occupancy_data()
            data = np.asarray(data).reshape(h, w)
            # The initial physical body already occupies this space, but the
            # LiDAR cannot ray-clear its own blind disk. Seed ONLY whole cells
            # inside that initial body. Never erase an observed occupied cell
            # or presume unknown space outside the body (including padding).
            xs = c.origin_x + (np.arange(w) + .5) * r
            ys = c.origin_y + (np.arange(h) + .5) * r
            inside = ((xs[None, :] - r/2 >= -self.footprint.rear)
                      & (xs[None, :] + r/2 <= self.footprint.front)
                      & (np.abs(ys[:, None]) + r/2 <= self.footprint.half_width))
            data[(data < 0) & inside] = 0
            self.observed_grid = OccupancyGrid(data, r, c.origin_x, c.origin_y)
        return self.grid, self.observed_grid

    def _zero(self, reason):
        self.reason = reason
        self.last_linear = self.last_angular = 0.0
        return 0.0, 0.0, self.state == 'COMPLETE'

    def _set_path(self, x, y, yaw, curvature, entry=False):
        station = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        cfg = self.control
        speed = build_speed_profile(
            station, curvature, cfg['entry_velocity'] if entry else cfg['cruise_velocity'],
            cfg['minimum_velocity'], cfg['entry_velocity'], cfg['exit_velocity'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'],
            cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'])
        self.path = TrackingPath(np.asarray(x), np.asarray(y), np.asarray(yaw), np.asarray(curvature), station, speed)
        self.index, self.path_version = 0, -1

    def _route_safe(self, grid, index):
        path = self.path
        for i in range(index, len(path.x) - 1):
            start = Pose2D(path.x[i], path.y[i], path.heading[i])
            distance = math.hypot(path.x[i+1]-start.x, path.y[i+1]-start.y)
            if not self.planner.primitive_is_collision_free(grid, start, float(path.curvature[i]), distance):
                return False
        return self.planner.pose_is_collision_free(grid, Pose2D(path.x[-1], path.y[-1], path.heading[-1]))

    def _motion_safe(self, grid, pose, linear, angular):
        # Source controller's measured/requested reaction + braking sweep.
        if not self.planner.pose_is_collision_free(grid, pose):
            return False
        reaction = self.control['safety_reaction_time']
        while reaction > 1e-9 or abs(linear) > 1e-6 or abs(angular) > 1e-6:
            dt = min(.02, self.planner.collision_check_step / max(abs(linear), 1e-9),
                     self.planner.collision_check_angle / max(abs(angular), 1e-9))
            if reaction > 1e-9:
                dt = min(dt, reaction)
            pose = propagate_twist(pose, linear, angular, dt)
            if not self.planner.pose_is_collision_free(grid, pose):
                return False
            if reaction > 1e-9:
                reaction = max(0., reaction-dt)
            else:
                linear = math.copysign(max(0., abs(linear)-self.control['safety_linear_deceleration']*dt), linear)
                angular = math.copysign(max(0., abs(angular)-self.control['safety_angular_deceleration']*dt), angular)
        return self.planner.primitive_is_collision_free(grid, pose, 0., self.control['safety_distance_margin'])

    def _begin_plan(self, grid, pose):
        if self.worker is not None and self.worker.is_alive():
            return
        generation = self.generation
        self.plan_result = None
        self.plans += 1
        self.reason = 'planning from measured rest'

        def search():
            try:
                result = self.planner.plan(grid, pose, self.goal)
                error = '' if result is not None else 'no feasible path'
            except (ValueError, RuntimeError) as exc:
                result, error = None, str(exc)
            with self.lock:
                if self.generation == generation:
                    self.plan_result = (result, error)

        self.worker = threading.Thread(target=search, daemon=True)
        self.worker.start()

    def step(self, now):
        with self.lock:
            if self.clock is not None:
                now = self.clock()
            elapsed = self.control['period'] if self.last_time is None else max(0., now-self.last_time)
            self.last_time = now
            if self.state in ('COMPLETE', 'FAILED'):
                return self._zero(self.reason)
            if self.odom is None or self.cloud is None:
                return self._zero('waiting for odometry and point cloud')
            if any(not -self.sensors['maximum_future_stamp'] <= now-stamp <= self.sensors['maximum_age']
                   for stamp in (self.odom[-1], self.cloud[-1])):
                return self._zero('waiting for fresh odometry and point cloud')
            if self.anchor is None:
                self._initialize()
            pose = relative_pose(self.odom[:3], self.anchor)
            grid, observed = self._grids(pose)
            measured_linear, measured_angular = self.odom[3:5]
            if self.state == 'WAIT_LANE':
                if (self.lane_stamp is not None and self.lane_stamp > self.completion_started
                        and 0 <= now-self.lane_stamp <= self.sensors['lane_maximum_age']
                        and self.lane_count >= self.sensors['lane_confirmation_frames']):
                    self.state, self.reason = 'COMPLETE', 'goal reached; exit lane observed'
                return self._zero(self.reason)
            if self.state == 'ENTRY' and self.path is None:
                entry = self.config['entry']
                distance = self.planner.select_entry_stop(
                    observed, pose, entry['minimum_distance'], entry['maximum_distance'],
                    entry['sample_step'], math.radians(entry['turn_angle_deg']))
                if distance is None:
                    return self._zero('waiting for an observed entry corridor with turning room')
                self.entry_stop = propagate_twist(pose, 1., 0., distance)
                n = int(math.ceil(distance/self.planner.path_sample_step)) + 1
                self._set_path(np.linspace(pose.x, self.entry_stop.x, n),
                               np.linspace(pose.y, self.entry_stop.y, n),
                               np.full(n, pose.yaw), np.zeros(n), entry=True)
            if self.state == 'PLANNING':
                if (abs(measured_linear) > self.control['planning_stopped_linear']
                        or abs(measured_angular) > self.control['planning_stopped_angular']):
                    return self._zero('waiting for measured rest before planning')
                if self.plan_result is None:
                    self._begin_plan(grid, pose)
                    return self._zero(self.reason)
                result, error = self.plan_result
                self.plan_result = None
                if result is None:
                    self.state = 'FAILED'
                    return self._zero(error)
                self._set_path(result.x, result.y, result.yaw, result.curvature)
                self.state = 'FOLLOWING'
            cfg = self.control
            tracking = calculate_tracking(self.path, pose.x, pose.y, pose.yaw, self.index,
                cfg['lookahead_distance'], cfg['maximum_angular_velocity'], cfg['heading_gain'],
                cfg['path_curvature_weight'], cfg['nearest_search_ahead'])
            self.index = tracking.path_index
            target = self.entry_stop if self.state == 'ENTRY' else self.goal
            distance = math.hypot(target.x-pose.x, target.y-pose.y)
            tolerance = self.config['goal']['position_tolerance']
            at_goal = distance <= tolerance and abs(normalize_angle(target.yaw-pose.yaw)) <= math.radians(self.config['goal']['heading_tolerance_deg'])
            if at_goal:
                if self.state == 'ENTRY':
                    self.state, self.path = 'PLANNING', None
                    self.plan_result = None
                    return self._zero('observed entry reached')
                self.state, self.completion_started = 'WAIT_LANE', now
                self.lane_count = 0
                return self._zero('waiting for exit lane at goal')
            if (tracking.position_error > cfg['tracking_position_tolerance']
                    or abs(tracking.heading_error) > math.radians(cfg['tracking_heading_tolerance_deg'])):
                return self._zero('tracking tolerance exceeded')
            if self.path_version != self.version:
                if not self._route_safe(grid, self.index):
                    # Only an actual remaining-route conflict requests search.
                    if self.state != 'ENTRY':
                        self.state, self.path, self.plan_result = 'PLANNING', None, None
                    return self._zero('remaining route intersects an obstacle')
                self.path_version = self.version
            if not self._motion_safe(observed, pose, measured_linear, measured_angular):
                return self._zero('measured stopping region is blocked or unobserved')
            decel = min(cfg['linear_deceleration'], cfg['safety_linear_deceleration'])
            reaction = cfg['safety_reaction_time'] + cfg['period']
            cap = math.sqrt((decel*reaction)**2 + 2*decel*max(0., distance-.5*tolerance)) - decel*reaction
            for scale in (1., .875, .75, .625, .5, .375, .25, .125, 0.):
                command = limit_tracking_command(
                    min(cap, tracking.target_speed*scale), tracking.target_speed, tracking.angular_velocity,
                    self.last_linear, self.last_angular, min(elapsed, cfg['period']),
                    cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'],
                    cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])
                if self._motion_safe(observed, pose, command.linear_velocity, command.angular_velocity):
                    published_at = self.clock() if self.clock is not None else now
                    if (published_at - self.odom[-1] > cfg['safety_reaction_time']
                            or published_at - self.cloud[-1] > self.sensors['maximum_age']):
                        return self._zero('checked inputs expired during control calculation')
                    self.last_linear, self.last_angular = command.linear_velocity, command.angular_velocity
                    self.reason = 'tracking observed free space'
                    return self.last_linear, self.last_angular, False
            return self._zero('requested stopping region is blocked or unobserved')
