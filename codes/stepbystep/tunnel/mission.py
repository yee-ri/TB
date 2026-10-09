"""TB-owned tunnel mission: straight entry, Hybrid A*, odom tracking.

No ROS publishers, AMCL or other mission controllers.
LiDAR inputs share one scan representation with an acquisition-time odom
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
                       geometry_from_poses, limit_tracking_command,
                       normalize_angle, quintic_pose_path)


def load_config(path, profile='normal'):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    profiles = config.pop('profiles', {'normal': {}})
    if profile not in profiles:
        raise ValueError('unknown tunnel profile: ' + str(profile))
    for section, values in profiles[profile].items():
        config[section].update(values)
    static_map = config['grid'].get('static_map')
    if static_map is not None:
        static_map['image'] = str((Path(path).resolve().parent / static_map['image']).resolve())
    return config


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
                    'safety_angular_deceleration', 'safety_reaction_time',
                    'plan_retry_period'):
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
        planning_goal = config['goal'].get('planning_pose', goal)
        self.planning_goal = Pose2D(float(planning_goal[0]), float(planning_goal[1]),
                                   math.radians(planning_goal[2]))
        self.lock = threading.RLock()
        self.odom = self.scan = None
        self.front_clearance = math.inf
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
            self.plan_result = None
            self.last_plan_attempt = None
            self.mapped_stamp = self.last_time = None
            self.version = self.path_version = self.index = self.plans = 0
            self.last_linear = self.last_angular = 0.0
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
        # Missing angular bins are NaN, never fabricated infinity/free rays.
        delta = points - np.asarray(sensor_pose[:2])
        ranges = np.linalg.norm(delta, axis=1)
        angles = np.arctan2(delta[:, 1], delta[:, 0]) - sensor_pose[2]
        bins = int(self.sensors['angular_bins'])
        angle_step = 2 * math.pi / bins
        indices = np.floor(((angles + math.pi) % (2 * math.pi)) / angle_step).astype(int)
        valid = ((ranges >= self.sensors['range_min']) & (ranges <= self.sensors['range_max']))
        scan = np.full(bins, np.inf)
        np.minimum.at(scan, indices[valid], ranges[valid])
        scan[~np.isfinite(scan)] = np.nan
        return self.update_scan(scan, -math.pi + .5 * angle_step, angle_step,
                                self.sensors['range_min'], self.sensors['range_max'],
                                sensor_pose, odom_pose, stamp)

    def update_scan(self, ranges, angle_min, angle_increment, range_min, range_max,
                    sensor_pose, odom_pose, stamp):
        metadata = (angle_min, angle_increment, range_min, range_max, *sensor_pose, *odom_pose, stamp)
        if (len(sensor_pose) != 3 or len(odom_pose) != 3
                or not all(math.isfinite(v) for v in metadata)
                or angle_increment == 0 or not 0 <= range_min < range_max):
            return False
        effective_max = min(range_max, self.sensors['range_max'])
        if effective_max <= range_min:
            return False
        ranges = np.asarray(ranges, dtype=float).reshape(-1).copy()
        hits = np.isfinite(ranges)
        hits[hits] = (ranges[hits] >= range_min) & (ranges[hits] <= range_max)
        valid = hits | np.isposinf(ranges)
        if not np.any(valid):
            return False
        ranges[~valid] = np.nan
        angles = angle_min + np.flatnonzero(hits) * angle_increment + sensor_pose[2]
        x = sensor_pose[0] + ranges[hits] * np.cos(angles)
        y = sensor_pose[1] + ranges[hits] * np.sin(angles)
        ahead = (x >= 0) & (np.abs(y) <= self.footprint.half_width + self.footprint.padding)
        clearance = float(np.min(x[ahead])) - self.footprint.front if np.any(ahead) else math.inf
        with self.lock:
            if self.scan is not None and stamp <= self.scan[-1]:
                return False
            self.scan = (ranges, float(angle_min), float(angle_increment), float(range_min),
                         float(effective_max), tuple(sensor_pose), tuple(odom_pose), float(stamp))
            self.front_clearance = clearance
            if self.anchor is not None:
                self._map_scan()
        return True

    def _map_scan(self):
        ranges, angle_min, increment, range_min, range_max, sensor, odom_pose, stamp = self.scan
        if self.mapped_stamp is not None and stamp <= self.mapped_stamp:
            return
        pose = relative_pose(odom_pose, self.anchor)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        sensor_local = (pose.x + c * sensor[0] - s * sensor[1],
                        pose.y + s * sensor[0] + c * sensor[1], pose.yaw + sensor[2])
        update = self.costmap.update_scan(ranges, angle_min, increment, range_min, range_max, sensor_local)
        self.mapped_stamp = stamp
        if (update.marked_cells or update.cleared_cells or update.decayed_cells
                or update.refined_cells or update.newly_observed_cells):
            self.version += 1
            self.grid = None

    def _initialize(self):
        self.anchor = self.odom[:3]
        cfg = self.config['grid']
        x0, y0, x1, y1 = cfg['bounds']
        resolution = cfg['resolution']
        width, height = int(math.ceil((x1-x0)/resolution)), int(math.ceil((y1-y0)/resolution))
        static_data = [0] * (width * height)
        static_map = cfg.get('static_map')
        if static_map is not None:
            import cv2
            pixels = cv2.imread(static_map['image'], cv2.IMREAD_GRAYSCALE)
            if pixels is None:
                raise ValueError('cannot read static tunnel map: ' + static_map['image'])
            # This copied source PGM contains black occupied / 254 free cells.
            static_data = np.rot90(np.flipud((pixels == 0).astype(np.int8) * 100),
                                   int(static_map['quarter_turns']))
            height, width = static_data.shape
            x0, y0 = static_map['origin']
        self.costmap = TunnelCostmap(
            static_data, width, height, resolution, x0, y0, planning_bounds=cfg['bounds'],
            **{k: v for k, v in cfg.items() if k not in ('bounds', 'resolution', 'static_map')})
        self._map_scan()

    def _grid(self):
        if self.grid is None:
            c = self.costmap
            # Static walls and measured obstacles share one collision grid;
            # unknown-only cells do not reject the robot's sensor blind area.
            values = np.asarray(c.to_occupancy_data()).reshape(c.height, c.width)
            self.grid = OccupancyGrid(values, c.resolution, c.origin_x, c.origin_y,
                                      occupied_threshold=c.static_occupied_threshold,
                                      soft_cost_data=np.asarray(c.to_soft_cost_data()).reshape(c.height, c.width),
                                      dynamic_cell_rectangles=c.refined_dynamic_cells())
        return self.grid

    def _zero(self, reason):
        self.reason = reason
        self.last_linear = self.last_angular = 0.0
        return 0.0, 0.0, self.state == 'COMPLETE'

    def _tracking_path(self, x, y, yaw, curvature):
        station = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        cfg = self.control
        speed = build_speed_profile(
            station, curvature, cfg['cruise_velocity'],
            cfg['minimum_velocity'], cfg['entry_velocity'], cfg['exit_velocity'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'],
            cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'])
        return TrackingPath(np.asarray(x), np.asarray(y), np.asarray(yaw), np.asarray(curvature), station, speed)

    def _planned_path(self, plan, grid):
        """Source internal goal + prevalidated moving exit, one tracking path."""
        if 'planning_pose' not in self.config['goal']:
            return self._tracking_path(plan.x, plan.y, plan.yaw, plan.curvature)
        terminal = Pose2D(float(plan.x[-1]), float(plan.y[-1]), float(plan.yaw[-1]))
        target, cfg = self.goal, self.config['exit_connector']
        forward = np.asarray((math.cos(self.planning_goal.yaw), math.sin(self.planning_goal.yaw)))
        if np.dot((target.x-terminal.x, target.y-terminal.y), forward) <= 1e-6:
            raise ValueError('outside pose is not ahead of the terminal pose')
        distance = math.hypot(target.x-terminal.x, target.y-terminal.y)
        geometric_maximum = .24 * distance
        tangent = max(min(cfg['minimum_tangent_length'], geometric_maximum),
                      min(cfg['tangent_ratio'] * distance,
                          min(cfg['maximum_tangent_length'], geometric_maximum)))
        poses = quintic_pose_path(
            (terminal.x, terminal.y, terminal.yaw), (target.x, target.y, target.yaw),
            tangent, tangent, max(3, int(math.ceil(distance/cfg['sample_step']))+1))
        heading, curvature, _ = geometry_from_poses(poses)
        if np.any(np.diff(poses[:, :2], axis=0) @ forward <= 1e-9):
            raise ValueError('exit connector is not strictly forward')
        if np.max(np.abs(curvature)) > 1./self.planner.minimum_turning_radius + 1e-6:
            raise ValueError('exit connector exceeds the minimum turning radius')
        # Keep all exact Hybrid A* primitive curvatures, including its zero-
        # curvature terminal sample. Rebuild speed once over the joined path.
        path = self._tracking_path(
            np.r_[plan.x, poses[1:, 0]], np.r_[plan.y, poses[1:, 1]],
            np.r_[plan.yaw, heading[1:]], np.r_[plan.curvature, curvature[1:]])
        if not self._route_safe(grid, 0, path):
            raise ValueError('exit connector is not collision-free')
        return path

    def _command(self, command, now, reason):
        published_at = self.clock() if self.clock is not None else now
        if (published_at - self.odom[-1] > self.control['safety_reaction_time']
                or published_at - self.scan[-1] > self.sensors['maximum_age']):
            return self._zero('checked inputs expired during control calculation')
        self.last_linear, self.last_angular = command.linear_velocity, command.angular_velocity
        self.reason = reason
        return self.last_linear, self.last_angular, False

    def _enter(self, pose, elapsed, now):
        cfg, entry = self.control, self.config['entry']
        remaining = min(entry['distance'] - pose.x,
                        self.front_clearance - entry['obstacle_clearance'])
        if remaining <= cfg['safety_distance_margin']:
            command = self._stop_command(elapsed)
            if command.linear_velocity > 1e-9 or abs(command.angular_velocity) > 1e-9:
                return self._command(command, now, 'stopping at straight entry target')
            self.state = 'PLANNING'
            return self._zero('straight entry complete')
        # Entry velocity is the initial profile speed, not a whole-leg cap.
        # Match the source's distance ramp and reaction + braking limit while
        # retaining this port's requested straight entry and stopping target.
        reference = min(entry['cruise_velocity'], math.sqrt(
            cfg['entry_velocity']**2 + 2 * cfg['linear_acceleration'] * max(0., pose.x)))
        decel = min(cfg['linear_deceleration'], cfg['safety_linear_deceleration'])
        reaction = cfg['safety_reaction_time'] + cfg['period']
        remaining = max(0., remaining - .5 * cfg['safety_distance_margin'])
        cap = math.sqrt((decel * reaction)**2 + 2 * decel * remaining) - decel * reaction
        command = limit_tracking_command(
            min(reference, cap), reference, -cfg['heading_gain'] * pose.yaw,
            self.last_linear, self.last_angular, min(elapsed, cfg['period']),
            cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])
        return self._command(command, now, 'straight entry')

    def _stop_command(self, elapsed):
        cfg = self.control
        return limit_tracking_command(
            0., 1., 0., self.last_linear, self.last_angular,
            min(elapsed, cfg['period']), cfg['linear_acceleration'],
            cfg['linear_deceleration'], cfg['angular_acceleration'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])

    def _route_safe(self, grid, index, path=None):
        path = self.path if path is None else path
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
            if grid.collision_resolution < grid.resolution - 1e-12:
                radius = math.hypot(max(self.footprint.front, self.footprint.rear)
                                    + self.footprint.padding,
                                    self.footprint.half_width + self.footprint.padding)
                corner_speed = abs(linear) + radius * abs(angular)
                if corner_speed > 1e-9:
                    dt = min(dt, .5 * grid.collision_resolution / corner_speed)
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

    def _begin_plan(self, grid, pose, now):
        if self.worker is not None and self.worker.is_alive():
            return
        if (self.last_plan_attempt is not None
                and now - self.last_plan_attempt < self.control['plan_retry_period']):
            return
        self.last_plan_attempt = now
        generation = self.generation
        self.plan_result = None
        self.plans += 1
        self.reason = 'planning from measured rest'

        def search():
            try:
                result = self.planner.plan(grid, pose, self.planning_goal)
                error = '' if result is not None else 'no feasible path'
                if result is not None:
                    result = self._planned_path(result, grid)
            except (ValueError, RuntimeError) as exc:
                result, error = None, str(exc)
            with self.lock:
                if self.generation == generation:
                    self.plan_result = (result, error, pose)

        self.worker = threading.Thread(target=search, daemon=True)
        self.worker.start()

    def step(self, now):
        with self.lock:
            if self.clock is not None:
                now = self.clock()
            elapsed = self.control['period'] if self.last_time is None else max(0., now-self.last_time)
            self.last_time = now
            if self.state == 'COMPLETE':
                return self._zero(self.reason)
            if self.odom is None or self.scan is None:
                return self._zero('waiting for odometry and lidar')
            if any(not -self.sensors['maximum_future_stamp'] <= now-stamp <= self.sensors['maximum_age']
                   for stamp in (self.odom[-1], self.scan[-1])):
                return self._zero('waiting for fresh odometry and lidar')
            if self.anchor is None:
                self._initialize()
            pose = relative_pose(self.odom[:3], self.anchor)
            measured_linear, measured_angular = self.odom[3:5]
            if self.state == 'WAIT_LANE':
                if (self.lane_stamp is not None and self.lane_stamp > self.completion_started
                        and 0 <= now-self.lane_stamp <= self.sensors['lane_maximum_age']
                        and self.lane_count >= self.sensors['lane_confirmation_frames']):
                    self.state, self.reason = 'COMPLETE', 'goal reached; exit lane observed'
                return self._zero(self.reason)
            if self.state == 'ENTRY':
                return self._enter(pose, elapsed, now)
            grid = self._grid()
            if self.state == 'PLANNING':
                if (abs(measured_linear) > self.control['planning_stopped_linear']
                        or abs(measured_angular) > self.control['planning_stopped_angular']):
                    return self._zero('waiting for measured rest before planning')
                if self.plan_result is None:
                    self._begin_plan(grid, pose, now)
                    return self._zero(self.reason)
                result, error, start = self.plan_result
                self.plan_result = None
                if result is None:
                    # The source retries from rest against the latest grid;
                    # one unsuccessful search is not a terminal mission failure.
                    return self._zero(error)
                if (math.hypot(pose.x-start.x, pose.y-start.y)
                        > self.control['planning_start_position_tolerance']
                        or abs(normalize_angle(pose.yaw-start.yaw))
                        > math.radians(self.control['planning_start_heading_tolerance_deg'])):
                    self.last_plan_attempt = None
                    return self._zero('discarding plan after stopped pose shifted')
                self.path = result
                self.index, self.path_version = 0, -1
                self.state = 'FOLLOWING'
            # Match the source's decision order. A transient endpoint under
            # the robot holds the same path until later free rays clear it;
            # it must not first erase the path and force a colliding-start plan.
            if not self.planner.pose_is_collision_free(grid, pose):
                return self._zero('localized footprint overlaps an obstacle')
            cfg = self.control
            tracking = calculate_tracking(self.path, pose.x, pose.y, pose.yaw, self.index,
                cfg['lookahead_distance'], cfg['maximum_angular_velocity'], cfg['heading_gain'],
                cfg['path_curvature_weight'], cfg['nearest_search_ahead'])
            self.index = tracking.path_index
            target = self.goal
            distance = math.hypot(target.x-pose.x, target.y-pose.y)
            tolerance = self.config['goal']['position_tolerance']
            at_goal = distance <= tolerance and abs(normalize_angle(target.yaw-pose.yaw)) <= math.radians(self.config['goal']['heading_tolerance_deg'])
            if at_goal:
                command = self._stop_command(elapsed)
                if (not self._motion_safe(grid, pose, measured_linear, measured_angular)
                        or not self._motion_safe(grid, pose, command.linear_velocity, command.angular_velocity)):
                    return self._zero('goal stopping region intersects an obstacle')
                if command.linear_velocity > 1e-9 or abs(command.angular_velocity) > 1e-9:
                    return self._command(command, now, 'stopping at outside goal')
                self.state, self.completion_started = 'WAIT_LANE', now
                self.lane_count = 0
                return self._zero('waiting for exit lane at goal')
            if (tracking.position_error > cfg['tracking_position_tolerance']
                    or abs(tracking.heading_error) > math.radians(cfg['tracking_heading_tolerance_deg'])):
                return self._zero('tracking tolerance exceeded')
            if self.path_version != self.version:
                if not self._route_safe(grid, self.index):
                    # Only an actual remaining-route conflict requests search.
                    self.state, self.path, self.plan_result = 'PLANNING', None, None
                    self.last_plan_attempt = None
                    return self._zero('remaining route intersects an obstacle')
                self.path_version = self.version
            if not self._motion_safe(grid, pose, measured_linear, measured_angular):
                return self._zero('measured stopping region intersects an obstacle')
            decel = min(cfg['linear_deceleration'], cfg['safety_linear_deceleration'])
            reaction = cfg['safety_reaction_time'] + cfg['period']
            cap = math.sqrt((decel*reaction)**2 + 2*decel*max(0., distance-.5*tolerance)) - decel*reaction
            for scale in (1., .875, .75, .625, .5, .375, .25, .125, 0.):
                command = limit_tracking_command(
                    min(cap, tracking.target_speed*scale), tracking.target_speed, tracking.angular_velocity,
                    self.last_linear, self.last_angular, min(elapsed, cfg['period']),
                    cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'],
                    cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])
                if self._motion_safe(grid, pose, command.linear_velocity, command.angular_velocity):
                    return self._command(command, now, 'tracking path')
            return self._zero('requested stopping region intersects an obstacle')
