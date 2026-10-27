"""TB-owned tunnel mission: heading-controlled entry, Hybrid A*, odom tracking.

No ROS publishers, AMCL or other mission controllers.
LiDAR inputs share one scan representation with an acquisition-time odom
pose. One lock serializes mapping and the final collision/command decision;
Stopped wall registration and Hybrid A* searches run asynchronously on
immutable input snapshots.
"""

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from .costmap import TunnelCostmap
from .static_geometry import refined_static_cells
from .localization import AlignmentResult, EntranceLocalizer
from .planner import HybridAStarPlanner, OccupancyGrid, Pose2D, RectangularFootprint
from .tracking import (LimitedCommand, TrackingPath, build_speed_profile, calculate_tracking,
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
        if static_map.get('geometry'):
            static_map['geometry'] = str((Path(path).resolve().parent / static_map['geometry']).resolve())
    localization = config.get('localization')
    if localization is not None:
        geometry = (Path(path).resolve().parent / localization['geometry']).resolve()
        localization['geometry'] = str(geometry)
        with geometry.open(encoding='utf-8') as stream:
            localization['wall_rectangles'] = json.load(stream)['wall_rectangles']
    return config


def relative_pose(pose, anchor):
    dx, dy = pose[0] - anchor[0], pose[1] - anchor[1]
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    return Pose2D(c * dx + s * dy, -s * dx + c * dy,
                  normalize_angle(pose[2] - anchor[2]))


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
        entry_acceleration = config['entry'].get('linear_acceleration', self.control['linear_acceleration'])
        if (isinstance(entry_acceleration, (bool, np.bool_))
                or not isinstance(entry_acceleration, (int, float, np.integer, np.floating))
                or not math.isfinite(entry_acceleration) or entry_acceleration <= 0):
            raise ValueError('entry/linear_acceleration must be finite and positive')
        self.entry_linear_acceleration = float(entry_acceleration)
        if (int(self.sensors['angular_bins']) != self.sensors['angular_bins']
                or self.sensors['angular_bins'] < 8
                or not 0 <= self.sensors['range_min'] < self.sensors['range_max']):
            raise ValueError('invalid LiDAR angular bins or range limits')
        self.footprint = RectangularFootprint(**config['footprint'])
        entry_half_width = config['entry'].get(
            'obstacle_half_width', self.footprint.half_width + self.footprint.padding)
        if (isinstance(entry_half_width, (bool, np.bool_))
                or not isinstance(entry_half_width, (int, float, np.integer, np.floating))):
            raise ValueError('entry/obstacle_half_width must be a finite number within the footprint margin')
        try:
            entry_half_width = float(entry_half_width)
        except (ValueError, OverflowError) as error:
            raise ValueError('entry/obstacle_half_width is out of range') from error
        if (not math.isfinite(entry_half_width)
                or not self.footprint.half_width <= entry_half_width
                <= self.footprint.half_width + self.footprint.padding):
            raise ValueError('entry/obstacle_half_width must cover the body and stay within its padding')
        self.entry_obstacle_half_width = entry_half_width
        entry_right_width = config['entry'].get('obstacle_right_width', entry_half_width)
        if (isinstance(entry_right_width, (bool, np.bool_))
                or not isinstance(entry_right_width, (int, float, np.integer, np.floating))):
            raise ValueError('entry/obstacle_right_width must be a finite positive number')
        try:
            entry_right_width = float(entry_right_width)
        except (ValueError, OverflowError) as error:
            raise ValueError('entry/obstacle_right_width is out of range') from error
        if not math.isfinite(entry_right_width) or entry_right_width <= 0.:
            raise ValueError('entry/obstacle_right_width must be a finite positive number')
        # Explicit right-side override may be inside the physical body edge;
        # it affects only the entry trigger, never the mapped scan/footprint.
        self.entry_obstacle_right_width = entry_right_width
        for key, default in (('obstacle_left_far_width', entry_half_width),
                             ('obstacle_check_distance', self.sensors['range_max'])):
            value = config['entry'].get(key, default)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(value) or value <= 0.):
                raise ValueError('entry/%s must be finite and positive' % key)
            setattr(self, 'entry_' + key, float(value))
        if self.entry_obstacle_left_far_width < entry_half_width:
            raise ValueError('entry/obstacle_left_far_width must cover the near left width')
        if not config['entry']['obstacle_clearance'] < self.entry_obstacle_check_distance <= self.sensors['range_max']:
            raise ValueError('entry/obstacle_check_distance must exceed the stop gap and stay within sensor range')
        self.planner = HybridAStarPlanner(self.footprint, **config['planner'])
        entry_heading = config['entry'].get('heading_offset_deg', 0.)
        if (isinstance(entry_heading, (bool, np.bool_))
                or not isinstance(entry_heading, (int, float, np.integer, np.floating))
                or not math.isfinite(entry_heading) or abs(entry_heading) > 15.):
            raise ValueError('entry/heading_offset_deg must be finite and within +/-15 degrees')
        self.entry_heading_offset = math.radians(entry_heading)
        entry_curvature = config['entry'].get(
            'maximum_curvature', min(.35, 1./self.planner.minimum_turning_radius))
        if (isinstance(entry_curvature, (bool, np.bool_))
                or not isinstance(entry_curvature, (int, float, np.integer, np.floating))
                or not math.isfinite(entry_curvature)
                or not 0. < entry_curvature <= 1./self.planner.minimum_turning_radius):
            raise ValueError('entry/maximum_curvature must be finite, positive and within the turning limit')
        self.entry_maximum_curvature = float(entry_curvature)
        for key, default, maximum in (
                ('alignment_angular_velocity', .10, self.control['maximum_angular_velocity']),
                ('alignment_angular_acceleration', .25, self.control['angular_acceleration']),
                ('alignment_tolerance_deg', .3, 1.)):
            value = config['entry'].get(key, default)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(value) or not 0. < value <= maximum):
                raise ValueError('entry/%s must be finite, positive and <= %s' % (key, maximum))
            setattr(self, 'entry_' + key, float(value))
        goal = config['goal']['pose']
        self.goal = Pose2D(float(goal[0]), float(goal[1]), math.radians(goal[2]))
        planning_goal = config['goal'].get('planning_pose', goal)
        self.planning_goal = Pose2D(float(planning_goal[0]), float(planning_goal[1]),
                                   math.radians(planning_goal[2]))
        self.lock = threading.RLock()
        self.odom = self.scan = None
        self.localization_scan = None
        self.localizer = None
        if 'localization' in config:
            settings = config['localization']
            confirmations = settings['confirmation_scans']
            if isinstance(confirmations, bool) or not isinstance(confirmations, int) or confirmations < 2:
                raise ValueError('localization/confirmation_scans must be an integer >= 2')
            self.localizer = EntranceLocalizer(
                settings['wall_rectangles'], search_bounds=settings['search_bounds'],
                yaw_limit_deg=settings['yaw_limit_deg'])
            window = settings.get('observation_window', 0.)
            if isinstance(window, bool) or not isinstance(window, (int, float)) or not math.isfinite(window) or not 0 <= window <= 2.:
                raise ValueError('localization/observation_window must be between 0 and 2 seconds')
        self.front_clearance = math.inf
        self.lane_stamp = None
        self.lane_count = 0
        self.generation = 0
        self.worker = None
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.state = 'LOCALIZING' if self.localizer is not None else 'ENTRY'
            self.reason = 'waiting for sensors'
            self.anchor = self.path = self.costmap = self.grid = None
            self.static_cell_rectangles = {}
            self.plan_result = None
            self.last_plan_attempt = None
            self.planning_started = self.planning_duration = None
            self.freshness_report = None
            self.mapped_stamp = self.last_time = None
            self.version = self.path_version = self.index = self.plans = 0
            self.last_linear = self.last_angular = 0.0
            self.lane_count = 0
            self.completion_started = None
            self.localization_result = self.localization_candidate = None
            self.localization_epoch = getattr(self, 'localization_epoch', 0) + 1
            self.localization_anchors = []
            self.localization_history = []
            self.localization_attempt_stamp = None
            self.localization_confirmations = 0
            self.localization_report = ''
            self.entry_start_x = 0.
            self.entry_start_yaw = 0.
            self.entry_phase = 'WAIT_STOP' if self.entry_heading_offset else 'DRIVE'
            self.entry_drive_yaw = None
            self.initial_map_updates = 0

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

    def update_cloud(self, points, sensor_pose, odom_pose, stamp, localization_points=None):
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
        wall_points = points[valid]
        if localization_points is not None:
            wall_points = np.asarray(localization_points, dtype=float).reshape((-1, 2))
            wall_range = np.linalg.norm(wall_points - np.asarray(sensor_pose[:2]), axis=1)
            wall_points = wall_points[np.isfinite(wall_points).all(axis=1)
                                      & (wall_range >= self.sensors['range_min'])
                                      & (wall_range <= self.sensors['range_max'])]
        return self.update_scan(scan, -math.pi + .5 * angle_step, angle_step,
                                self.sensors['range_min'], self.sensors['range_max'],
                                sensor_pose, odom_pose, stamp, _points=wall_points)

    def update_scan(self, ranges, angle_min, angle_increment, range_min, range_max,
                    sensor_pose, odom_pose, stamp, _points=None):
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
        # Entry only: widen the left edge toward the far end of a trapezoid.
        # Keep the complete scan for mapping and path collision checks.
        ahead = (x >= 0) & (y >= -self.entry_obstacle_right_width)
        left_limit = self.entry_obstacle_half_width
        if self.entry_obstacle_left_far_width > left_limit:
            far_x = self.footprint.front + self.entry_obstacle_check_distance
            ahead &= x <= far_x
            left_limit = left_limit + (self.entry_obstacle_left_far_width - left_limit) * x / far_x
        ahead &= y <= left_limit
        clearance = float(np.min(x[ahead])) - self.footprint.front if np.any(ahead) else math.inf
        with self.lock:
            if self.scan is not None and stamp <= self.scan[-1]:
                return False
            self.scan = (ranges, float(angle_min), float(angle_increment), float(range_min),
                         float(effective_max), tuple(sensor_pose), tuple(odom_pose), float(stamp))
            self.front_clearance = clearance
            if self.localizer is not None and self.anchor is None:
                # Preserve accepted raw XY for alignment: scan bin centres
                # quantize the bearing and can shift distant walls by mm/cm.
                points = np.column_stack((x, y))[ranges[hits] <= effective_max] if _points is None else _points.copy()
                self.localization_scan = (points, tuple(odom_pose), float(stamp))
                window = self.config['localization'].get('observation_window', 0.)
                self.localization_history = [row for row in self.localization_history if stamp-row[2] <= window+.001]
                self.localization_history.append(self.localization_scan)
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
        self.initial_map_updates += 1
        if (update.marked_cells or update.cleared_cells or update.decayed_cells
                or update.refined_cells or update.newly_observed_cells):
            self.version += 1
            self.grid = None

    def _initialize(self, anchor=None):
        self.anchor = self.odom[:3] if anchor is None else anchor
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
        self.static_cell_rectangles = refined_static_cells(
            static_map, resolution, static_data, self.costmap.static_occupied_threshold,
            crop=(self.costmap.source_column, self.costmap.source_row,
                  self.costmap.width, self.costmap.height))
        if self.scan is not None:
            self._map_scan()

    def _invalidate_localization(self, clear_observations=True):
        self.localization_epoch += 1
        self.localization_candidate = self.localization_result = None
        self.localization_confirmations = 0
        self.localization_anchors = []
        if clear_observations:
            self.localization_history = []

    def _localize(self, now):
        """Fit the stopped robot to the surveyed map, never move the map with it."""
        if (abs(self.odom[3]) > self.control['planning_stopped_linear']
                or abs(self.odom[4]) > self.control['planning_stopped_angular']):
            self._invalidate_localization()
            self.reason = 'waiting for measured rest before wall alignment'
            return False
        if self.localization_result is not None:
            result, scan_odom, stamp = self.localization_result
            self.localization_result = None
            moved = (math.hypot(self.odom[0]-scan_odom[0], self.odom[1]-scan_odom[1]) > .005
                     or abs(normalize_angle(self.odom[2]-scan_odom[2])) > math.radians(.5))
            expired = now-stamp > self.sensors['maximum_age']
            if expired or moved or not result.accepted:
                # Discard an expired fit and all confirmations, but a slow
                # worker does not invalidate fresh stopped scans collected
                # meanwhile. Recheck their poses before the next fit below.
                self._invalidate_localization(
                    clear_observations=not (expired and not moved and result.accepted))
                self.reason = ('wall alignment input expired or robot moved' if moved or
                               now-stamp > self.sensors['maximum_age'] else result.reason)
                return False
            else:
                # odom_T_map = odom_T_base(at scan time) * inverse(map_T_base).
                fitted = result.pose
                yaw = normalize_angle(scan_odom[2]-fitted.yaw)
                c, s = math.cos(yaw), math.sin(yaw)
                anchor = (scan_odom[0]-c*fitted.x+s*fitted.y,
                          scan_odom[1]-s*fitted.x-c*fitted.y, yaw)
                previous = self.localization_candidate
                previous_pose = relative_pose(scan_odom, previous) if previous is not None else None
                consistent = (previous is not None
                    and math.hypot(fitted.x-previous_pose.x, fitted.y-previous_pose.y) <= .015
                    and abs(normalize_angle(fitted.yaw-previous_pose.yaw)) <= math.radians(.5))
                self.localization_confirmations = self.localization_confirmations+1 if consistent else 1
                if not consistent:
                    self.localization_candidate = anchor
                    self.localization_anchors = []
                self.localization_anchors.append(anchor)
                required = self.config['localization']['confirmation_scans']
                self.reason = 'wall alignment confirmation %d/%d' % (self.localization_confirmations, required)
                if self.localization_confirmations >= required:
                    # Average observations at a common nearby base pose. Raw
                    # odom_T_map translations exaggerate yaw noise when odom's
                    # origin is many metres away after the preceding course.
                    estimates = [relative_pose(scan_odom, a) for a in self.localization_anchors]
                    mean_x = float(np.mean([p.x for p in estimates]))
                    mean_y = float(np.mean([p.y for p in estimates]))
                    mean_yaw = math.atan2(np.mean([math.sin(p.yaw) for p in estimates]),
                                          np.mean([math.cos(p.yaw) for p in estimates]))
                    yaw = normalize_angle(scan_odom[2]-mean_yaw)
                    c, s = math.cos(yaw), math.sin(yaw)
                    anchor = (scan_odom[0]-c*mean_x+s*mean_y,
                              scan_odom[1]-s*mean_x-c*mean_y, yaw)
                    self._initialize(anchor)
                    pose = relative_pose(self.odom[:3], anchor)
                    self.entry_start_x = pose.x
                    self.entry_start_yaw = pose.yaw
                    self.localization_report = 'wall alignment x=%.3fm y=%.3fm yaw=%.2fdeg rms=%.4fm' % (
                        pose.x, pose.y, math.degrees(pose.yaw), result.rms)
                    self.reason = self.localization_report
                    # Freeze the measured entry direction. The optional bias
                    # is relative to it and is applied only while advancing.
                    self.state = 'ENTRY'
                    return True
        if self.worker is not None and self.worker.is_alive():
            return False
        if self.localization_scan is None:
            self.reason = 'waiting for wall points for alignment'
            return False
        points, scan_odom, stamp = self.localization_scan
        if stamp == self.localization_attempt_stamp:
            return False
        window = self.config['localization'].get('observation_window', 0.)
        if window:
            if (not self.localization_history
                    or stamp-self.localization_history[0][2] < window-.11):
                self.reason = 'collecting stopped wall observations'
                return False
            combined = []
            for hits, acquired_pose, _ in self.localization_history:
                relative = relative_pose(acquired_pose, scan_odom)
                if math.hypot(relative.x, relative.y) > .005 or abs(relative.yaw) > math.radians(.5):
                    self._invalidate_localization()
                    self.reason = 'robot moved during wall observation'
                    return False
                c, s = math.cos(relative.yaw), math.sin(relative.yaw)
                combined.append(hits @ np.array(((c, s), (-s, c))) + (relative.x, relative.y))
            points = np.concatenate(combined)
        self.localization_attempt_stamp = stamp
        generation = self.generation
        epoch = self.localization_epoch
        def align():
            try:
                result = self.localizer.estimate(points)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                result = AlignmentResult(False, reason='wall alignment failed: '+str(error))
            with self.lock:
                if self.generation == generation and self.localization_epoch == epoch:
                    self.localization_result = (result, scan_odom, stamp)
        self.worker = threading.Thread(target=align, daemon=True)
        self.worker.start()
        return False

    def _grid(self):
        if self.grid is None:
            c = self.costmap
            # Static walls and measured obstacles share one collision grid;
            # unknown-only cells do not reject the robot's sensor blind area.
            values = np.asarray(c.to_occupancy_data()).reshape(c.height, c.width)
            self.grid = OccupancyGrid(values, c.resolution, c.origin_x, c.origin_y,
                                      occupied_threshold=c.static_occupied_threshold,
                                      soft_cost_data=np.asarray(c.to_soft_cost_data()).reshape(c.height, c.width),
                                      dynamic_cell_rectangles=c.refined_dynamic_cells(),
                                      static_cell_rectangles=self.static_cell_rectangles)
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
            path = self._tracking_path(plan.x, plan.y, plan.yaw, plan.curvature)
            if not self._route_safe(grid, 0, path):
                raise ValueError('planned path is not collision-free')
            return path
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

    def _align_entry(self, pose, elapsed, now):
        """Turn once; return None to begin straight entry in this same tick."""
        cfg = self.control
        measured_linear, measured_angular = self.odom[3:5]
        stopped = (abs(measured_linear) <= cfg['planning_stopped_linear']
                   and abs(measured_angular) <= cfg['planning_stopped_angular'])
        error = normalize_angle(self.entry_start_yaw + self.entry_heading_offset - pose.yaw)
        direction = math.copysign(1., self.entry_heading_offset)
        remaining = direction * error
        tolerance = math.radians(self.entry_alignment_tolerance_deg)
        if self.entry_phase == 'WAIT_STOP':
            if not stopped:
                return self._zero('waiting for measured rest before entry heading turn')
            self.entry_phase = 'ALIGN'
        if remaining <= tolerance:
            # Latch the reached heading once and start forward acceleration
            # without waiting for angular rest. Moving entry corrects drift.
            self.entry_drive_yaw = pose.yaw
            self.entry_phase = 'DRIVE'
            return None
        if abs(measured_linear) > cfg['planning_stopped_linear']:
            return self._zero('waiting for measured rest before entry heading turn')

        acceleration = self.entry_alignment_angular_acceleration
        deceleration = min(acceleration, cfg['safety_angular_deceleration'])
        reaction = cfg['safety_reaction_time'] + cfg['period']
        # Slow near the target; reaching or passing it ends the turn above.
        cap = math.sqrt((deceleration * reaction)**2 + 2. * deceleration * remaining) - deceleration * reaction
        target = direction * min(self.entry_alignment_angular_velocity, cap)
        change = acceleration * min(elapsed, cfg['period'])
        angular = max(self.last_angular-change, min(self.last_angular+change, target))
        return self._command(LimitedCommand(0., angular), now,
                             'aligning entry heading before straight drive')

    def _enter(self, pose, elapsed, now):
        if self.entry_heading_offset and not self.planner.pose_is_collision_free(self._grid(), pose):
            return self._zero('current body overlaps mapped obstacle; physical contact not measured')
        if self.entry_phase != 'DRIVE':
            alignment_command = self._align_entry(pose, elapsed, now)
            if alignment_command is not None:
                return alignment_command
        cfg, entry = self.control, self.config['entry']
        entry_name = 'aligned straight entry' if self.entry_heading_offset else 'straight entry'
        traveled = max(0., (pose.x-self.entry_start_x)/math.cos(self.entry_start_yaw))
        distance_remaining = entry['distance'] - pose.x
        clearance_remaining = self.front_clearance - entry['obstacle_clearance']
        remaining = min(distance_remaining, clearance_remaining)
        if remaining <= cfg['safety_distance_margin']:
            command = self._stop_command(elapsed)
            if command.linear_velocity > 1e-9 or abs(command.angular_velocity) > 1e-9:
                return self._command(command, now, 'stopping at ' + entry_name + ' target')
            self.state = 'PLANNING'
            cause = 'distance target' if distance_remaining <= clearance_remaining else 'front clearance'
            return self._zero('%s complete: %s; traveled=%.3fm, front_clearance=%.3fm, required_gap=%.3fm'
                              % (entry_name, cause, traveled, self.front_clearance, entry['obstacle_clearance']))
        # Entry velocity is the initial profile speed, not a whole-leg cap.
        # Match the source's distance ramp and reaction + braking limit while
        # retaining the entry stopping target.
        reference = min(entry['cruise_velocity'], math.sqrt(
            cfg['entry_velocity']**2 + 2 * self.entry_linear_acceleration * traveled))
        decel = min(cfg['linear_deceleration'], cfg['safety_linear_deceleration'])
        reaction = cfg['safety_reaction_time'] + cfg['period']
        remaining = max(0., remaining - .5 * cfg['safety_distance_margin'])
        cap = math.sqrt((decel * reaction)**2 + 2 * decel * remaining) - decel * reaction
        target_yaw = (self.entry_start_yaw if self.entry_drive_yaw is None
                      else self.entry_drive_yaw)
        angular = cfg['heading_gain'] * normalize_angle(target_yaw-pose.yaw)
        if self.entry_heading_offset:
            # After the separate pivot, only correct drift from the frozen
            # straight heading. Keep these moving corrections gentle.
            maximum_angular = reference * self.entry_maximum_curvature
            angular = max(-maximum_angular, min(maximum_angular, angular))
        command = limit_tracking_command(
            min(reference, cap), reference, angular,
            self.last_linear, self.last_angular, min(elapsed, cfg['period']),
            self.entry_linear_acceleration, cfg['linear_deceleration'], cfg['angular_acceleration'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])
        return self._command(command, now, entry_name)

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

    def _planning_status(self):
        """Report worker progress without consuming a result or bypassing gates."""
        if self.plan_result is not None:
            result, error, _ = self.plan_result
            duration = '' if self.planning_duration is None else ' in %.2fs' % self.planning_duration
            return ('plan ready' + duration if result is not None
                    else 'plan finished' + duration + ': ' + error)
        if self.worker is not None and self.worker.is_alive():
            if self.planning_started is None:
                return 'waiting for previous planning worker'
            return 'planning from measured rest (elapsed %ds)' % max(0., time.monotonic()-self.planning_started)
        return ''

    def _planning_gate_reason(self, reason):
        status = self._planning_status() if self.state == 'PLANNING' else ''
        return reason + ('; ' + status if status else '')

    def _stale_sensor_reason(self, now):
        # Numeric diagnostics refresh once per second. A changed failing
        # sensor or completed worker is reported immediately; gates still
        # evaluate the exact timestamps on every control tick.
        ages = (now-self.odom[-1], now-self.scan[-1])
        failures = tuple('future' if age < -self.sensors['maximum_future_stamp']
                         else 'stale' if age > self.sensors['maximum_age'] else 'fresh'
                         for age in ages)
        status = self._planning_status() if self.state == 'PLANNING' else ''
        key = (int(time.monotonic()), failures, status)
        if self.freshness_report is None or self.freshness_report[0] != key:
            reason = 'waiting for fresh odometry and lidar (odom_age=%.3fs, lidar_age=%.3fs)' % ages
            self.freshness_report = (key, reason + ('; ' + status if status else ''))
        return self.freshness_report[1]

    def _begin_plan(self, pose, now):
        if self.worker is not None and self.worker.is_alive():
            self.reason = self._planning_status()
            return
        if (self.last_plan_attempt is not None
                and now - self.last_plan_attempt < self.control['plan_retry_period']):
            return
        # step still holds the mission lock: build the current immutable
        # snapshot only when a search will consume it. A pending search or
        # retry delay needs no new grid, although incoming scans keep mapping.
        grid = self._grid()
        self.last_plan_attempt = now
        generation = self.generation
        self.plan_result = None
        started = self.planning_started = time.monotonic()
        self.planning_duration = None
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
            duration = max(0., time.monotonic()-started)
            with self.lock:
                if self.generation == generation:
                    self.planning_duration = duration
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
                if self.localizer is not None and self.anchor is None:
                    self._invalidate_localization()
                return self._zero(self._stale_sensor_reason(now))
            self.freshness_report = None
            if self.anchor is None:
                if self.localizer is not None:
                    self._localize(now)
                    return self._zero(self.reason)
                self._initialize()
            pose = relative_pose(self.odom[:3], self.anchor)
            measured_linear, measured_angular = self.odom[3:5]
            if self.localizer is not None:
                if self.initial_map_updates < self.config['grid']['mark_observations']:
                    return self._zero('waiting for fresh mapped scans after wall alignment')
                if not self.planner.pose_is_collision_free(self._grid(), pose):
                    return self._zero('localized footprint or clearance is blocked in map; physical contact not measured')
            if self.state == 'WAIT_STOP':
                if (self.odom[-1] > self.completion_started
                        and abs(measured_linear) <= self.control['planning_stopped_linear']
                        and abs(measured_angular) <= self.control['planning_stopped_angular']):
                    self.state = 'COMPLETE'
                    return self._zero('goal reached; measured stop confirmed')
                return self._zero('waiting for measured stop at outside goal')
            if self.state == 'ENTRY':
                return self._enter(pose, elapsed, now)
            tracking_reason = 'tracking path'
            if self.state == 'PLANNING':
                if (abs(measured_linear) > self.control['planning_stopped_linear']
                        or abs(measured_angular) > self.control['planning_stopped_angular']):
                    return self._zero(self._planning_gate_reason('waiting for measured rest before planning'))
                if self.plan_result is None:
                    self._begin_plan(pose, now)
                    return self._zero(self.reason)
                result, error, start = self.plan_result
                completed_status = self._planning_status()
                self.plan_result = None
                if result is None:
                    # The source retries from rest against the latest grid;
                    # one unsuccessful search is not a terminal mission failure.
                    return self._zero(completed_status)
                if (math.hypot(pose.x-start.x, pose.y-start.y)
                        > self.control['planning_start_position_tolerance']
                        or abs(normalize_angle(pose.yaw-start.yaw))
                        > math.radians(self.control['planning_start_heading_tolerance_deg'])):
                    self.last_plan_attempt = None
                    return self._zero('discarding plan after stopped pose shifted')
                self.path = result
                self.index, self.path_version = 0, -1
                self.state = 'FOLLOWING'
                if self.planning_duration is not None:
                    tracking_reason += '; plan finished in %.2fs' % self.planning_duration
            # A completed plan is checked against the latest scan snapshot,
            # including all updates received while its search was running.
            grid = self._grid()
            # Match the source's decision order. A transient endpoint under
            # the robot holds the same path until later free rays clear it;
            # it must not first erase the path and force a colliding-start plan.
            if not self.planner.pose_is_collision_free(grid, pose):
                return self._zero('localized footprint or clearance is blocked in map; physical contact not measured')
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
                if command.linear_velocity > 1e-9 or abs(command.angular_velocity) > 1e-9:
                    return self._command(command, now, 'stopping at outside goal')
                self.state = 'WAIT_STOP'
                # Require a new odometry sample after requesting zero, even
                # when the current sample has an allowed future timestamp.
                self.completion_started = max(now, self.odom[-1])
                return self._zero('waiting for measured stop at outside goal')
            if (tracking.position_error > cfg['tracking_position_tolerance']
                    or abs(tracking.heading_error) > math.radians(cfg['tracking_heading_tolerance_deg'])):
                return self._zero('tracking tolerance exceeded: position=%.3fm, heading=%.1fdeg'
                                  % (tracking.position_error, math.degrees(tracking.heading_error)))
            if self.path_version != self.version:
                if not self._route_safe(grid, self.index):
                    # Only an actual remaining-route conflict requests search.
                    self.state, self.path, self.plan_result = 'PLANNING', None, None
                    self.last_plan_attempt = None
                    return self._zero('remaining route intersects an obstacle')
                self.path_version = self.version
            decel = min(cfg['linear_deceleration'], cfg['safety_linear_deceleration'])
            reaction = cfg['safety_reaction_time'] + cfg['period']
            cap = math.sqrt((decel*reaction)**2 + 2*decel*max(0., distance-.5*tolerance)) - decel*reaction
            # Obstacle decisions use the current padded body and remaining
            # path geometry above. Apply the existing speed/acceleration limits
            # once; no velocity-dependent projected stopping sweep is added.
            command = limit_tracking_command(
                min(cap, tracking.target_speed), tracking.target_speed, tracking.angular_velocity,
                self.last_linear, self.last_angular, min(elapsed, cfg['period']),
                cfg['linear_acceleration'], cfg['linear_deceleration'], cfg['angular_acceleration'],
                cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])
            return self._command(command, now, tracking_reason)
