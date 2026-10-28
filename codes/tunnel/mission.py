"""Tunnel sensor input, localization, entry, route planning and command control."""

from dataclasses import dataclass
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
from .geometry import CollisionGeometry, OccupancyGrid, Pose2D, RectangularFootprint
from .tracking import LimitedCommand, limit_tracking_command, normalize_angle, turn_command
from .planner import GlobalLocalPlanner, GlobalRoute, load_navigation_settings


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

@dataclass(frozen=True)
class ControlFrame:
    now: float
    elapsed: float
    pose: Pose2D


@dataclass(frozen=True)
class RouteUpdate:
    route: GlobalRoute
    error: str
    duration: float


class TunnelMission:
    def __init__(self, config, clock=None):
        self.navigation = load_navigation_settings(config.get('navigation'))
        period = self.navigation['replan_period']
        if (isinstance(period, bool) or not isinstance(period, (int, float))
                or not math.isfinite(period) or period <= 0):
            raise ValueError('navigation/replan_period must be finite and positive')
        self.config = config
        self.clock = clock
        self.control, self.sensors = config['control'], config['sensors']
        lidar_age = self.sensors.get('lidar_maximum_age', self.sensors['maximum_age'])
        if lidar_age is not None and (
                isinstance(lidar_age, (bool, np.bool_))
                or not isinstance(lidar_age, (int, float, np.integer, np.floating))
                or not math.isfinite(lidar_age) or lidar_age <= 0.):
            raise ValueError('sensors/lidar_maximum_age must be null or finite and positive')
        # Null disables the LiDAR age limit; other timestamp checks remain active.
        self.lidar_maximum_age = math.inf if lidar_age is None else float(lidar_age)
        for key in ('period', 'linear_acceleration', 'linear_deceleration',
                    'angular_acceleration', 'safety_linear_deceleration',
                    'safety_angular_deceleration', 'safety_reaction_time'):
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
        self.geometry = self._create_geometry(config['planner'])
        entry_heading = config['entry'].get('heading_offset_deg', 0.)
        if (isinstance(entry_heading, (bool, np.bool_))
                or not isinstance(entry_heading, (int, float, np.integer, np.floating))
                or not math.isfinite(entry_heading) or abs(entry_heading) > 15.):
            raise ValueError('entry/heading_offset_deg must be finite and within +/-15 degrees')
        self.entry_heading_offset = math.radians(entry_heading)
        entry_curvature = config['entry'].get(
            'maximum_curvature', min(.35, 1./self.geometry.minimum_turning_radius))
        if (isinstance(entry_curvature, (bool, np.bool_))
                or not isinstance(entry_curvature, (int, float, np.integer, np.floating))
                or not math.isfinite(entry_curvature)
                or not 0. < entry_curvature <= 1./self.geometry.minimum_turning_radius):
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
        settings = dict(self.navigation)
        settings['local_rotation_tolerance'] = config['planner'].get('goal_heading_tolerance', math.radians(7.5))
        self.local_planner = GlobalLocalPlanner(self.geometry, settings)
        self.completion = config['goal'].get('completion', 'stop')
        if self.completion not in ('stop', 'lane'):
            raise ValueError('goal/completion must be stop or lane')
        self.exit_crossing = self._exit_crossing_geometry(config['goal'].get('exit_crossing'))

    def reset(self):
        with self.lock:
            self.generation += 1
            self.state = 'LOCALIZING' if self.localizer is not None else 'ENTRY'
            self.reason = 'waiting for sensors'
            self.anchor = self.costmap = self.grid = None
            self.static_cell_rectangles = {}
            self.last_plan_attempt = None
            self.freshness_report = None
            self.mapped_stamp = self.last_time = None
            self.plans = 0
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

            self.global_path = None
            self.pending_route = None
            self.global_updates = 0
            self.global_report = 'waiting for first global route'
            self.exit_previous_position = None
            self.exit_crossed = False

    def close(self):
        self.reset()

    def step(self, now):
        with self.lock:
            frame, response = self._prepare_step(now)
            if response is not None:
                return response
            now, elapsed, pose = frame.now, frame.elapsed, frame.pose
            if self.state == 'ENTRY':
                return self._enter(pose, elapsed, now)
            grid = self._grid()
            if not self.geometry.pose_is_collision_free(grid, pose):
                return self._zero('current footprint overlaps mapped obstacle')
            if self.state in ('WAIT_STOP', 'WAIT_LANE'):
                return self._wait_for_completion(pose, elapsed, now, grid)
            if self.state == 'FINISHING' or self._goal_reached(pose):
                return self._finish_goal(pose, elapsed, now, grid)
            self._accept_global_plan()
            self._request_global_plan(pose, now)
            if self.global_path is None:
                return self._zero('waiting for first global route; ' + self.global_report)
            command = self.local_planner.local_command(
                grid, pose, self.global_path, self.control,
                self.last_linear, self.last_angular, elapsed)
            if command is None:
                return self._zero('no safe local motion; ' + self.global_report)
            return self._command(command, now, command.reason + '; ' + self.global_report)

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
                or update.refined_cells):
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
            # The map uses black occupied cells and 254-valued free cells.
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

    def _command(self, command, now, reason):
        published_at = self.clock() if self.clock is not None else now
        if (published_at - self.odom[-1] > self.control['safety_reaction_time']
                or published_at - self.scan[-1] > self.lidar_maximum_age):
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
        if self.entry_heading_offset and not self.geometry.pose_is_collision_free(self._grid(), pose):
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
        # Accelerate from entry_velocity; decelerate toward the entry stop gap.
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

    def _sensor_times_valid(self, now):
        ages = (now-self.odom[-1], now-self.scan[-1])
        limits = (self.sensors['maximum_age'], self.lidar_maximum_age)
        return all(-self.sensors['maximum_future_stamp'] <= age <= limit
                   for age, limit in zip(ages, limits))

    def _stale_sensor_reason(self, now):
        # Numeric diagnostics refresh once per second. A changed failing
        # sensor or completed worker is reported immediately; gates still
        # evaluate the exact timestamps on every control tick.
        ages = (now-self.odom[-1], now-self.scan[-1])
        limits = (self.sensors['maximum_age'], self.lidar_maximum_age)
        failures = tuple('future' if age < -self.sensors['maximum_future_stamp']
                         else 'stale' if age > limit else 'fresh'
                         for age, limit in zip(ages, limits))
        key = (int(time.monotonic()), failures)
        if self.freshness_report is None or self.freshness_report[0] != key:
            reason = 'waiting for fresh odometry and lidar (odom_age=%.3fs, lidar_age=%.3fs)' % ages
            self.freshness_report = (key, reason)
        return self.freshness_report[1]

    def _create_geometry(self, settings):
        keys = ('minimum_turning_radius', 'inflation_radius',
                'collision_check_step', 'collision_check_angle', 'keep_in_rectangles')
        return CollisionGeometry(self.footprint, **{key: settings[key] for key in keys if key in settings})

    def _prepare_step(self, now):
        """Return a fresh control frame, or an already-decided stopped response."""
        if self.clock is not None:
            now = self.clock()
        elapsed = self.control['period'] if self.last_time is None else max(0., now-self.last_time)
        self.last_time = now
        if self.state == 'COMPLETE':
            return None, self._zero(self.reason)
        if self.odom is None or self.scan is None:
            return None, self._zero('waiting for odometry and lidar')
        if not self._sensor_times_valid(now):
            if self.localizer is not None and self.anchor is None:
                self._invalidate_localization()
            return None, self._zero(self._stale_sensor_reason(now))
        self.freshness_report = None
        if self.anchor is None:
            if self.localizer is not None:
                self._localize(now)
                return None, self._zero(self.reason)
            self._initialize()
        pose = relative_pose(self.odom[:3], self.anchor)
        if self.localizer is not None:
            if self.initial_map_updates < self.config['grid']['mark_observations']:
                return None, self._zero('waiting for fresh mapped scans after wall alignment')
            if not self.geometry.pose_is_collision_free(self._grid(), pose):
                return None, self._zero('localized footprint or clearance is blocked in map; physical contact not measured')
        return ControlFrame(now, elapsed, pose), None

    def _measured_stop_confirmed(self):
        return (self.odom[-1] > self.completion_started
                and abs(self.odom[3]) <= self.control['planning_stopped_linear']
                and abs(self.odom[4]) <= self.control['planning_stopped_angular'])

    def _exit_crossing_geometry(self, segment):
        """Surveyed opening; the outside guide selects its outward side."""
        if segment is None:
            return None
        if (not isinstance(segment, (list, tuple)) or len(segment) != 2
                or any(not isinstance(point, (list, tuple)) or len(point) != 2
                       for point in segment)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for point in segment for value in point)):
            raise ValueError('goal/exit_crossing must contain two finite XY endpoints')
        (x, y), (end_x, end_y) = segment
        width = math.hypot(end_x-x, end_y-y)
        if width <= 1e-9:
            raise ValueError('goal/exit_crossing must have positive width')
        tx, ty = (end_x-x)/width, (end_y-y)/width
        nx, ny = -ty, tx
        dx, dy = self.goal.x-x, self.goal.y-y
        side = nx*dx+ny*dy
        if abs(side) <= 1e-9 or not 0. <= tx*dx+ty*dy <= width:
            raise ValueError('goal must be off the exit plane and laterally within exit_crossing')
        if side < 0.:
            nx, ny = -nx, -ny
        return x, y, tx, ty, nx, ny, width

    def _goal_reached(self, pose):
        if self.exit_crossing is None:
            return (math.hypot(self.goal.x-pose.x, self.goal.y-pose.y)
                    <= self.config['goal']['position_tolerance'])
        x, y, tx, ty, nx, ny, width = self.exit_crossing
        forward = nx*(pose.x-x)+ny*(pose.y-y)
        lateral = tx*(pose.x-x)+ty*(pose.y-y)
        previous = self.exit_previous_position
        self.exit_previous_position = forward, lateral
        if previous is not None:
            old_forward, old_lateral = previous
            if old_forward <= 0. <= forward and forward-old_forward > 1e-9:
                fraction = -old_forward/(forward-old_forward)
                crossing = old_lateral+fraction*(lateral-old_lateral)
                self.exit_crossed = 0. <= crossing <= width
            elif old_forward >= 0. > forward:
                # Returning inside requires a new outward passage through
                # this opening; an earlier passage cannot qualify another exit.
                self.exit_crossed = False
        if not self.exit_crossed:
            return False
        cx, cy, c, s, half_length, half_width = self.geometry.footprint_geometry(pose)
        forward = nx*(cx-x)+ny*(cy-y)
        forward_extent = half_length*abs(nx*c+ny*s)+half_width*abs(-nx*s+ny*c)
        # Hand off when every padded body corner is outside the exit plane.
        return forward-forward_extent >= 0.

    def _request_global_plan(self, pose, now):
        if (self.worker is not None and self.worker.is_alive()) or self.pending_route is not None:
            return
        if (self.last_plan_attempt is not None
                and now-self.last_plan_attempt < self.navigation['replan_period']):
            return
        # Grid snapshots are replaced by mapping, not modified in place. Search
        # runs outside the mission lock so odometry and scans can keep arriving.
        grid, generation = self._grid(), self.generation
        self.last_plan_attempt = now
        started = time.monotonic()
        self.plans += 1

        def search():
            try:
                route = self.local_planner.global_route(grid, pose, self.goal)
                error = '' if route is not None else 'no global route; previous guidance retained'
            except (ValueError, RuntimeError) as exc:
                route, error = None, str(exc)
            duration = time.monotonic()-started
            with self.lock:
                if self.generation == generation:
                    self.pending_route = RouteUpdate(route, error, duration)

        self.worker = threading.Thread(target=search, daemon=True)
        self.worker.start()

    def _accept_global_plan(self):
        if self.pending_route is None:
            return
        update = self.pending_route
        self.pending_route = None
        if update.route is None:
            self.global_report = update.error
            return
        # This polyline is guidance, not a committed trajectory. A new scan or
        # motion during search need not discard it: local control projects from
        # the current pose and checks the actual body against the current map.
        self.global_path = update.route
        self.global_updates += 1
        self.state = 'FOLLOWING'
        self.global_report = 'global route %d updated in %.3fs' % (self.global_updates, update.duration)

    def _finish_goal(self, pose, elapsed, now, grid):
        """Hand off after physical passage; retain endpoint alignment otherwise."""
        # Once reached, finish even if momentum changes the accepted pose.
        self.state = 'FINISHING'
        cfg = self.control
        if self.exit_crossing is not None:
            # End tunnel ownership now. The adapter clears old lane commands
            # and lets the next image drive; no measured-stop wait is needed.
            self.state = 'COMPLETE'
            return self._zero('exit crossed; handing control to lane tracking')
        measured_linear = self.odom[3]
        heading_error = normalize_angle(self.goal.yaw-pose.yaw)
        heading_ok = abs(heading_error) <= math.radians(self.config['goal']['heading_tolerance_deg'])
        if (self.last_linear > 1e-9
                or abs(measured_linear) > cfg['planning_stopped_linear']):
            command = self._stop_command(elapsed)
            reason = 'stopping at outside goal'
        elif not heading_ok:
            # Align at the final stop; check the next control-period sweep below.
            command = turn_command(
                heading_error, 0., self.last_angular, min(elapsed, cfg['period']),
                cfg, self.navigation['local_rotation_gain'])
            reason = 'aligning heading at outside goal'
        else:
            command = self._stop_command(elapsed)
            reason = 'stopping at outside goal'
            if command.linear_velocity <= 1e-9 and abs(command.angular_velocity) <= 1e-9:
                self.state = 'WAIT_LANE' if self.completion == 'lane' else 'WAIT_STOP'
                self.completion_started = max(now, self.odom[-1])
                self.lane_count = 0
                return self._zero('waiting for exit lane observation' if self.completion == 'lane'
                                  else 'waiting for measured stop at outside goal')
        if not self.local_planner.command_is_collision_free(grid, pose, command, cfg):
            return self._zero('outside goal command overlaps mapped obstacle')
        return self._command(command, now, reason)

    def _wait_for_completion(self, pose, elapsed, now, grid):
        if self.state == 'WAIT_LANE':
            fresh_lane = (self.lane_stamp is not None and self.lane_stamp > self.completion_started
                          and 0. <= now-self.lane_stamp <= self.sensors['lane_maximum_age'])
            if fresh_lane and self.lane_count >= self.sensors['lane_confirmation_frames']:
                self.state = 'COMPLETE'
                return self._zero('exit lane confirmed')
            return self._zero('waiting for exit lane observation')
        if self._measured_stop_confirmed():
            heading_error = abs(normalize_angle(self.goal.yaw-pose.yaw))
            if heading_error > math.radians(self.config['goal']['heading_tolerance_deg']):
                return self._finish_goal(pose, elapsed, now, grid)
            self.state = 'COMPLETE'
            return self._zero('goal reached; measured stop confirmed')
        return self._zero('waiting for measured stop at outside goal')
