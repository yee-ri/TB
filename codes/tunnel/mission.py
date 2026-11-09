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
from .localization import AlignmentResult, WallLocalizer
from .entrance_localization import EntranceLandmarkLocalizer
from .exit_localization import ExitLandmarkLocalizer
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


def map_anchor(scan_odom, fitted):
    """odom_T_map from the fitted base pose and odometry at scan acquisition."""
    yaw = normalize_angle(scan_odom[2] - fitted.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return (scan_odom[0] - c*fitted.x + s*fitted.y,
            scan_odom[1] - s*fitted.x - c*fitted.y, yaw)

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


@dataclass
class RouteTurn:
    phase: str
    target_yaw: float
    angular_velocity: float
    target_angle: float = 0.
    direction: float = 0.
    angle: float = 0.


class TunnelMission:
    def __init__(self, config, clock=None):
        self.navigation = load_navigation_settings(config.get('navigation'))
        period = self.navigation['replan_period']
        if (isinstance(period, bool) or not isinstance(period, (int, float))
                or not math.isfinite(period) or period <= 0):
            raise ValueError('navigation/replan_period must be finite and positive')
        change_angle = self.navigation['route_change_alignment_angle_deg']
        if (isinstance(change_angle, bool) or not isinstance(change_angle, (int, float))
                or not math.isfinite(change_angle) or not 0. < change_angle <= 180.):
            raise ValueError('navigation/route_change_alignment_angle_deg must be in (0, 180]')
        self.route_change_alignment_angle = math.radians(change_angle)
        self.config = config
        self.clock = clock
        self.control, self.sensors = config['control'], config['sensors']
        for key in ('period', 'linear_acceleration', 'linear_deceleration',
                    'angular_acceleration', 'safety_linear_deceleration',
                    'safety_reaction_time'):
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
                ('alignment_angular_acceleration', .25, self.control['angular_acceleration'])):
            value = config['entry'].get(key, default)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(value) or not 0. < value <= maximum):
                raise ValueError('entry/%s must be finite, positive and <= %s' % (key, maximum))
            setattr(self, 'entry_' + key, float(value))
        goal = config['goal']['pose']
        self.goal = Pose2D(float(goal[0]), float(goal[1]), math.radians(goal[2]))
        exit_speed = config['goal'].get(
            'exit_alignment_angular_velocity', self.entry_alignment_angular_velocity)
        if (isinstance(exit_speed, bool) or not isinstance(exit_speed, (int, float))
                or not math.isfinite(exit_speed)
                or not 0. < exit_speed <= self.control['maximum_angular_velocity']):
            raise ValueError('goal/exit_alignment_angular_velocity must be finite, positive and within the angular limit')
        self.exit_alignment_angular_velocity = float(exit_speed)
        approach_velocity = config['goal'].get('approach_velocity', self.control['cruise_velocity'])
        if (isinstance(approach_velocity, bool) or not isinstance(approach_velocity, (int, float))
                or not math.isfinite(approach_velocity) or approach_velocity <= 0.):
            raise ValueError('goal/approach_velocity must be finite and positive')
        self.exit_approach_velocity = float(approach_velocity)
        slowdown_distance = config['goal'].get('approach_slowdown_distance', 0.)
        if (isinstance(slowdown_distance, bool) or not isinstance(slowdown_distance, (int, float))
                or not math.isfinite(slowdown_distance) or slowdown_distance < 0.):
            raise ValueError('goal/approach_slowdown_distance must be finite and nonnegative')
        self.exit_approach_slowdown_distance = float(slowdown_distance)
        self.exit_crossing = self._exit_crossing_geometry(config['goal'].get('exit_crossing'))
        self.exit_approach_goal = None
        approach_distance = config['goal'].get('approach_distance')
        if approach_distance is not None:
            if (isinstance(approach_distance, bool)
                    or not isinstance(approach_distance, (int, float))
                    or not math.isfinite(approach_distance) or approach_distance <= 0.
                    or self.exit_crossing is None):
                raise ValueError('goal/approach_distance requires a positive distance and exit_crossing')
            x, y, tx, ty, nx, ny, width = self.exit_crossing
            self.exit_approach_goal = Pose2D(
                x+tx*width/2.-nx*approach_distance,
                y+ty*width/2.-ny*approach_distance, math.atan2(ny, nx))
        self.lock = threading.RLock()
        self.odom = self.scan = None
        self.localization_scan = None
        self.localizer = None
        self.entrance_localizer = None
        self.exit_localizer = None
        self.wall_tracking_period = 0.
        if 'localization' in config:
            settings = config['localization']
            self.localizer = WallLocalizer(settings['wall_rectangles'])
            self.entrance_localizer = EntranceLandmarkLocalizer.from_geometry(
                settings, **settings.get('entrance', {}))
            if 'exit' in settings:
                self.exit_localizer = ExitLandmarkLocalizer.from_geometry(
                    dict(settings, exit_clear_segment=config['goal']['exit_crossing']),
                    **settings['exit'])
            window = settings.get('observation_window', 0.)
            if isinstance(window, bool) or not isinstance(window, (int, float)) or not math.isfinite(window) or not 0 <= window <= 2.:
                raise ValueError('localization/observation_window must be between 0 and 2 seconds')
            for key, default in (('tracking_period', 0.), ('tracking_max_translation', .08),
                                 ('tracking_max_yaw_deg', 6.)):
                value = settings.get(key, default)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0.
                        or (key != 'tracking_period' and value == 0.)):
                    raise ValueError('localization/%s must be finite and positive (period may be zero)' % key)
                setattr(self, 'wall_' + key, float(value))
            if self.wall_tracking_max_yaw_deg >= 90.:
                raise ValueError('localization/tracking_max_yaw_deg must be less than 90')
        self.front_clearance = math.inf
        self.lane_stamp = None
        self.lane_count = 0
        self.generation = 0
        self.worker = None
        self.localization_worker = None
        self.wall_tracking_worker = None
        self.reset()
        self.local_planner = GlobalLocalPlanner(self.geometry, self.navigation)
        self.completion = config['goal'].get('completion', 'stop')
        if self.completion not in ('stop', 'lane'):
            raise ValueError('goal/completion must be stop or lane')

    def reset(self):
        with self.lock:
            self.generation += 1
            self.state = 'LOCALIZING' if self.localizer is not None else 'ENTRY'
            self.reason = 'waiting for sensors'
            self.anchor = self.costmap = self.grid = None
            self.static_cell_rectangles = {}
            self.last_plan_attempt = None
            self.mapped_stamp = self.last_time = None
            self.plans = 0
            self.last_linear = self.last_angular = 0.0
            self.lane_count = 0
            self.completion_started = None
            self.localization_result = None
            self.localization_epoch = getattr(self, 'localization_epoch', 0) + 1
            self.localization_history = []
            self.localization_start = None
            self.localization_attempt_time = None
            self.localization_report = ''
            self.wall_tracking_attempt_stamp = None
            self.wall_tracking_started = None
            self.wall_tracking_scans = None
            self.wall_tracking_updates = 0
            self.wall_tracking_report = ''
            self.entry_start_x = 0.
            self.entry_start_yaw = 0.
            self.entry_phase = 'ALIGN' if self.entry_heading_offset else 'DRIVE'
            self.entry_turn_angle = 0.
            self.entry_drive_yaw = None

            self.global_path = None
            self.route_goal = self.exit_approach_goal or self.goal
            self.route_turn = None
            self.pending_route = None
            self.global_updates = 0
            self.global_report = 'waiting for first global route'
            self.exit_previous_position = None
            self.exit_crossed = False
            self.exit_crossing_position = None

    def close(self):
        self.reset()

    def step(self, now):
        with self.lock:
            frame, response = self._prepare_step(now)
            if response is not None:
                return response
            now, elapsed, pose = frame.now, frame.elapsed, frame.pose
            if self.state == 'ENTRY':
                return self._enter(pose, elapsed)
            grid = self._grid()
            if self.state in ('WAIT_STOP', 'WAIT_LANE'):
                return self._wait_for_completion(pose, elapsed, now)
            if self.route_turn is None and (self.state == 'FINISHING' or self._goal_reached(pose)):
                return self._finish_goal(pose, elapsed, now)
            if self._advance_exit_goal(pose):
                return self._zero('exit approach area reached; planning outside route')
            previous_route = self.global_path
            updated = self._accept_global_plan()
            self._request_global_plan(pose, now)
            if self.global_path is None:
                return self._zero('waiting for first global route; ' + self.global_report)
            if updated:
                heading = self.local_planner.route_heading(self.global_path, pose)
                changed = (previous_route is None or abs(normalize_angle(
                    heading-self.local_planner.route_heading(previous_route, pose)))
                    >= self.route_change_alignment_angle)
                if changed:
                    exit_alignment = (previous_route is None
                                      and self.exit_approach_goal is not None
                                      and self.route_goal is self.goal)
                    speed = (self.exit_alignment_angular_velocity if exit_alignment
                             else self.entry_alignment_angular_velocity)
                    return self._start_route_alignment(heading, speed)
            if self.route_turn is not None:
                response = self._follow_route_turn(pose, elapsed)
                if response is not None:
                    return response
            command = self.local_planner.local_command(
                grid, pose, self.global_path, self.control,
                self.last_linear, self.last_angular, elapsed,
                speed_limit=self._approach_speed_limit(pose))
            if command is None:
                return self._zero('no local command; ' + self.global_report)
            return self._command(command, command.reason + '; ' + self.global_report)

    def update_odometry(self, pose, linear, angular, stamp):
        values = tuple(float(x) for x in pose) + (float(linear), float(angular), float(stamp))
        if len(values) != 6 or not all(math.isfinite(x) for x in values):
            return False
        with self.lock:
            if self.odom is not None and stamp <= self.odom[-1]:
                return False
            self.odom = values
        return True

    def _start_route_alignment(self, heading, angular_velocity):
        self.route_turn = RouteTurn('PREPARE', heading, angular_velocity)
        return self._zero('route alignment: stopping before heading change')

    def _follow_route_turn(self, pose, elapsed):
        turn = self.route_turn
        if turn.phase == 'PREPARE':
            error = normalize_angle(turn.target_yaw-pose.yaw)
            turn.direction = math.copysign(1., error)
            turn.target_angle = abs(error)
            turn.phase = 'TURN'
        if turn.angle >= turn.target_angle:
            self.route_turn = None
            return None
        # Send the requested turn at the configured pivot speed; do not taper
        # into a motor dead zone or wait for a measured final heading.
        target = turn.direction * turn.angular_velocity
        delta = self.control['angular_acceleration'] * min(elapsed, self.control['period'])
        angular = max(self.last_angular-delta, min(self.last_angular+delta, target))
        command = LimitedCommand(0., angular)
        return self._command(command, 'route alignment: turning toward route heading')

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
        if not all(math.isfinite(v) for v in (*sensor_pose, *odom_pose, stamp)):
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
        if not np.any(valid) and (_points is None or not len(_points)):
            return False
        ranges[~valid] = np.nan
        angles = angle_min + np.flatnonzero(hits) * angle_increment + sensor_pose[2]
        x = sensor_pose[0] + ranges[hits] * np.cos(angles)
        y = sensor_pose[1] + ranges[hits] * np.sin(angles)
        # Entry only: widen the left edge toward the far end of a trapezoid.
        # Keep the complete scan for mapping and path clearance scores.
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
            if self.localizer is not None and (self.anchor is None or self.wall_tracking_period):
                # Preserve accepted raw XY for alignment: scan bin centres
                # quantize the bearing and can shift distant walls by mm/cm.
                points = np.column_stack((x, y))[ranges[hits] <= effective_max] if _points is None else _points.copy()
                self.localization_scan = (points, tuple(odom_pose), float(stamp))
                if self.anchor is None:
                    window = self.config['localization'].get('observation_window', 0.)
                    self.localization_history = [row for row in self.localization_history if stamp-row[2] <= window+.001]
                    self.localization_history.append(self.localization_scan)
            if self.anchor is not None:
                if not self._track_walls():
                    self._map_scan()
        return True

    def _map_scan(self, scan=None):
        ranges, angle_min, increment, range_min, range_max, sensor, odom_pose, stamp = self.scan if scan is None else scan
        if self.mapped_stamp is not None and stamp <= self.mapped_stamp:
            return
        pose = relative_pose(odom_pose, self.anchor)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        sensor_local = (pose.x + c * sensor[0] - s * sensor[1],
                        pose.y + s * sensor[0] + c * sensor[1], pose.yaw + sensor[2])
        update = self.costmap.update_scan(ranges, angle_min, increment, range_min, range_max, sensor_local)
        self.mapped_stamp = stamp
        if (update.marked_cells or update.cleared_cells or update.decayed_cells
                or update.refined_cells):
            self.grid = None

    def _track_walls(self):
        """Fit in the background; map this scan only after its pose is checked.

        Old obstacles stay in the fixed map where they were observed. The fit's
        scan and the bounded set received while fitting use the corrected pose,
        avoiding duplicate obstacle marks on either side of a pose correction.
        Control and global planning continue using the previous pose/map while
        the fit runs; rejecting a fit only falls back to odometry.
        """
        if not self.wall_tracking_period or self.state == 'COMPLETE':
            return False
        if self.wall_tracking_worker is not None and self.wall_tracking_worker.is_alive():
            pending = self.wall_tracking_scans
            if pending is None:
                return False
            # Never drop intermediate scans: they confirm/clear obstacles.
            # If fitting falls behind, flush with odometry and reject its late
            # correction, rather than queue indefinitely or freeze mapping.
            if (len(pending) < 4 and
                    time.monotonic() - self.wall_tracking_started <= self.sensors['maximum_age']):
                pending.append(self.scan)
                return True
            self.wall_tracking_scans = None
            for deferred in pending:
                self._map_scan(deferred)
            return False
        if self.localization_scan is None:
            return False
        points, scan_odom, stamp = self.localization_scan
        if (self.wall_tracking_attempt_stamp is not None
                and stamp - self.wall_tracking_attempt_stamp < self.wall_tracking_period - 1e-6):
            return False
        self.wall_tracking_attempt_stamp = stamp
        scan, generation = self.scan, self.generation
        sensor_origin = scan[5][:2]
        pending = [scan]
        self.wall_tracking_scans = pending
        prior = relative_pose(scan_odom, self.anchor)
        started = time.monotonic()
        self.wall_tracking_started = started

        def track():
            source = 'walls'
            try:
                arguments = (points, prior, self.wall_tracking_max_translation,
                             math.radians(self.wall_tracking_max_yaw_deg), sensor_origin)
                result = (self.exit_localizer.track(*arguments)
                          if self.exit_localizer is not None else AlignmentResult(False))
                if result.accepted:
                    source = 'exit opening'
                else:
                    result = self.localizer.track(*arguments)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                result = AlignmentResult(False, reason='wall tracking failed: ' + str(error))
            duration = time.monotonic() - started
            with self.lock:
                if self.generation != generation or self.anchor is None or self.state == 'COMPLETE':
                    return
                self.wall_tracking_started = None
                # This limit discards a late correction, not a drive command.
                # Odometry predicts current pose from the fitted scan's time.
                age = max(time.monotonic() - started, self.scan[-1] - stamp)
                if self.wall_tracking_scans is not pending or age > self.sensors['maximum_age']:
                    result = AlignmentResult(False, reason='late wall correction discarded')
                self.wall_tracking_scans = None
                if result.accepted:
                    self._apply_tracking_pose(scan_odom, result.pose)
                    self.wall_tracking_updates += 1
                    self.wall_tracking_report = (
                        'accepted dx=%.3fm dy=%.3fm dyaw=%.2fdeg rms=%.4fm inliers=%d/%d fit=%.1fms source=%s'
                        % (result.pose.x-prior.x, result.pose.y-prior.y,
                           math.degrees(normalize_angle(result.pose.yaw-prior.yaw)),
                           result.rms, result.inliers, result.total, duration*1000., source))
                else:
                    self.wall_tracking_report = 'odom retained: %s; fit=%.1fms' % (result.reason, duration*1000.)
                for deferred in pending:
                    self._map_scan(deferred)
                self._map_scan()

        self.wall_tracking_worker = threading.Thread(target=track, daemon=True)
        self.wall_tracking_worker.start()
        return True

    def _apply_tracking_pose(self, scan_odom, fitted):
        new_anchor = map_anchor(scan_odom, fitted)
        if self.exit_crossing is not None:
            x, y, tx, ty, nx, ny, width = self.exit_crossing
            ax, ay, yaw = self.anchor
            c, s = math.cos(yaw), math.sin(yaw)

            def corrected_position(px, py):
                return relative_pose((ax+c*px-s*py, ay+s*px+c*py, 0.), new_anchor)

            if self.exit_previous_position is not None:
                # Compare both positions in the same corrected map frame so
                # a localization jump cannot be mistaken for outward motion.
                forward, lateral = self.exit_previous_position
                px, py = x+tx*lateral+nx*forward, y+ty*lateral+ny*forward
                previous = corrected_position(px, py)
                self.exit_previous_position = (
                    nx*(previous.x-x)+ny*(previous.y-y),
                    tx*(previous.x-x)+ty*(previous.y-y))
            if self.exit_crossing_position is not None:
                crossing = corrected_position(*self.exit_crossing_position)
                self.exit_crossing_position = crossing.x, crossing.y
                if not 0. <= tx*(crossing.x-x)+ty*(crossing.y-y) <= width:
                    self.exit_crossed = False
            current = relative_pose(self.odom[:3], new_anchor)
            if nx*(current.x-x)+ny*(current.y-y) < 0.:
                self.exit_crossed = False
            if not self.exit_crossed:
                self.exit_crossing_position = None
        self.anchor = new_anchor

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

    def _invalidate_localization(self):
        self.localization_epoch += 1
        self.localization_result = None
        self.localization_history = []

    def _start_localized_entry(self, fitted, reference_odom, report):
        """Initialize once; later position corrections belong to wall tracking."""
        self._invalidate_localization()
        self._initialize(map_anchor(reference_odom, fitted))
        pose = relative_pose(self.odom[:3], self.anchor)
        self.entry_start_x = pose.x
        self.entry_start_yaw = pose.yaw
        self.localization_report = report
        self.reason = report
        self.state = 'ENTRY'
        return True

    def _start_from_entrance_prior(self, reason):
        fitted = Pose2D(*self.entrance_localizer.prior)
        return self._start_localized_entry(
            fitted, self.localization_start[1],
            'entrance prior assumed x=%.3fm y=%.3fm yaw=%.2fdeg: %s' % (
                fitted.x, fitted.y, math.degrees(fitted.yaw), reason))

    def _localize(self, now):
        """Prefer the measured entrance pose; unusable fits retain the entry prior."""
        if self.localization_start is None:
            self.localization_start = (now, self.odom[:3])
        if self.localization_result is not None:
            result, scan_odom, stamp = self.localization_result
            self.localization_result = None
            if not result.accepted:
                return self._start_from_entrance_prior(result.reason)
            if now-stamp > self.sensors['maximum_age']:
                return self._start_from_entrance_prior('entrance fit input expired')
            fitted = result.pose
            diagnostics = self.entrance_localizer.diagnostics
            corner = diagnostics.get('endpoint_base')
            details = (' corner_base=(%.3f,%.3f)m heading=%s' % (
                corner[0], corner[1], diagnostics.get('heading_source', 'right wall'))
                if corner is not None else '')
            return self._start_localized_entry(
                fitted, scan_odom,
                'entrance measured x=%.3fm y=%.3fm yaw=%.2fdeg rms=%.4fm%s' % (
                    fitted.x, fitted.y, math.degrees(fitted.yaw), result.rms, details))
        if self.localization_attempt_time is not None:
            # Stop waiting once this fit could no longer supply a usable result.
            if now-self.localization_attempt_time >= self.sensors['maximum_age']:
                return self._start_from_entrance_prior('entrance fit exceeded input lifetime')
            if not self.localization_worker.is_alive():
                return self._start_from_entrance_prior('entrance fit ended without a result')
            return False
        window = self.config['localization'].get('observation_window', 0.)
        if now-self.localization_start[0] < window:
            self.reason = 'collecting entrance observations'
            return False
        if self.localization_scan is None:
            return self._start_from_entrance_prior('entrance observations missing')
        points, scan_odom, stamp = self.localization_scan
        if now-stamp > self.sensors['maximum_age']:
            return self._start_from_entrance_prior('entrance observations expired')
        if window and self.localization_history:
            combined = []
            for hits, acquired_pose, _ in self.localization_history:
                relative = relative_pose(acquired_pose, scan_odom)
                c, s = math.cos(relative.yaw), math.sin(relative.yaw)
                combined.append(hits @ np.array(((c, s), (-s, c))) + (relative.x, relative.y))
            points = np.concatenate(combined)
        self.localization_attempt_time = now
        generation = self.generation
        epoch = self.localization_epoch
        sensor_origin = self.scan[5][:2]
        def align():
            try:
                result = self.entrance_localizer.estimate(points, sensor_origin)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                result = AlignmentResult(False, reason='wall alignment failed: '+str(error))
            with self.lock:
                if self.generation == generation and self.localization_epoch == epoch:
                    self.localization_result = (result, scan_odom, stamp)
        self.localization_worker = threading.Thread(target=align, daemon=True)
        self.localization_worker.start()
        return False

    def _grid(self):
        if self.grid is None:
            c = self.costmap
            # Static walls and measured obstacles share one planning grid;
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

    def _command(self, command, reason):
        self.last_linear, self.last_angular = command.linear_velocity, command.angular_velocity
        self.reason = reason
        return self.last_linear, self.last_angular, False

    def _align_entry(self, pose, elapsed):
        """Turn once; return None to begin straight entry in this same tick."""
        cfg = self.control
        direction = math.copysign(1., self.entry_heading_offset)
        if self.entry_turn_angle >= abs(self.entry_heading_offset):
            # Finish from the sent turn, then hold the current heading while driving.
            self.entry_drive_yaw = pose.yaw
            self.entry_phase = 'DRIVE'
            return None
        acceleration = self.entry_alignment_angular_acceleration
        target = direction * self.entry_alignment_angular_velocity
        change = acceleration * min(elapsed, cfg['period'])
        angular = max(self.last_angular-change, min(self.last_angular+change, target))
        return self._command(LimitedCommand(0., angular),
                             'aligning entry heading before straight drive')

    def _enter(self, pose, elapsed):
        if self.entry_phase != 'DRIVE':
            alignment_command = self._align_entry(pose, elapsed)
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
                return self._command(command, 'stopping at ' + entry_name + ' target')
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
        return self._command(command, entry_name)

    def _stop_command(self, elapsed):
        cfg = self.control
        return limit_tracking_command(
            0., 1., 0., self.last_linear, self.last_angular,
            min(elapsed, cfg['period']), cfg['linear_acceleration'],
            cfg['linear_deceleration'], cfg['angular_acceleration'],
            cfg['maximum_angular_velocity'], cfg['maximum_lateral_acceleration'])

    def _sensor_times_valid(self, now):
        latest = now + self.sensors['maximum_future_stamp']
        return self.odom[-1] <= latest and self.scan[-1] <= latest

    def _create_geometry(self, settings):
        keys = ('minimum_turning_radius', 'inflation_radius', 'keep_in_rectangles')
        return CollisionGeometry(self.footprint, **{key: settings[key] for key in keys if key in settings})

    def _prepare_step(self, now):
        """Return the available control frame, or an already-decided stop."""
        if self.clock is not None:
            now = self.clock()
        elapsed = self.control['period'] if self.last_time is None else max(0., now-self.last_time)
        self.last_time = now
        if self.state == 'ENTRY' and self.entry_phase == 'ALIGN':
            # Count the command held since the previous tick, including before a stop.
            self.entry_turn_angle += math.copysign(1., self.entry_heading_offset) * self.last_angular * elapsed
        if self.route_turn is not None:
            # Integrate the command held since the previous tick, before any
            # stop branch can reset it. This is commanded, not measured, yaw.
            self.route_turn.angle += self.route_turn.direction * self.last_angular * elapsed
        if self.state == 'COMPLETE':
            return None, self._zero(self.reason)
        if self.odom is None or self.scan is None:
            return None, self._zero('waiting for odometry and lidar')
        if not self._sensor_times_valid(now):
            if self.localizer is not None and self.anchor is None:
                self._invalidate_localization()
            return None, self._zero('odometry or lidar timestamp is in the future')
        if self.anchor is None:
            if self.localizer is not None:
                if not self._localize(now):
                    return None, self._zero(self.reason)
            else:
                self._initialize()
        pose = relative_pose(self.odom[:3], self.anchor)
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
                self.exit_crossing_position = (x+tx*crossing, y+ty*crossing) if self.exit_crossed else None
            elif old_forward >= 0. > forward:
                # Returning inside requires a new outward passage through
                # this opening; an earlier passage cannot qualify another exit.
                self.exit_crossed = False
                self.exit_crossing_position = None
        if not self.exit_crossed:
            return False
        cx, cy, c, s, half_length, half_width = self.geometry.footprint_geometry(pose)
        forward = nx*(cx-x)+ny*(cy-y)
        forward_extent = half_length*abs(nx*c+ny*s)+half_width*abs(-nx*s+ny*c)
        # Hand off when every padded body corner is outside the exit plane.
        return forward-forward_extent >= 0.

    def _approach_speed_limit(self, pose):
        """Lower the speed target only within the inside goal's approach zone."""
        if self.exit_approach_goal is None or self.route_goal is not self.exit_approach_goal:
            return None
        distance = math.hypot(pose.x-self.route_goal.x, pose.y-self.route_goal.y)
        if (distance > self.exit_approach_slowdown_distance
                or self.exit_approach_velocity >= self.control['cruise_velocity']):
            return None
        return self.exit_approach_velocity

    def _advance_exit_goal(self, pose):
        """Accept the axle inside the exit corridor, then latch the outside goal."""
        if self.exit_approach_goal is None or self.route_goal is not self.exit_approach_goal:
            return False
        x, y, tx, ty, nx, ny, width = self.exit_crossing
        forward = nx*(pose.x-x)+ny*(pose.y-y)
        lateral = tx*(pose.x-x)+ty*(pose.y-y)
        # Keep the former earliest arrival depth, but accept the space from
        # there up to the opening instead of chasing a 3.5 cm target circle.
        depth = (self.config['goal']['approach_distance']
                 + self.config['goal']['position_tolerance'])
        half_width = self.footprint.half_width+self.footprint.padding
        epsilon = 1e-9  # Coordinate roundoff only; no added physical clearance.
        if (not -depth-epsilon <= forward <= epsilon
                or not half_width-epsilon <= lateral <= width-half_width+epsilon):
            return False
        self.route_goal = self.goal
        self.global_path = self.pending_route = self.route_turn = None
        self.last_plan_attempt = None
        self.worker = None
        self.state = 'PLANNING'
        self.global_report = 'waiting for outside route'
        return True

    def _request_global_plan(self, pose, now):
        if (self.worker is not None and self.worker.is_alive()) or self.pending_route is not None:
            return
        if (self.last_plan_attempt is not None
                and now-self.last_plan_attempt < self.navigation['replan_period']):
            return
        # Grid snapshots are replaced by mapping, not modified in place. Search
        # runs outside the mission lock so odometry and scans can keep arriving.
        grid, generation, goal = self._grid(), self.generation, self.route_goal
        self.last_plan_attempt = now
        started = time.monotonic()
        self.plans += 1

        def search():
            try:
                route = self.local_planner.global_route(grid, pose, goal)
                error = '' if route is not None else 'no global route; previous guidance retained'
            except (ValueError, RuntimeError) as exc:
                route, error = None, str(exc)
            duration = time.monotonic()-started
            with self.lock:
                if self.generation == generation and self.route_goal == goal:
                    self.pending_route = RouteUpdate(route, error, duration)

        self.worker = threading.Thread(target=search, daemon=True)
        self.worker.start()

    def _accept_global_plan(self):
        if self.pending_route is None:
            return False
        update = self.pending_route
        self.pending_route = None
        if update.route is None:
            self.global_report = update.error
            return False
        # This polyline is guidance, not a committed trajectory. A new scan or
        # motion during search need not discard it: local control projects from
        # the current pose and scores body clearance against the current map.
        self.global_path = update.route
        self.global_updates += 1
        self.state = 'FOLLOWING'
        self.global_report = 'global route %d updated in %.3fs' % (self.global_updates, update.duration)
        return True

    def _finish_goal(self, pose, elapsed, now):
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
            # Align at the final stop using the normal command limits.
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
        return self._command(command, reason)

    def _wait_for_completion(self, pose, elapsed, now):
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
                return self._finish_goal(pose, elapsed, now)
            self.state = 'COMPLETE'
            return self._zero('goal reached; measured stop confirmed')
        return self._zero('waiting for measured stop at outside goal')
