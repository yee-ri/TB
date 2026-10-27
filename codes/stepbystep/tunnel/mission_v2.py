"""Reference-style 2-D global guidance and current-pose local control.

The global worker never owns motor commands. Each control tick checks local
motion against the latest map, including the actual rate-limited command and
its braking sweep. Updating guidance does not reset speed or require a stop.
"""

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from .mission import TunnelMission, relative_pose
from .planner_v2 import GlobalLocalPlanner
from .tracking import normalize_angle


class TunnelMissionV2(TunnelMission):
    def __init__(self, config, clock=None):
        with Path(__file__).with_name('v2.yaml').open(encoding='utf-8') as stream:
            self.v2 = yaml.safe_load(stream)
        self.v2.update(config.get('v2', {}))
        period = self.v2['replan_period']
        if (isinstance(period, bool) or not isinstance(period, (int, float))
                or not math.isfinite(period) or period <= 0):
            raise ValueError('v2/replan_period must be finite and positive')
        super().__init__(config, clock)
        self.local_planner = GlobalLocalPlanner(self.planner, self.v2)
        self.exit_crossing = self._exit_crossing_geometry(config['goal'].get('exit_crossing'))

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
        cx, cy, c, s, half_length, half_width = self.planner._footprint_geometry(pose)
        forward = nx*(cx-x)+ny*(cy-y)
        forward_extent = half_length*abs(nx*c+ny*s)+half_width*abs(-nx*s+ny*c)
        # Once every corner is outside, the aperture does not extend into an
        # imaginary corridor. The goal point/yaw no longer owns the handoff.
        return forward-forward_extent >= 0.

    def reset(self):
        super().reset()
        with self.lock:
            self.global_path = None
            self.pending_route = None
            self.global_updates = 0
            self.global_report = 'waiting for first global route'
            self.exit_previous_position = None
            self.exit_crossed = False

    def _request_global_plan(self, pose, now):
        if (self.worker is not None and self.worker.is_alive()) or self.pending_route is not None:
            return
        if (self.last_plan_attempt is not None
                and now-self.last_plan_attempt < self.v2['replan_period']):
            return
        # Grid snapshots are replaced by mapping, not modified in place. Search
        # runs outside the mission lock so odometry and scans can keep arriving.
        grid, generation = self._grid(), self.generation
        self.last_plan_attempt = now
        started = self.planning_started = time.monotonic()
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
                    self.pending_route = (route, error, pose, duration)

        self.worker = threading.Thread(target=search, daemon=True)
        self.worker.start()

    def _accept_global_plan(self, pose, grid):
        if self.pending_route is None:
            return
        route, error, start, duration = self.pending_route
        self.pending_route = None
        self.planning_duration = duration
        if route is None:
            self.global_report = error
            return
        # This polyline is guidance, not a committed trajectory. A new scan or
        # motion during search need not discard it: local control projects from
        # the current pose and checks the actual body against the current map.
        self.global_path = route
        self.global_updates += 1
        self.state = 'FOLLOWING'
        self.global_report = 'global route %d updated in %.3fs' % (self.global_updates, duration)

    def _finish_goal(self, pose, elapsed, now, grid):
        """Hand off after physical passage; retain endpoint alignment otherwise."""
        # Once reached, finish even if momentum changes the accepted pose.
        self.state = 'FINISHING'
        cfg = self.control
        measured_linear, measured_angular = self.odom[3:5]
        if self.exit_crossing is not None:
            # End tunnel ownership now. The adapter clears old lane commands
            # and lets the next image drive; no measured-stop wait is needed.
            command = SimpleNamespace(linear_velocity=0., angular_velocity=0.)
            if not self.local_planner.command_is_safe(
                    grid, pose, command, cfg, measured_linear, measured_angular):
                return self._zero('exit crossing stopping sweep is blocked')
            self.state = 'COMPLETE'
            return self._zero('exit crossed; handing control to lane tracking')
        heading_error = normalize_angle(self.goal.yaw-pose.yaw)
        heading_ok = abs(heading_error) <= math.radians(self.config['goal']['heading_tolerance_deg'])
        if (self.last_linear > 1e-9
                or abs(measured_linear) > cfg['planning_stopped_linear']):
            command = self._stop_command(elapsed)
            reason = 'stopping at outside goal'
        elif not heading_ok:
            # Goal alignment happens only at the already-required final stop.
            # A limited angular command must pass the same swept-body and
            # braking check as every local driving command.
            decel = cfg['safety_angular_deceleration']
            reaction = cfg['safety_reaction_time'] + cfg['period']
            cap = math.sqrt((decel*reaction)**2 + 2.*decel*abs(heading_error)) - decel*reaction
            target = math.copysign(min(cfg['maximum_angular_velocity'], cap,
                                      self.v2['local_rotation_gain']*abs(heading_error)), heading_error)
            change = cfg['angular_acceleration']*min(elapsed, cfg['period'])
            angular = max(self.last_angular-change, min(self.last_angular+change, target))
            command = SimpleNamespace(linear_velocity=0., angular_velocity=angular)
            reason = 'aligning heading at outside goal'
        else:
            command = self._stop_command(elapsed)
            reason = 'stopping at outside goal'
            if command.linear_velocity <= 1e-9 and abs(command.angular_velocity) <= 1e-9:
                self.state = 'WAIT_STOP'
                self.completion_started = max(now, self.odom[-1])
                return self._zero('waiting for measured stop at outside goal')
        if not self.local_planner.command_is_safe(
                grid, pose, command, cfg, measured_linear, measured_angular):
            return self._zero('outside goal stopping or rotation sweep is blocked')
        return self._command(command, now, reason)

    def step(self, now):
        with self.lock:
            # Entry registration stays identical to V1. Only unconfigured
            # exits retain the final stopped-heading alignment below.
            if self.state in ('LOCALIZING', 'ENTRY', 'COMPLETE') or self.anchor is None:
                return super().step(now)
            if self.clock is not None:
                now = self.clock()
            elapsed = self.control['period'] if self.last_time is None else max(0., now-self.last_time)
            self.last_time = now
            if self.odom is None or self.scan is None:
                return self._zero('waiting for odometry and lidar')
            if any(not -self.sensors['maximum_future_stamp'] <= now-stamp <= self.sensors['maximum_age']
                   for stamp in (self.odom[-1], self.scan[-1])):
                return self._zero(self._stale_sensor_reason(now))
            self.freshness_report = None
            pose = relative_pose(self.odom[:3], self.anchor)
            grid = self._grid()
            if not self.planner.pose_is_collision_free(grid, pose):
                return self._zero('current footprint overlaps mapped obstacle')
            if self.state == 'WAIT_STOP':
                stopped = (self.odom[-1] > self.completion_started
                           and abs(self.odom[3]) <= self.control['planning_stopped_linear']
                           and abs(self.odom[4]) <= self.control['planning_stopped_angular'])
                heading_error = abs(normalize_angle(self.goal.yaw-pose.yaw))
                if stopped and heading_error > math.radians(self.config['goal']['heading_tolerance_deg']):
                    return self._finish_goal(pose, elapsed, now, grid)
                return super().step(now)
            if self.state == 'FINISHING' or self._goal_reached(pose):
                return self._finish_goal(pose, elapsed, now, grid)
            self._accept_global_plan(pose, grid)
            self._request_global_plan(pose, now)
            if self.global_path is None:
                return self._zero('waiting for first global route; ' + self.global_report)
            command = self.local_planner.local_command(
                grid, pose, self.global_path, self.control,
                self.last_linear, self.last_angular, elapsed,
                measured_linear=self.odom[3], measured_angular=self.odom[4])
            if command is None:
                return self._zero('no safe local motion; ' + self.global_report)
            return self._command(command, now, command.reason + '; ' + self.global_report)
