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

    def reset(self):
        super().reset()
        with self.lock:
            self.global_path = None
            self.pending_route = None
            self.global_updates = 0
            self.global_report = 'waiting for first global route'

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
        """Preserve the existing outside position, heading and stop contract."""
        cfg = self.control
        heading_error = normalize_angle(self.goal.yaw-pose.yaw)
        heading_ok = abs(heading_error) <= math.radians(self.config['goal']['heading_tolerance_deg'])
        measured_linear, measured_angular = self.odom[3:5]
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
                self.state = 'WAIT_LANE'
                self.completion_started = now
                self.lane_count = 0
                return self._zero('waiting for exit lane observation')
        if not self.local_planner.command_is_safe(
                grid, pose, command, cfg, measured_linear, measured_angular):
            return self._zero('outside goal stopping or rotation sweep is blocked')
        return self._command(command, now, reason)

    def step(self, now):
        with self.lock:
            # Entry registration and final handoff stay identical to V1.
            if self.state in ('ENTRY', 'WAIT_LANE', 'COMPLETE') or self.anchor is None:
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
            distance = math.hypot(self.goal.x-pose.x, self.goal.y-pose.y)
            if distance <= self.config['goal']['position_tolerance']:
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
