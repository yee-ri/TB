"""ROS-free port integration tests, not Gazebo or physical driving evidence.

The closed-loop fixture supplies ideal, finite LiDAR wall returns and integrates
the commanded differential-drive twist. It does not model wheel slip, sensor
noise, TF delay, camera recognition, or the actual TB course geometry.
"""

import copy
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes/stepbystep"))

from tunnel.mission import TunnelMission, load_config, relative_pose


CONFIG = Path(__file__).resolve().parents[1] / "codes/stepbystep/tunnel.yaml"


def configuration(goal=(.65, 0., 0.)):
    config = copy.deepcopy(load_config(CONFIG))
    config['goal']['pose'] = list(goal)
    config['grid']['bounds'] = [-.5, -.75, 1.5, 1.25]
    config['entry'].update(minimum_distance=.05, maximum_distance=.30,
                           sample_step=.02, turn_angle_deg=60.)
    return config


def transform(pose, origin):
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return (origin[0] + c*pose[0] - s*pose[1],
            origin[1] + s*pose[0] + c*pose[1], origin[2] + pose[2])


def integrate(pose, linear, angular, duration):
    """Independent exact ideal differential-drive motion, in metres/radians."""
    x, y, yaw = pose
    next_yaw = yaw + angular*duration
    if abs(angular) < 1e-10:
        return (x + linear*duration*math.cos(yaw),
                y + linear*duration*math.sin(yaw), next_yaw)
    return (x + linear/angular*(math.sin(next_yaw)-math.sin(yaw)),
            y - linear/angular*(math.cos(next_yaw)-math.cos(yaw)), next_yaw)


def wall_returns(pose, bounds=(-.4, -.65, 1.4, 1.1), bins=720,
                 sensor=(-.033073, 0., 0.)):
    """Raycast a rectangular room; returns are in robot base coordinates."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    origin = np.array((pose[0] + c*sensor[0] - s*sensor[1],
                       pose[1] + s*sensor[0] + c*sensor[1]))
    angle = -math.pi + (np.arange(bins) + .5)*(2.*math.pi/bins)
    direction = np.column_stack((np.cos(angle+pose[2]+sensor[2]),
                                 np.sin(angle+pose[2]+sensor[2])))
    x0, y0, x1, y1 = bounds
    distances = []
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        boundary = np.where(direction[:, axis] > 0., high, low)
        distances.append((boundary-origin[axis])/direction[:, axis])
    ranges = np.minimum(distances[0], distances[1])
    return np.column_stack((sensor[0] + ranges*np.cos(angle+sensor[2]),
                            sensor[1] + ranges*np.sin(angle+sensor[2])))


class TunnelMissionPortTest(unittest.TestCase):
    def make_mission(self, goal=(.65, 0., 0.)):
        mission = TunnelMission(configuration(goal))
        self.addCleanup(mission.close)
        return mission

    def feed(self, mission, now, pose=(0., 0., 0.), velocity=(0., 0.),
             origin=(0., 0., 0.), bounds=(-.4, -.65, 1.4, 1.1)):
        odom = transform(pose, origin)
        mission.update_odometry(odom, *velocity, now)
        mission.update_cloud(wall_returns(pose, bounds), (-.033073, 0., 0.), odom, now)

    def test_waits_for_both_sensors_and_does_not_initialize_from_missing_input(self):
        mission = self.make_mission()
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)
        mission.update_odometry((1., 2., .3), 0., 0., 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_default_goal_uses_source_gazebo_staging_to_exit_transform(self):
        config = load_config(CONFIG)
        staging = (-1.7475895, .140, -math.pi/2.)
        exit_pose = (.200, -1.748373, 0.)
        expected = relative_pose(exit_pose, staging)
        configured = config['goal']['pose']
        np.testing.assert_allclose(
            [configured[0], configured[1], math.radians(configured[2])],
            [expected.x, expected.y, expected.yaw], atol=1e-10)

    def test_anchor_coordinate_conversion_is_rigid_transform_invariant(self):
        origin = (4.3, -8.2, 1.2)
        relative = (.615, .53, math.radians(85.))
        result = relative_pose(transform(relative, origin), origin)
        np.testing.assert_allclose((result.x, result.y, result.yaw), relative, atol=1e-12)
        mission = self.make_mission()
        self.feed(mission, 10., origin=origin)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.assertEqual(mission.anchor, origin)

    def test_out_of_order_or_nonfinite_odometry_and_cloud_are_rejected(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        initial = mission.odom
        self.assertFalse(mission.update_odometry((100., 0., 0.), 0., 0., 9.))
        self.assertFalse(mission.update_odometry((math.nan, 0., 0.), 0., 0., 11.))
        self.assertEqual(mission.odom, initial)
        self.assertFalse(mission.update_cloud([[1., 0.]], (0., 0., 0.), (0., 0., 0.), 9.))
        self.assertFalse(mission.update_cloud([], (0., 0., 0.), (0., 0., 0.), 11.))
        self.assertEqual(mission.cloud[-1], 10.)

    def test_unobserved_space_cannot_authorize_entry(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_cloud([[1., 0.]], (0., 0., 0.), (0., 0., 0.), 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIsNone(mission.path)
        self.assertEqual(mission.plans, 0)

    def test_future_sensor_timestamp_cannot_authorize_motion(self):
        mission = self.make_mission()
        self.feed(mission, 10.5)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_clock_refresh_after_waiting_for_lock_rejects_expired_inputs(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.clock = lambda: 10.5
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_odometry_expiring_during_calculation_is_not_used_to_drive(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        readings = iter((10., 10.2))
        mission.clock = lambda: next(readings)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIn('expired during', mission.reason)
        self.assertIsNotNone(mission.path)

    def test_scan_expiring_during_calculation_is_not_used_to_drive(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.2)
        mission.control['safety_reaction_time'] = .5
        readings = iter((10.2, 10.35))
        mission.clock = lambda: next(readings)
        self.assertEqual(mission.step(10.2), (0., 0., False))
        self.assertIn('expired during', mission.reason)

    def test_initial_body_blind_region_does_not_erase_real_obstacle_return(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        cloud = np.concatenate((wall_returns((0., 0., 0.)), [[.055, .035]]))
        mission.update_cloud(cloud, (-.033073, 0., 0.), (0., 0., 0.), 10.05)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.05)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertFalse(mission.planner.pose_is_collision_free(mission.observed_grid, (0., 0., 0.)))

    def test_planner_failure_remains_stopped_without_repeated_search(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        with patch.object(mission.planner, 'plan', return_value=None) as search:
            self.assertEqual(mission.step(10.05), (0., 0., False))
            mission.worker.join(timeout=2.)
            self.assertEqual(mission.step(10.10), (0., 0., False))
            self.assertEqual(mission.state, 'FAILED')
            self.assertEqual(mission.step(10.15), (0., 0., False))
            self.assertEqual(search.call_count, 1)

    def test_planning_waits_for_measured_rest_not_only_zero_last_command(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        self.feed(mission, 10.05, velocity=(.05, .1))
        with patch.object(mission.planner, 'plan') as search:
            self.assertEqual(mission.step(10.05), (0., 0., False))
            self.assertIn('measured rest', mission.reason)
            search.assert_not_called()

    def test_reset_discards_result_from_superseded_planning_worker(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        started, release = threading.Event(), threading.Event()

        def slow_search(*args):
            started.set()
            release.wait(timeout=2.)
            return None

        with patch.object(mission.planner, 'plan', side_effect=slow_search):
            mission.step(10.05)
            self.assertTrue(started.wait(timeout=1.))
            try:
                mission.reset()
            finally:
                release.set()
                mission.worker.join(timeout=2.)
            self.assertIsNone(mission.plan_result)
            self.assertEqual(mission.state, 'ENTRY')

    def test_lane_handoff_requires_new_consecutive_visible_frames_at_goal(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.completion_started, mission.lane_count = 'WAIT_LANE', 10., 0
        self.assertEqual(mission.step(10.), (0., 0., False))
        mission.update_lane(True, 10.05)
        mission.update_lane(False, 10.10)
        mission.update_lane(True, 10.15)
        mission.update_lane(True, 10.20)
        self.assertEqual(mission.step(10.20), (0., 0., False))
        mission.update_lane(True, 10.25)
        self.assertEqual(mission.step(10.25), (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')

    def test_stale_lane_confirmation_does_not_hand_off(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.completion_started, mission.lane_count = 'WAIT_LANE', 10., 0
        for stamp in (10.05, 10.1, 10.15):
            mission.update_lane(True, stamp)
        self.feed(mission, 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_LANE')

    def test_stale_input_stops_without_discarding_valid_route(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        path = mission.path
        mission.update_odometry((0., 0., 0.), 0., 0., 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertIs(mission.path, path)
        self.assertEqual(mission.plans, 0)
        self.feed(mission, 10.55)
        self.assertGreater(mission.step(10.55)[0], 0.)
        self.assertIs(mission.path, path)

    def test_harmless_new_scan_keeps_route_and_positive_command(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        path, version = mission.path, mission.version
        for tick in range(1, 4):
            now = 10. + tick*.05
            self.feed(mission, now)
            self.assertGreater(mission.step(now)[0], 0.)
            self.assertIs(mission.path, path)
        self.assertGreater(mission.version, version)
        self.assertEqual(mission.plans, 0)

    def test_new_wall_on_entry_route_stops_without_completing(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.feed(mission, 10.05, bounds=(-.4, -.65, .20, 1.1))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIn('obstacle', mission.reason)

    def test_reset_keeps_sensor_input_but_reanchors_and_drops_old_route(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        old_path = mission.path
        self.feed(mission, 10.05, pose=(.02, 0., 0.))
        mission.reset()
        self.assertIsNone(mission.path)
        self.assertIsNone(mission.anchor)
        self.assertIsNotNone(mission.odom)
        self.assertIsNotNone(mission.cloud)
        mission.step(10.05)
        self.assertEqual(mission.anchor, (.02, 0., 0.))
        self.assertIsNot(mission.path, old_path)

    def run_ideal_mission(self, goal, origin=(0., 0., 0.)):
        mission = self.make_mission(goal)
        pose, command, now = (0., 0., 0.), (0., 0.), 10.
        states = []
        lane_frames = 0
        deadline = time.monotonic() + 45.
        for _ in range(1000):
            self.assertLess(time.monotonic(), deadline, mission.reason)
            self.feed(mission, now, pose, command, origin)
            if mission.state == 'WAIT_LANE':
                if lane_frames == 0:
                    self.assertFalse(mission.step(now)[2])
                mission.update_lane(True, now)
                lane_frames += 1
            linear, angular, finished = mission.step(now)
            if not states or states[-1] != mission.state:
                states.append(mission.state)
            self.assertLessEqual(abs(linear), mission.control['cruise_velocity'] + 1e-9)
            self.assertLessEqual(abs(angular), mission.control['maximum_angular_velocity'] + 1e-9)
            self.assertNotEqual(mission.state, 'FAILED', mission.reason)
            if mission.worker is not None and mission.worker.is_alive():
                mission.worker.join(timeout=5.)
            if finished:
                self.assertEqual((linear, angular), (0., 0.))
                break
            command = (linear, angular)
            pose = integrate(pose, linear, angular, .05)
            now += .05
        self.assertEqual(mission.state, 'COMPLETE', (mission.state, mission.reason, pose))
        self.assertEqual(states, ['ENTRY', 'PLANNING', 'FOLLOWING', 'WAIT_LANE', 'COMPLETE'])
        self.assertEqual(mission.plans, 1)
        self.assertGreaterEqual(lane_frames, mission.sensors['lane_confirmation_frames'])
        self.assertLessEqual(math.hypot(pose[0]-goal[0], pose[1]-goal[1]),
                             mission.config['goal']['position_tolerance'])

    def test_ideal_lidar_and_differential_drive_complete_straight_mission(self):
        self.run_ideal_mission((.65, 0., 0.), origin=(2.2, -3.1, -.8))

    def test_ideal_lidar_and_differential_drive_complete_turning_mission(self):
        self.run_ideal_mission((.615, .53, 85.))


if __name__ == '__main__':
    unittest.main()
