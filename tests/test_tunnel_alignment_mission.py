"""ROS-free tests of initial alignment and its mission lifecycle.

Lifecycle cases substitute a wall fit to isolate acquisition-frame handling.
The closed loops use independently raycast fixed walls and ideal odometry;
they are not evidence of physical sensor precision or braking performance.
"""

import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))

from tunnel.localization import AlignmentResult
from collision_inspection import inspect_geometry
from tunnel.mission import load_config, relative_pose
from tunnel.mission import TunnelMission
from tunnel.geometry import Pose2D


CONFIG = Path(__file__).resolve().parents[1] / 'codes/stepbystep/tunnel.real.yaml'
SENSOR = (-.033073, 0., 0.)


def compose(pose, frame):
    c, s = math.cos(frame[2]), math.sin(frame[2])
    return (frame[0] + c*pose[0] - s*pose[1],
            frame[1] + s*pose[0] + c*pose[1], frame[2] + pose[2])


def integrate(pose, linear, angular, duration):
    x, y, yaw = pose
    next_yaw = yaw + angular*duration
    if abs(angular) < 1e-10:
        return (x + linear*duration*math.cos(yaw),
                y + linear*duration*math.sin(yaw), next_yaw)
    return (x + linear/angular*(math.sin(next_yaw)-math.sin(yaw)),
            y - linear/angular*(math.cos(next_yaw)-math.cos(yaw)), next_yaw)


def fixed_wall_cloud(pose, rectangles):
    """Independent continuous ray/rectangle intersections, no moving walls."""
    origin = np.asarray(compose(SENSOR, pose)[:2])
    angles = -math.pi + (np.arange(720)+.5)*(2*math.pi/720)
    directions = np.column_stack((np.cos(angles+pose[2]), np.sin(angles+pose[2])))
    ranges = np.full(len(angles), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for rectangle in rectangles.values():
            low, high = np.asarray(rectangle[:2]), np.asarray(rectangle[2:])
            first, second = (low-origin)/directions, (high-origin)/directions
            near = np.max(np.minimum(first, second), axis=1)
            far = np.min(np.maximum(first, second), axis=1)
            hit = (far >= np.maximum(near, 0.)) & (near >= .05)
            ranges[hit] = np.minimum(ranges[hit], near[hit])
    valid = np.isfinite(ranges) & (ranges <= 3.)
    return np.column_stack((SENSOR[0]+ranges[valid]*np.cos(angles[valid]),
                            SENSOR[1]+ranges[valid]*np.sin(angles[valid])))


class TunnelAlignmentMissionTest(unittest.TestCase):
    def mission(self, fitted=(.04, 0., 0.), observation_window=0.):
        config = load_config(CONFIG)
        # These lifecycle tests supply fits directly. Separate cases below
        # enable accumulation and test its independent timing/frame behavior.
        config['localization']['observation_window'] = observation_window
        # Moving localization has separate lifecycle tests; these fixtures
        # deliberately supply the initial fit, independently of raw geometry.
        config['localization']['tracking_period'] = 0.
        mission = TunnelMission(config)
        mission.entrance_localizer.estimate = Mock(return_value=AlignmentResult(
            True, Pose2D(*fitted), 'fixture fit', .001, 120, 140))
        self.addCleanup(mission.close)
        return mission

    def feed(self, mission, now, pose=(.04, 0., 0.), frame=(0., 0., 0.),
             velocity=(0., 0.), latest=None):
        odom = compose(pose, frame)
        mission.update_odometry(compose(pose if latest is None else latest, frame),
                                *velocity, now)
        mission.update_cloud(fixed_wall_cloud(pose, mission.config['localization']['wall_rectangles']),
                             SENSOR, odom, now)

    def join(self, mission):
        for worker in (mission.localization_worker, mission.worker):
            if worker is None:
                continue
            # The fixture advances ideal simulated time only after the
            # asynchronous wall fit or global route worker has finished.
            worker.join(timeout=10.)
            self.assertFalse(worker.is_alive(), 'offline wall fit/planning worker did not finish')

    def assert_entry_motion(self, mission, now):
        linear, angular, complete = mission.step(now)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertGreater(abs(linear)+abs(angular), 0.)
        self.assertFalse(complete)

    def initialize_from_first_fit(self, mission, pose=(.04, 0., 0.),
                                  frame=(0., 0., 0.), now=10., latest=None):
        self.feed(mission, now, pose, frame, latest=latest)
        self.assertEqual(mission.step(now), (0., 0., False))
        self.join(mission)
        now += .01  # Advance the control clock so acceleration has a nonzero interval.
        self.assert_entry_motion(mission, now)
        self.assertIsNotNone(mission.anchor)
        self.assertEqual(mission.state, 'ENTRY')
        mission.entrance_localizer.estimate.assert_called_once()
        return now

    def test_reference_140mm_and_100mm_share_the_same_fixed_map(self):
        for start_x in (0., .04, .09, -.06):
            with self.subTest(axle_gap_mm=round((.14-start_x)*1000)):
                mission = self.mission((start_x, 0., 0.))
                self.initialize_from_first_fit(mission, (start_x, 0., 0.))
                np.testing.assert_allclose(mission.anchor, (0., 0., 0.), atol=1e-12)
                self.assertAlmostEqual(mission.entry_start_x, start_x)
                self.assertEqual(mission.state, 'ENTRY')
                np.testing.assert_allclose(compose((mission.goal.x, mission.goal.y, mission.goal.yaw),
                                                   mission.anchor), (1.808, 1.868, math.pi/2))

    def test_anchor_uses_acquisition_odometry_not_latest_odometry(self):
        pose, frame = (.04, .02, .03), (2., -1., .7)
        mission = self.mission(pose)
        latest = (.043, .02, .03)
        self.initialize_from_first_fit(mission, pose, frame, latest=latest)
        np.testing.assert_allclose(mission.anchor, frame, atol=1e-12)
        localized = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((localized.x, localized.y, localized.yaw), latest, atol=1e-12)

    def test_first_fit_is_used_without_averaging_in_far_rotated_odom_frame(self):
        actual = (.04, .02, .03)
        observations = ((.042, .019, .033), (.038, .021, .027), actual)
        for frame in ((0., 0., 0.), (120., -87., 3.139)):
            with self.subTest(frame=frame):
                mission = self.mission(actual)
                mission.entrance_localizer.estimate.side_effect = [
                    AlignmentResult(True, Pose2D(*p), 'fixture fit', .001, 120, 140)
                    for p in observations]
                self.initialize_from_first_fit(mission, actual, frame)
                localized = relative_pose(mission.odom[:3], mission.anchor)
                np.testing.assert_allclose((localized.x, localized.y, localized.yaw),
                                           observations[0], atol=1e-10)

    def test_observation_window_transforms_all_hits_to_the_latest_base_frame(self):
        mission = self.mission(observation_window=1.)
        frame = (15., -8., .8)
        measured_world = np.array(((.7, .1), (1.1, .3), (1.5, .5)))
        for index in range(11):
            now = 10. + .1*index
            pose = (.04 + .002*index, -.01, .002*index)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            measured_base = (measured_world - pose[:2]) @ np.array(((c, -s), (s, c)))
            odom = compose(pose, frame)
            mission.update_odometry(odom, 0., 0., now)
            mission.update_cloud(((1., 0.),), SENSOR, odom, now,
                                 localization_points=measured_base)
            self.assertEqual(mission.step(now), (0., 0., False))
            if index < 10:
                mission.entrance_localizer.estimate.assert_not_called()
        self.join(mission)
        mission.entrance_localizer.estimate.assert_called_once()
        self.assertEqual(mission.entrance_localizer.estimate.call_args.args[1], SENSOR[:2])
        merged = mission.entrance_localizer.estimate.call_args.args[0]
        self.assertEqual(merged.shape, (33, 2))
        # Every scan saw these same physical points. After transforming each
        # acquisition, all eleven copies agree in the final base frame.
        np.testing.assert_allclose(merged, np.tile(measured_base, (11, 1)), atol=1e-12)

    def test_observation_window_uses_control_time_with_irregular_scan_arrival(self):
        mission = self.mission(observation_window=1.)
        for now in (10., 10.12, 10.41, 10.97):
            self.feed(mission, now)
            self.assertEqual(mission.step(now), (0., 0., False))
            mission.entrance_localizer.estimate.assert_not_called()
        # No scan arrives exactly at the observation deadline.
        self.assertEqual(mission.step(11.), (0., 0., False))
        self.join(mission)
        mission.entrance_localizer.estimate.assert_called_once()
        self.assert_entry_motion(mission, 11.01)

    def test_one_unchanged_scan_cannot_restart_the_observation_window(self):
        mission = self.mission(observation_window=1.)
        self.feed(mission, 10.)
        for now in (10., 10.2, 10.8):
            self.assertEqual(mission.step(now), (0., 0., False))
        self.assert_entry_motion(mission, 11.)
        mission.entrance_localizer.estimate.assert_not_called()
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIn('observations expired', mission.localization_report)
        self.assertIsNotNone(mission.anchor)

    def test_window_fit_rejection_reason_remains_visible(self):
        mission = self.mission(observation_window=1.)
        mission.entrance_localizer.estimate.return_value = AlignmentResult(
            False, reason='right wall returns missing')
        for index in range(11):
            now = 10. + .1*index
            self.feed(mission, now)
            self.assertEqual(mission.step(now), (0., 0., False))
        self.join(mission)
        self.assert_entry_motion(mission, now+.01)
        self.assertIn('right wall returns missing', mission.localization_report)
        self.assertIsNotNone(mission.anchor)
        self.assertEqual(mission.state, 'ENTRY')

    def test_localization_only_points_do_not_change_the_collision_scan(self):
        mission = self.mission()
        now, odom = 10., (.04, 0., 0.)
        collision_points = np.array(((1., 0.),))
        wall_points = np.array(((.12, 0.), (.4, .2), (.6, -.2)))
        mission.update_odometry(odom, 0., 0., now)
        mission.update_cloud(collision_points, SENSOR, odom, now,
                             localization_points=wall_points)
        self.assertEqual(mission.step(now), (0., 0., False))
        self.join(mission)
        np.testing.assert_allclose(mission.entrance_localizer.estimate.call_args.args[0], wall_points)
        self.assertEqual(np.count_nonzero(np.isfinite(mission.scan[0])), 1)
        # The additional close wall-only point must neither trigger entry
        # stopping nor become a projected obstacle in the collision map.
        self.assertGreater(mission.front_clearance, .9)

    def test_localization_preserves_straight_entry_and_the_activation_heading(self):
        for pose in ((.04, .02, 0.), (.04, 0., math.radians(5.))):
            with self.subTest(pose=pose):
                mission = self.mission(pose)
                mission.entry_heading_offset = 0.  # Explicit legacy straight mode.
                mission.entry_phase = 'DRIVE'
                now = self.initialize_from_first_fit(mission, pose)
                self.assertEqual(mission.state, 'ENTRY')
                self.assertEqual(mission.plans, 0)
                self.feed(mission, now+.1, pose)
                linear, angular, finished = mission.step(now+.1)
                self.assertGreater(linear, 0.)
                self.assertAlmostEqual(angular, 0., places=12)
                self.assertFalse(finished)

    def test_reusing_one_scan_does_not_refit_the_accepted_initial_pose(self):
        mission = self.mission()
        now = self.initialize_from_first_fit(mission)
        anchor = mission.anchor
        for index in range(12):
            self.assert_entry_motion(mission, now+.01*(index+1))
        self.assertEqual(mission.anchor, anchor)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertEqual(mission.mapped_stamp, 10.)
        self.assertEqual(mission.entrance_localizer.estimate.call_count, 1)

    def test_rejected_fit_uses_prior_once_and_allows_entry_motion(self):
        mission = self.mission()
        mission.entrance_localizer.estimate.return_value = AlignmentResult(
            False, reason='right wall direction is undetermined')
        self.feed(mission, 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.join(mission)
        self.assert_entry_motion(mission, 10.01)
        self.assertIsNotNone(mission.anchor)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIn('right wall direction is undetermined', mission.localization_report)
        pose = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), mission.entrance_localizer.prior)
        self.feed(mission, 10.1)
        linear, angular, finished = mission.step(10.1)
        self.assertGreater(abs(linear)+abs(angular), 0.)
        self.assertFalse(finished)
        mission.entrance_localizer.estimate.assert_called_once()

    def test_nonzero_reported_velocity_allows_first_real_wall_fit_after_observation_window(self):
        pose = (.226, -.016, math.radians(1.68))
        frame = (2., -1., .7)
        for velocity in ((.03, 0.), (0., .1), (.03, -.1)):
            with self.subTest(velocity=velocity):
                config = load_config(CONFIG)
                config['localization']['tracking_period'] = 0.
                mission = TunnelMission(config)
                self.addCleanup(mission.close)
                self.assertEqual(config['localization']['observation_window'], 1.)
                # The scan and acquired poses stay fixed; only reported twist
                # is nonzero. The first real fit must initialize the map.
                results = []
                for index in range(11):
                    now = 10. + .1*index
                    self.feed(mission, now, pose, frame, velocity=velocity)
                    self.assertEqual(mission.step(now), (0., 0., False))
                    self.join(mission)
                    if index < 10:
                        self.assertIsNone(mission.localization_worker)
                        self.assertIsNone(mission.anchor)
                    else:
                        self.assertIsNotNone(mission.localization_result, mission.reason)
                        results.append(mission.localization_result[0])
                        self.assertTrue(results[-1].accepted)
                    if index < 10:
                        self.assertEqual(mission.step(now), (0., 0., False))
                        self.assertEqual(mission.state, 'LOCALIZING')
                    else:
                        self.assert_entry_motion(mission, now+.01)
                self.assertEqual(len(results), 1)
                self.assertEqual(mission.state, 'ENTRY')
                self.assertIsNotNone(mission.anchor)
                localized = relative_pose(mission.odom[:3], mission.anchor)
                np.testing.assert_allclose((localized.x, localized.y, localized.yaw),
                                           pose, atol=.003)
                self.assertEqual(mission.mapped_stamp, now)

    def test_future_inputs_cannot_start_alignment(self):
        mission = self.mission()
        self.feed(mission, 10.)
        self.assertEqual(mission.step(9.9), (0., 0., False))
        self.assertIn('future', mission.reason)
        mission.entrance_localizer.estimate.assert_not_called()

    def test_worker_result_expiring_before_acceptance_uses_prior(self):
        mission = self.mission()
        self.feed(mission, 10.)
        mission.step(10.)
        self.join(mission)
        self.feed(mission, 10.4)
        self.assert_entry_motion(mission, 10.4)
        self.assertIsNotNone(mission.anchor)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIn('expired', mission.localization_report)
        pose = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((pose.x, pose.y, pose.yaw), mission.entrance_localizer.prior)
        mission.entrance_localizer.estimate.assert_called_once()

    def test_prior_preserves_motion_since_activation_in_rotated_odom_frame(self):
        first = (.04, .02, .03)
        frame = (120., -87., 2.7)
        for failure in ('rejected', 'expired'):
            with self.subTest(failure=failure):
                mission = self.mission(observation_window=1.)
                if failure == 'rejected':
                    mission.entrance_localizer.estimate.return_value = AlignmentResult(
                        False, reason='right wall returns missing')
                # The configured prior belongs to activation, not an older scan.
                activation = (.045, .021, .04)
                self.feed(mission, 10., first, frame, latest=activation)
                mission.step(10.)
                latest = (.07, .025, .09)
                self.feed(mission, 11., (.06, .024, .06), frame, latest=latest)
                mission.step(11.)
                self.join(mission)
                now = 11. if failure == 'rejected' else 11.4
                mission.update_odometry(compose(latest, frame), 0., 0., now)
                mission.step(now)
                self.assertEqual(mission.state, 'ENTRY')
                relative = relative_pose(latest, activation)
                expected = compose((relative.x, relative.y, relative.yaw),
                                   mission.entrance_localizer.prior)
                localized = relative_pose(mission.odom[:3], mission.anchor)
                np.testing.assert_allclose((localized.x, localized.y, localized.yaw),
                                           expected, atol=1e-12)
                self.assertAlmostEqual(mission.entry_start_x, expected[0])
                self.assertAlmostEqual(mission.entry_start_yaw, expected[2])
                mission.entrance_localizer.estimate.assert_called_once()

    def test_slow_fit_cannot_block_entry_or_global_planning_or_overwrite_the_map(self):
        mission = self.mission()
        started, release = threading.Event(), threading.Event()
        def delayed(_points, _sensor_origin):
            started.set()
            if not release.wait(timeout=3.):
                raise RuntimeError('test wall fit was never released')
            return AlignmentResult(True, Pose2D(.9, .4, .8), 'late fit', .001, 120, 140)
        mission.entrance_localizer.estimate.side_effect = delayed
        self.feed(mission, 10.)
        mission.step(10.)
        self.assertTrue(started.wait(timeout=1.))
        worker = mission.localization_worker
        try:
            deadline = 10.+mission.sensors['maximum_age']+.001
            self.feed(mission, deadline, pose=(.06, 0., .02))
            self.assert_entry_motion(mission, deadline)
            self.assertIn('exceeded', mission.localization_report)
            self.assertTrue(worker.is_alive())
            anchor, costmap, mapped_stamp = mission.anchor, mission.costmap, mission.mapped_stamp
            occupancy = list(costmap.to_occupancy_data())
            mission.local_planner.global_route = Mock(return_value=None)
            pose = relative_pose(mission.odom[:3], mission.anchor)
            mission._request_global_plan(pose, deadline)
            self.assertIsNotNone(mission.worker)
            self.assertIsNot(mission.worker, worker)
            mission.worker.join(timeout=3.)
            self.assertFalse(mission.worker.is_alive())
            mission.local_planner.global_route.assert_called_once()
            self.assertIsNotNone(mission.pending_route)
        finally:
            release.set()
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive())
        self.assertIsNone(mission.localization_result)
        self.assertEqual(mission.anchor, anchor)
        self.assertIs(mission.costmap, costmap)
        self.assertEqual(mission.mapped_stamp, mapped_stamp)
        self.assertEqual(list(mission.costmap.to_occupancy_data()), occupancy)
        self.assert_entry_motion(mission, deadline)
        mission.entrance_localizer.estimate.assert_called_once()

    def test_missing_initial_observations_use_prior_after_the_existing_window(self):
        mission = self.mission(observation_window=1.)
        self.feed(mission, 10.)
        mission.localization_scan = None
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assert_entry_motion(mission, 11.)
        self.assertIn('observations missing', mission.localization_report)
        mission.entrance_localizer.estimate.assert_not_called()

    def test_movement_since_acquisition_is_propagated_from_an_accepted_fit(self):
        for pose in ((.05, 0., 0.), (.04, 0., math.radians(1.))):
            with self.subTest(latest_pose=pose):
                mission = self.mission()
                self.feed(mission, 10.)
                mission.step(10.)
                self.join(mission)
                mission.update_odometry(pose, 0., 0., 10.05)
                self.assert_entry_motion(mission, 10.05)
                self.assertIsNotNone(mission.anchor)
                self.assertEqual(mission.state, 'ENTRY')
                localized = relative_pose(mission.odom[:3], mission.anchor)
                np.testing.assert_allclose((localized.x, localized.y, localized.yaw), pose,
                                           atol=1e-12)

    def test_reset_discards_a_worker_from_the_previous_activation(self):
        mission = self.mission()
        started, release = threading.Event(), threading.Event()
        accepted = mission.entrance_localizer.estimate.return_value
        def delayed(_points, _sensor_origin):
            started.set()
            if not release.wait(timeout=3.):
                raise RuntimeError('test wall fit was never released')
            return accepted
        mission.entrance_localizer.estimate.side_effect = delayed
        self.feed(mission, 10.)
        mission.step(10.)
        self.assertTrue(started.wait(timeout=1.))
        worker = mission.localization_worker
        try:
            # A running fit cannot hold the mission lock and block callbacks.
            self.assertTrue(mission.update_odometry((.04, 0., 0.), 0., 0., 10.01))
            mission.reset()
        finally:
            release.set()
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive())
        self.assertEqual(mission.state, 'LOCALIZING')
        self.assertIsNone(mission.anchor)
        self.assertIsNone(mission.localization_result)

    def test_reported_velocity_during_a_fit_does_not_discard_fixed_pose_result(self):
        mission = self.mission()
        started, release = threading.Event(), threading.Event()
        accepted = mission.entrance_localizer.estimate.return_value
        def delayed(_points, _sensor_origin):
            started.set()
            if not release.wait(timeout=3.):
                raise RuntimeError('test wall fit was never released')
            return accepted
        mission.entrance_localizer.estimate.side_effect = delayed
        self.feed(mission, 10.)
        mission.step(10.)
        self.assertTrue(started.wait(timeout=1.))
        epoch = mission.localization_epoch
        try:
            mission.update_odometry((.04, 0., 0.), .02, .1, 10.01)
            self.assertEqual(mission.step(10.01), (0., 0., False))
            self.assertEqual(mission.localization_epoch, epoch)
            mission.update_odometry((.04, 0., 0.), 0., 0., 10.02)
        finally:
            release.set()
            self.join(mission)
        self.assert_entry_motion(mission, 10.02)
        self.assertIsNotNone(mission.anchor)
        self.assertIsNone(mission.localization_result)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertGreater(mission.localization_epoch, epoch)
        mission.entrance_localizer.estimate.assert_called_once()

    def test_expected_fit_exceptions_are_reported_and_allow_entry_without_retry(self):
        for error in (ValueError, RuntimeError, np.linalg.LinAlgError):
            with self.subTest(error=error.__name__):
                mission = self.mission()
                mission.entrance_localizer.estimate.side_effect = error('invalid fit fixture')
                self.feed(mission, 10.)
                mission.step(10.)
                self.join(mission)
                self.assert_entry_motion(mission, 10.01)
                self.assertIn('wall alignment failed', mission.localization_report)
                self.assertIsNotNone(mission.anchor)
                self.feed(mission, 10.1)
                self.assert_entry_motion(mission, 10.1)
                mission.entrance_localizer.estimate.assert_called_once()

    def test_alignment_allows_entry_before_second_mapped_scan(self):
        mission = self.mission()
        now = self.initialize_from_first_fit(mission)
        self.assertEqual(mission.mapped_stamp, now-.01)
        self.assertEqual(mission.config['grid']['mark_observations'], 2)
        self.assert_entry_motion(mission, now)
        self.feed(mission, now+.1)
        linear, angular, finished = mission.step(now+.1)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)  # The first allowed motion is the heading turn.
        self.assertFalse(finished)

    def test_map_overlap_at_localized_pose_does_not_block_entry_turn(self):
        pose = (.14, .20, 0.)
        mission = self.mission(pose)
        now = self.initialize_from_first_fit(mission, pose)
        self.feed(mission, now+.1, pose)
        self.assertFalse(inspect_geometry(mission.geometry).pose_is_collision_free(mission._grid(), Pose2D(*pose)))
        linear, angular, complete = mission.step(now+.1)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)
        self.assertFalse(complete)
        self.assertEqual(mission.plans, 0)

    def test_recorded_stop_pose_uses_actual_body_without_added_padding(self):
        pose = (.511643, -.003789, math.radians(26.334))
        mission = self.mission(pose)
        now = self.initialize_from_first_fit(mission, pose)
        self.feed(mission, now + .1, pose)
        grid = mission._grid()
        self.assertTrue(inspect_geometry(mission.geometry).pose_is_collision_free(grid, Pose2D(*pose)))
        self.assertEqual(mission.footprint.padding, 0.)
        linear, angular, _ = mission.step(now + .1)
        self.assertGreater(abs(linear) + abs(angular), 0.)

        # An empty corner of the outer rectangle no longer stops motion.
        tight = (pose[0], pose[1] - .015, pose[2])
        envelope_gap = (tight[1] - mission.footprint.rear * math.sin(tight[2])
                        - mission.footprint.half_width * math.cos(tight[2]) + .150)
        self.assertLess(envelope_gap, 0.)
        self.assertTrue(inspect_geometry(mission.geometry).pose_is_collision_free(grid, Pose2D(*tight)))
        self.feed(mission, now + .2, tight)
        linear, angular, _ = mission.step(now + .2)
        self.assertGreater(abs(linear) + abs(angular), 0.)

        # Move a real part 1 mm through the surveyed right wall surface.
        wall_y = mission.config['localization']['wall_rectangles']['right_side_wall'][3]
        lowest_part = min(x * math.sin(pose[2]) + y * math.cos(pose[2])
                          for x0, y0, x1, y1 in mission.footprint.parts
                          for x in (x0, x1) for y in (y0, y1))
        overlapping = (pose[0], wall_y - lowest_part - .001, pose[2])
        self.assertFalse(inspect_geometry(mission.geometry).pose_is_collision_free(grid, Pose2D(*overlapping)))
        self.feed(mission, now + .3, overlapping)
        linear, angular, _ = mission.step(now + .3)
        self.assertGreater(abs(linear) + abs(angular), 0.)

    def test_entry_endpoint_is_fixed_while_reported_travel_adapts(self):
        mission = self.mission()
        now = self.initialize_from_first_fit(mission)
        # Isolate the straight leg after the entry heading turn.
        mission.entry_phase, mission.entry_drive_yaw = 'DRIVE', 0.
        self.feed(mission, now+.1, (.911, 0., 0.))
        self.assertEqual(mission.step(now+.1), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertIn('distance target', mission.reason)
        self.assertIn('traveled=0.871m', mission.reason)

    def test_fixed_wall_ideal_closed_loops_complete_from_both_staging_distances(self):
        for start_x in (0., .04):
            with self.subTest(axle_gap_mm=round((.14-start_x)*1000)):
                pose = (start_x, 0., 0.)
                mission = self.mission(pose)
                now = self.initialize_from_first_fit(mission, pose)
                previous = (0., 0.)
                for index in range(800):
                    now += .05
                    self.feed(mission, now, pose, velocity=previous)
                    linear, angular, finished = mission.step(now)
                    # Only wall-clock search is awaited. This ideal loop has
                    # no ROS transport or physical actuator timing model.
                    self.join(mission)
                    localized = relative_pose(pose, mission.anchor)
                    self.assertTrue(inspect_geometry(mission.geometry).pose_is_collision_free(mission._grid(), localized),
                                    (pose, mission.state, mission.reason))
                    if finished:
                        self.assertEqual((linear, angular), (0., 0.))
                        break
                    pose = integrate(pose, linear, angular, .05)
                    previous = (linear, angular)
                else:
                    self.fail('ideal mission failed to reach exit: %s %s' % (mission.state, mission.reason))
                # The real-profile opening is horizontal. Check the complete
                # physical rectangle independently of the controller helper.
                exit_y = mission.config['goal']['exit_crossing'][0][1]
                corner_y = [pose[1] + x*math.sin(pose[2]) + y*math.cos(pose[2])
                            for x in (-mission.footprint.rear, mission.footprint.front)
                            for y in (-mission.footprint.half_width, mission.footprint.half_width)]
                self.assertGreaterEqual(min(corner_y), exit_y)
                self.assertTrue(mission.exit_crossed)
                self.assertGreaterEqual(mission.global_updates, 1)
                self.assertGreater(previous[0], 0.)
                # Portal completion returns ownership while measured motion
                # remains. No lane sample or measured-stop sample is supplied.
                self.assertEqual(mission.step(now), (0., 0., True))
                self.assertEqual(mission.state, 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
