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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))

from tunnel.localization import AlignmentResult
from tunnel.mission import TunnelMission, load_config, relative_pose
from tunnel.planner import OccupancyGrid, Pose2D


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
        mission = TunnelMission(config)
        mission.localizer.estimate = Mock(return_value=AlignmentResult(
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
        worker = mission.worker
        if worker is not None:
            # This fixture checks geometric completion, not a 5 s CPU budget.
            # Exact thin-wall planning can take just over 5 s on this host;
            # live planning is asynchronous and has no such deadline.
            worker.join(timeout=10.)
            self.assertFalse(worker.is_alive(), 'offline wall fit/planning worker did not finish')

    def confirm(self, mission, pose=(.04, 0., 0.), frame=(0., 0., 0.),
                start=10., latest=None):
        for index in range(mission.config['localization']['confirmation_scans']):
            now = start + index*.1
            self.feed(mission, now, pose, frame, latest=latest)
            self.assertEqual(mission.step(now), (0., 0., False))
            self.join(mission)
            self.assertEqual(mission.step(now), (0., 0., False))
        self.assertIsNotNone(mission.anchor)
        return now

    def test_reference_140mm_and_100mm_share_the_same_fixed_map(self):
        for start_x in (0., .04, .09, -.06):
            with self.subTest(axle_gap_mm=round((.14-start_x)*1000)):
                mission = self.mission((start_x, 0., 0.))
                self.confirm(mission, (start_x, 0., 0.))
                np.testing.assert_allclose(mission.anchor, (0., 0., 0.), atol=1e-12)
                self.assertAlmostEqual(mission.entry_start_x, start_x)
                self.assertEqual(mission.state, 'ENTRY')
                np.testing.assert_allclose(compose((mission.goal.x, mission.goal.y, mission.goal.yaw),
                                                   mission.anchor), (1.808, 1.868, math.pi/2))

    def test_anchor_uses_acquisition_odometry_not_latest_odometry(self):
        pose, frame = (.04, .02, .03), (2., -1., .7)
        mission = self.mission(pose)
        latest = (.043, .02, .03)
        self.confirm(mission, pose, frame, latest=latest)
        np.testing.assert_allclose(mission.anchor, frame, atol=1e-12)
        localized = relative_pose(mission.odom[:3], mission.anchor)
        np.testing.assert_allclose((localized.x, localized.y, localized.yaw), latest, atol=1e-12)

    def test_confirmation_and_average_are_invariant_to_far_rotated_odom_origin(self):
        actual = (.04, .02, .03)
        observations = ((.042, .019, .033), (.038, .021, .027), actual)
        for frame in ((0., 0., 0.), (120., -87., 3.139)):
            with self.subTest(frame=frame):
                mission = self.mission(actual)
                mission.localizer.estimate.side_effect = [
                    AlignmentResult(True, Pose2D(*p), 'fixture fit', .001, 120, 140)
                    for p in observations]
                self.confirm(mission, actual, frame)
                localized = relative_pose(mission.odom[:3], mission.anchor)
                np.testing.assert_allclose((localized.x, localized.y, localized.yaw), actual, atol=1e-10)

    def test_observation_window_transforms_all_hits_to_the_latest_base_frame(self):
        mission = self.mission(observation_window=1.)
        frame = (15., -8., .8)
        measured_world = np.array(((.7, .1), (1.1, .3), (1.5, .5)))
        for index in range(10):
            now = 10. + .1*index
            pose = (.04 + .0001*index, -.01, .0002*index)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            measured_base = (measured_world - pose[:2]) @ np.array(((c, -s), (s, c)))
            odom = compose(pose, frame)
            mission.update_odometry(odom, 0., 0., now)
            mission.update_cloud(((1., 0.),), SENSOR, odom, now,
                                 localization_points=measured_base)
            self.assertEqual(mission.step(now), (0., 0., False))
            if index < 9:
                mission.localizer.estimate.assert_not_called()
        self.join(mission)
        mission.localizer.estimate.assert_called_once()
        merged = mission.localizer.estimate.call_args.args[0]
        self.assertEqual(merged.shape, (30, 2))
        # Every scan saw these same physical points. After transforming each
        # acquisition, all ten copies agree in the final base frame.
        np.testing.assert_allclose(merged, np.tile(measured_base, (10, 1)), atol=1e-12)

    def test_observation_window_restarts_after_motion_or_stale_input(self):
        for interruption in ('motion', 'stale'):
            with self.subTest(interruption=interruption):
                mission = self.mission(observation_window=1.)
                for index in range(8):
                    now = 10. + .1*index
                    self.feed(mission, now)
                    self.assertEqual(mission.step(now), (0., 0., False))
                mission.localizer.estimate.assert_not_called()
                if interruption == 'motion':
                    self.feed(mission, 10.8, velocity=(.02, 0.))
                    self.assertEqual(mission.step(10.8), (0., 0., False))
                else:
                    self.assertEqual(mission.step(11.1), (0., 0., False))
                for index in range(10):
                    now = 11.2 + .1*index
                    self.feed(mission, now)
                    self.assertEqual(mission.step(now), (0., 0., False))
                    if index < 9:
                        mission.localizer.estimate.assert_not_called()
                self.join(mission)
                mission.localizer.estimate.assert_called_once()
                self.assertIsNone(mission.anchor)

    def test_window_fit_rejection_reason_remains_visible(self):
        mission = self.mission(observation_window=1.)
        mission.localizer.estimate.return_value = AlignmentResult(
            False, reason='no extended perpendicular wall support')
        for index in range(10):
            now = 10. + .1*index
            self.feed(mission, now)
            self.assertEqual(mission.step(now), (0., 0., False))
        self.join(mission)
        self.assertEqual(mission.step(now), (0., 0., False))
        self.assertIn('no extended perpendicular wall support', mission.reason)
        self.assertIsNone(mission.anchor)

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
        np.testing.assert_allclose(mission.localizer.estimate.call_args.args[0], wall_points)
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
                now = self.confirm(mission, pose)
                self.assertEqual(mission.state, 'ENTRY')
                self.assertEqual(mission.plans, 0)
                self.feed(mission, now+.1, pose)
                linear, angular, finished = mission.step(now+.1)
                self.assertGreater(linear, 0.)
                self.assertAlmostEqual(angular, 0., places=12)
                self.assertFalse(finished)

    def test_reusing_one_scan_cannot_confirm_alignment(self):
        mission = self.mission()
        self.feed(mission, 10.)
        mission.step(10.)
        self.join(mission)
        for _ in range(12):
            self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertEqual(mission.localization_confirmations, 1)
        self.assertEqual(mission.localizer.estimate.call_count, 1)

    def test_ambiguous_result_breaks_the_confirmation_sequence(self):
        mission = self.mission()
        accepted = mission.localizer.estimate.return_value
        mission.localizer.estimate.side_effect = [
            accepted, AlignmentResult(False, reason='ambiguous wall alignment'), accepted]
        for index in range(3):
            now = 10. + .1*index
            self.feed(mission, now)
            self.assertEqual(mission.step(now), (0., 0., False))
            self.join(mission)
            self.assertEqual(mission.step(now), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertEqual(mission.localization_confirmations, 1)

    def test_measured_motion_prevents_alignment_worker_start(self):
        mission = self.mission()
        self.feed(mission, 10., velocity=(.03, 0.))
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIn('measured rest', mission.reason)
        mission.localizer.estimate.assert_not_called()

    def test_stale_inputs_cannot_start_alignment(self):
        mission = self.mission()
        self.feed(mission, 10.)
        self.assertEqual(mission.step(10.4), (0., 0., False))
        self.assertIn('fresh', mission.reason)
        mission.localizer.estimate.assert_not_called()

    def test_worker_result_expiring_before_acceptance_is_rejected(self):
        mission = self.mission()
        self.feed(mission, 10.)
        mission.step(10.)
        self.join(mission)
        self.feed(mission, 10.4)
        self.assertEqual(mission.step(10.4), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertEqual(mission.localization_confirmations, 0)
        self.assertIn('expired', mission.reason)
        self.join(mission)

    def test_expired_fit_reuses_fresh_stopped_window_but_restarts_confirmations(self):
        mission = self.mission(observation_window=1.)
        for index in range(14):
            self.feed(mission, 10.+index*.1)
        # A slow worker finishes an earlier scan while fresh scans continue.
        mission.localization_result = (mission.localizer.estimate.return_value,
                                       (.04, 0., 0.), 10.9)
        mission.localization_attempt_stamp = 10.9
        mission.localization_candidate = (0., 0., 0.)
        mission.localization_confirmations = 2
        mission.localization_anchors = [(0., 0., 0.)]*2
        history_stamps = [row[2] for row in mission.localization_history]
        self.assertEqual(mission.step(11.3), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertIsNone(mission.localization_candidate)
        self.assertEqual(mission.localization_confirmations, 0)
        self.assertEqual(mission.localization_anchors, [])
        self.assertEqual([row[2] for row in mission.localization_history], history_stamps)
        mission.localizer.estimate.assert_not_called()

        # Refit on the next tick, without another second of collection.
        self.assertEqual(mission.step(11.35), (0., 0., False))
        self.join(mission)
        mission.localizer.estimate.assert_called_once()
        self.assertEqual(mission.localization_attempt_stamp, 11.3)
        self.assertEqual(mission.step(11.35), (0., 0., False))
        self.assertEqual(mission.localization_confirmations, 1)
        self.assertIsNone(mission.anchor)
        for now in (11.4, 11.5):
            self.feed(mission, now)
            self.assertEqual(mission.step(now), (0., 0., False))
            self.join(mission)
            self.assertEqual(mission.step(now), (0., 0., False))
            if now == 11.4:
                self.assertIsNone(mission.anchor)
        self.assertIsNotNone(mission.anchor)
        self.assertEqual(mission.localizer.estimate.call_count, 3)

    def test_expired_fit_still_clears_window_on_motion_or_rejected_geometry(self):
        for failure in ('motion', 'rejected_geometry'):
            with self.subTest(failure=failure):
                mission = self.mission(observation_window=1.)
                for index in range(14):
                    self.feed(mission, 10.+index*.1)
                scan_pose = (.034, 0., 0.) if failure == 'motion' else (.04, 0., 0.)
                result = (mission.localizer.estimate.return_value if failure == 'motion'
                          else AlignmentResult(False, reason='ambiguous wall alignment'))
                mission.localization_result = (result, scan_pose, 10.9)
                self.assertTrue(mission.localization_history)
                self.assertEqual(mission.step(11.3), (0., 0., False))
                self.assertEqual(mission.localization_history, [])
                self.assertEqual(mission.localization_confirmations, 0)
                self.assertIsNone(mission.anchor)

    def test_movement_since_acquisition_rejects_an_otherwise_good_result(self):
        mission = self.mission()
        self.feed(mission, 10.)
        mission.step(10.)
        self.join(mission)
        mission.update_odometry((.05, 0., 0.), 0., 0., 10.05)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertEqual(mission.localization_confirmations, 0)
        self.assertIn('moved', mission.reason)

    def test_reset_discards_a_worker_from_the_previous_activation(self):
        mission = self.mission()
        started, release = threading.Event(), threading.Event()
        accepted = mission.localizer.estimate.return_value
        def delayed(_points):
            started.set()
            if not release.wait(timeout=3.):
                raise RuntimeError('test wall fit was never released')
            return accepted
        mission.localizer.estimate.side_effect = delayed
        self.feed(mission, 10.)
        mission.step(10.)
        self.assertTrue(started.wait(timeout=1.))
        try:
            # A running fit cannot hold the mission lock and block callbacks.
            self.assertTrue(mission.update_odometry((.04, 0., 0.), 0., 0., 10.01))
            mission.reset()
        finally:
            release.set()
            self.join(mission)
        self.assertEqual(mission.state, 'LOCALIZING')
        self.assertIsNone(mission.anchor)
        self.assertIsNone(mission.localization_result)
        self.assertEqual(mission.localization_confirmations, 0)

    def test_motion_during_a_fit_invalidates_it_even_if_robot_returns(self):
        mission = self.mission()
        started, release = threading.Event(), threading.Event()
        accepted = mission.localizer.estimate.return_value
        def delayed(_points):
            started.set()
            if not release.wait(timeout=3.):
                raise RuntimeError('test wall fit was never released')
            return accepted
        mission.localizer.estimate.side_effect = delayed
        self.feed(mission, 10.)
        mission.step(10.)
        self.assertTrue(started.wait(timeout=1.))
        try:
            mission.update_odometry((.04, 0., 0.), .02, 0., 10.01)
            self.assertEqual(mission.step(10.01), (0., 0., False))
            mission.update_odometry((.04, 0., 0.), 0., 0., 10.02)
        finally:
            release.set()
            self.join(mission)
        self.assertEqual(mission.step(10.02), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertIsNone(mission.localization_result)
        self.assertEqual(mission.localization_confirmations, 0)

    def test_fit_exception_is_reported_without_motion_and_new_scan_can_retry(self):
        mission = self.mission()
        accepted = mission.localizer.estimate.return_value
        mission.localizer.estimate.side_effect = [ValueError('invalid fit fixture'), accepted]
        self.feed(mission, 10.)
        mission.step(10.)
        self.join(mission)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIn('wall alignment failed', mission.reason)
        self.assertIsNone(mission.anchor)
        self.feed(mission, 10.1)
        mission.step(10.1)
        self.join(mission)
        self.assertEqual(mission.step(10.1), (0., 0., False))
        self.assertEqual(mission.localization_confirmations, 1)

    def test_alignment_requires_new_mapped_scans_before_motion(self):
        mission = self.mission()
        now = self.confirm(mission)
        self.assertEqual(mission.initial_map_updates, 1)
        self.assertEqual(mission.step(now), (0., 0., False))
        self.assertIn('mapped scans', mission.reason)
        self.feed(mission, now+.1)
        linear, angular, finished = mission.step(now+.1)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)  # The first allowed motion is the heading turn.
        self.assertFalse(finished)

    def test_collision_at_the_localized_pose_keeps_the_robot_stopped(self):
        pose = (.14, .20, 0.)
        mission = self.mission(pose)
        now = self.confirm(mission, pose)
        self.feed(mission, now+.1, pose)
        self.assertEqual(mission.step(now+.1), (0., 0., False))
        self.assertIn('footprint or clearance is blocked in map', mission.reason)
        self.assertEqual(mission.plans, 0)

    def test_recorded_stop_pose_uses_actual_body_without_added_padding(self):
        pose = (.511643, -.003789, math.radians(26.334))
        mission = self.mission(pose)
        now = self.confirm(mission, pose)
        self.feed(mission, now + .1, pose)
        grid = mission._grid()
        self.assertTrue(mission.planner.pose_is_collision_free(grid, Pose2D(*pose)))
        self.assertEqual(mission.footprint.padding, 0.)
        linear, angular, _ = mission.step(now + .1)
        self.assertGreater(abs(linear) + abs(angular), 0.)

        # A real body overlap still stops motion; merely entering the former
        # 10 mm padded rectangle no longer does.
        tight = (pose[0], pose[1] - .015, pose[2])
        body_gap = (tight[1] - mission.footprint.rear * math.sin(tight[2])
                    - mission.footprint.half_width * math.cos(tight[2]) + .150)
        self.assertLess(body_gap, 0.)
        self.feed(mission, now + .2, tight)
        self.assertEqual(mission.step(now + .2), (0., 0., False))
        self.assertIn('clearance is blocked in map', mission.reason)

    def test_entry_endpoint_is_fixed_while_reported_travel_adapts(self):
        mission = self.mission()
        now = self.confirm(mission)
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
                now = self.confirm(mission, pose)
                previous = (0., 0.)
                for index in range(800):
                    now += .05
                    self.feed(mission, now, pose, velocity=previous)
                    linear, angular, finished = mission.step(now)
                    # Only wall-clock search is awaited; virtual time advances
                    # after the stopped worker completes, with no ROS transport.
                    self.join(mission)
                    self.assertFalse(finished)
                    localized = relative_pose(pose, mission.anchor)
                    self.assertTrue(mission.planner.pose_is_collision_free(mission._grid(), localized),
                                    (pose, mission.state, mission.reason))
                    if mission.state == 'WAIT_STOP':
                        self.assertEqual((linear, angular), (0., 0.))
                        break
                    pose = integrate(pose, linear, angular, .05)
                    previous = (linear, angular)
                else:
                    self.fail('ideal mission failed to reach exit: %s %s' % (mission.state, mission.reason))
                self.assertLess(math.hypot(pose[0]-1.808, pose[1]-1.868), .035)
                self.assertEqual(mission.plans, 1)
                # No lane observations are supplied anywhere in this closed
                # loop. The handoff follows a fresh measured stopped sample,
                # not the final zero command alone.
                self.assertEqual(mission.step(now), (0., 0., False))
                now += .05
                self.feed(mission, now, pose)
                self.assertEqual(mission.step(now), (0., 0., True))
                self.assertEqual(mission.state, 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
