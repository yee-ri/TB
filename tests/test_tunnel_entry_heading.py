"""Direct turn-to-drive entry, legacy settling, and actual-body checks."""

import copy
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes/stepbystep'))
from tunnel import mission as step_mission
from tunnel.mission_v2 import TunnelMissionV2

# Exercise the deployed final copy as well as the step-by-step implementation.
spec = importlib.util.spec_from_file_location(
    'entry_test_final_tunnel', ROOT / 'codes/final_codes/tunnel/__init__.py',
    submodule_search_locations=[str(ROOT / 'codes/final_codes/tunnel')])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
final_mission = importlib.import_module('entry_test_final_tunnel.mission')
IMPLEMENTATIONS = (('stepbystep', step_mission), ('final_codes', final_mission))
LEGACY_IMPLEMENTATIONS = (('final_codes', final_mission),)
DIRECT_MISSIONS = (('v1', step_mission.TunnelMission), ('v2', TunnelMissionV2))


def integrate(pose, linear, angular, period):
    x, y, yaw = pose
    heading = yaw + angular * period
    if abs(angular) < 1e-12:
        return x + linear * period * math.cos(yaw), y + linear * period * math.sin(yaw), heading
    return (x + linear / angular * (math.sin(heading) - math.sin(yaw)),
            y - linear / angular * (math.cos(heading) - math.cos(yaw)), heading)


class TunnelEntryHeadingTest(unittest.TestCase):
    def configuration(self, folder):
        config = step_mission.load_config(ROOT / 'codes' / folder / 'tunnel.real.yaml')
        config.pop('localization', None)
        config['grid'].pop('static_map', None)
        config['grid'].pop('static_hit_exclusion_radius', None)
        config['grid']['bounds'] = [-1., -1., 2., 2.]
        config['planner']['keep_in_rectangles'] = None
        return config

    def make_mission(self, folder, module, config=None, start_yaw=0., mission_type=None):
        mission_type = module.TunnelMission if mission_type is None else mission_type
        mission = mission_type(self.configuration(folder) if config is None else config)
        self.addCleanup(mission.close)
        mission._initialize(anchor=(0., 0., 0.))
        mission.entry_start_x, mission.entry_start_yaw = 0., start_yaw
        return mission

    def feed(self, mission, pose, now, linear=0., angular=0.):
        mission.update_odometry(pose, linear, angular, now)
        mission.update_scan([math.inf], 0., 1., .05, 4.,
                            (0., 0., 0.), pose, now)

    def box_grid(self, module, rectangle):
        data = np.zeros((100, 100), dtype=np.int8)
        left, bottom, right, top = rectangle
        cells = {}
        for row in range(math.floor((bottom + 1.) / .02), math.ceil((top + 1.) / .02)):
            for column in range(math.floor((left + 1.) / .02), math.ceil((right + 1.) / .02)):
                x, y = -1. + column * .02, -1. + row * .02
                box = (max(left, x), max(bottom, y), min(right, x + .02), min(top, y + .02))
                if box[2] - box[0] < 1e-10 or box[3] - box[1] < 1e-10:
                    continue
                data[row, column] = 100
                cells[row * 100 + column] = [box]
        return module.OccupancyGrid(data, .02, -1., -1., static_cell_rectangles=cells)

    def test_physical_entry_turns_then_drives_with_expected_handoff(self):
        for folder, module in IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                config = self.configuration(folder)
                self.assertEqual(config['entry']['heading_offset_deg'], 5.)
                self.assertEqual(config['entry']['maximum_curvature'], .35)
                config['entry']['distance'] = .7
                mission = self.make_mission(folder, module, config)
                pose, linear, angular, now = (0., 0., 0.), 0., 0., 10.
                period = mission.control['period']
                maximum_speed = 0.
                first_forward = None
                phases = set()
                for _ in range(400):
                    self.feed(mission, pose, now, linear, angular)
                    command, turn, complete = mission.step(now)
                    self.assertFalse(complete)
                    self.assertGreaterEqual(command, 0.)
                    self.assertLessEqual(command, config['entry']['cruise_velocity'] + 1e-9)
                    self.assertLessEqual(command - linear,
                                         config['entry']['linear_acceleration'] * period + 1e-9)
                    phases.add(mission.entry_phase)
                    if first_forward is None:
                        self.assertAlmostEqual(pose[0], 0.)
                        self.assertAlmostEqual(pose[1], 0.)
                    if command > 0.:
                        self.assertEqual(mission.entry_phase, 'DRIVE')
                        if first_forward is None:
                            first_forward = pose
                            if folder == 'final_codes':
                                self.assertLessEqual(abs(angular), mission.control['planning_stopped_angular'])
                            else:
                                # Forward motion starts while the previous
                                # pivot command is still present in feedback.
                                self.assertGreater(abs(angular), 0.)
                            self.assertLessEqual(abs(pose[2] - math.radians(5.)),
                                                 math.radians(config['entry']['alignment_tolerance_deg']))
                        # The accepted heading is frozen: ideal straight
                        # motion no longer keeps bending toward a new target.
                        self.assertAlmostEqual(turn, 0., places=12)
                        self.assertAlmostEqual(pose[2], first_forward[2], places=12)
                    elif mission.entry_phase != 'DRIVE':
                        self.assertLessEqual(abs(turn), config['entry']['alignment_angular_velocity'] + 1e-9)
                    self.assertGreaterEqual(pose[2], -1e-9)
                    self.assertLessEqual(pose[2], math.radians(5.3) + 1e-9)
                    maximum_speed = max(maximum_speed, command)
                    if mission.state == 'PLANNING':
                        self.assertEqual((command, turn), (0., 0.))
                        break
                    self.assertEqual(mission.state, 'ENTRY')
                    pose = integrate(pose, command, turn, period)
                    linear, angular, now = command, turn, now + period
                self.assertEqual(mission.state, 'PLANNING')
                self.assertEqual(mission.plans, 0)
                self.assertIn('ALIGN', phases)
                if folder == 'final_codes':
                    self.assertIn('SETTLE', phases)
                else:
                    self.assertNotIn('SETTLE', phases)
                self.assertIsNotNone(first_forward)
                self.assertAlmostEqual(maximum_speed, config['entry']['cruise_velocity'])
                self.assertGreater(pose[1], .01)
                self.assertGreater(pose[2], math.radians(1.))
                self.assertLessEqual(pose[0], .7)

    def test_legacy_rotation_waits_for_rest_and_requires_new_stopped_feedback(self):
        for folder, module in LEGACY_IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                mission = self.make_mission(folder, module)
                self.feed(mission, (0., 0., 0.), 10., .03, .1)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'WAIT_STOP')
                self.feed(mission, (0., 0., 0.), 10.05)
                linear, angular, _ = mission.step(10.05)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)
                self.assertLessEqual(angular, mission.config['entry']['alignment_angular_acceleration'] * .05 + 1e-9)
                self.assertEqual(mission.entry_phase, 'ALIGN')
                target = (0., 0., math.radians(5.))
                self.feed(mission, target, 10.10)
                self.assertEqual(mission.step(10.10), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')
                # Reusing the sample that requested zero is not proof of rest.
                self.assertEqual(mission.step(10.10), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')
                self.feed(mission, target, 10.15, angular=.08)
                self.assertEqual(mission.step(10.15), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')
                self.feed(mission, target, 10.20)
                self.assertEqual(mission.step(10.20), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'DRIVE')
                self.feed(mission, target, 10.25)
                linear, angular, _ = mission.step(10.25)
                self.assertGreater(linear, 0.)
                self.assertAlmostEqual(angular, 0., places=12)

    def test_legacy_recorded_stopping_drift_does_not_restart_or_reverse_the_turn(self):
        for folder, module in LEGACY_IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                mission = self.make_mission(folder, module)
                self.feed(mission, (0., 0., 0.), 10.)
                linear, angular, _ = mission.step(10.)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)
                self.assertEqual(mission.entry_phase, 'ALIGN')

                # Trial 20261014_115316 reached 4.997 degrees while still
                # turning at .248 rad/s, then stopped at 7.966 degrees.
                # Instantaneous ideal-actuator tests cannot expose this drift.
                reached = (0., 0., math.radians(4.997))
                self.feed(mission, reached, 10.05, angular=.248)
                self.assertEqual(mission.step(10.05), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')
                self.assertIsNone(mission.entry_drive_yaw)

                # Even a stopped value cannot reuse the sample that requested
                # zero; continued rotation in a newer sample also forbids drive.
                self.feed(mission, reached, 10.05)
                self.assertEqual(mission.step(10.05), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')
                self.feed(mission, (0., 0., math.radians(6.5)), 10.10, angular=.15)
                self.assertEqual(mission.step(10.10), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'SETTLE')

                stopped = (0., 0., math.radians(7.966))
                self.feed(mission, stopped, 10.15)
                self.assertEqual(mission.step(10.15), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'DRIVE')
                self.assertAlmostEqual(mission.entry_drive_yaw, stopped[2])
                self.feed(mission, stopped, 10.20)
                linear, angular, _ = mission.step(10.20)
                self.assertGreater(linear, 0.)
                self.assertAlmostEqual(angular, 0.)

    def test_legacy_skipped_heading_target_stops_without_reversal_in_either_direction(self):
        for folder, module in LEGACY_IMPLEMENTATIONS:
            for sign in (-1, 1):
                with self.subTest(folder=folder, direction=sign):
                    config = self.configuration(folder)
                    config['entry']['heading_offset_deg'] = sign * 5.
                    start = math.radians(sign * 179.)
                    mission = self.make_mission(folder, module, config, start_yaw=start)
                    pose = (0., 0., start)
                    self.feed(mission, pose, 10.)
                    linear, angular, _ = mission.step(10.)
                    self.assertEqual(linear, 0.)
                    self.assertGreater(sign * angular, 0.)
                    self.assertLessEqual(abs(angular), mission.config['entry']['alignment_angular_velocity'])

                    # A sample jumps from zero to seven degrees of relative
                    # rotation, crossing both the target and the +/-pi wrap.
                    # Missing the tolerance band must not command a reversal.
                    crossed = (0., 0., math.radians(-sign * 174.))
                    self.feed(mission, crossed, 10.05, angular=sign * .10)
                    self.assertEqual(mission.step(10.05), (0., 0., False))
                    self.assertEqual(mission.entry_phase, 'SETTLE')
                    self.feed(mission, crossed, 10.10)
                    self.assertEqual(mission.step(10.10), (0., 0., False))
                    self.assertEqual(mission.entry_phase, 'DRIVE')
                    self.feed(mission, crossed, 10.15)
                    linear, angular, _ = mission.step(10.15)
                    self.assertGreater(linear, 0.)
                    self.assertAlmostEqual(angular, 0.)

    def test_direct_entry_retains_initial_rest_but_moves_on_the_target_tick(self):
        for version, mission_type in DIRECT_MISSIONS:
            with self.subTest(version=version):
                mission = self.make_mission('stepbystep', step_mission, mission_type=mission_type)
                self.feed(mission, (0., 0., 0.), 10., .03, .1)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertEqual(mission.entry_phase, 'WAIT_STOP')
                self.feed(mission, (0., 0., 0.), 10.05)
                linear, angular, _ = mission.step(10.05)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)
                self.assertEqual(mission.entry_phase, 'ALIGN')

                target = (0., 0., math.radians(5.))
                self.feed(mission, target, 10.10, linear=.015, angular=.248)
                linear, angular, complete = mission.step(10.10)
                self.assertFalse(complete)
                self.assertGreater(linear, 0., mission.reason)
                self.assertLessEqual(linear, mission.entry_linear_acceleration * .05 + 1e-9)
                self.assertAlmostEqual(angular, 0.)
                self.assertEqual(mission.entry_phase, 'DRIVE')
                self.assertAlmostEqual(mission.entry_drive_yaw, target[2])

    def test_direct_entry_coasting_does_not_restart_an_in_place_turn(self):
        for version, mission_type in DIRECT_MISSIONS:
            with self.subTest(version=version):
                mission = self.make_mission('stepbystep', step_mission, mission_type=mission_type)
                self.feed(mission, (0., 0., 0.), 10.)
                self.assertGreater(mission.step(10.)[1], 0.)
                reached = (0., 0., math.radians(4.997))
                self.feed(mission, reached, 10.05, angular=.248)
                linear, angular, _ = mission.step(10.05)
                self.assertGreater(linear, 0.)
                self.assertAlmostEqual(angular, 0.)
                self.assertEqual(mission.entry_phase, 'DRIVE')

                # Replay the observed coasting yaw. Gentle moving corrections
                # are allowed, but the pivot must not restart or wait for rest.
                for now, yaw, measured_angular in ((10.10, 6.5, .15), (10.15, 7.966, 0.)):
                    self.feed(mission, (.001, 0., math.radians(yaw)), now,
                              linear=linear, angular=measured_angular)
                    linear, angular, _ = mission.step(now)
                    self.assertGreater(linear, 0., mission.reason)
                    self.assertLessEqual(angular, 0.)
                    self.assertLessEqual(abs(angular), linear * mission.entry_maximum_curvature + 1e-9)
                    self.assertEqual(mission.entry_phase, 'DRIVE')
                    self.assertAlmostEqual(mission.entry_drive_yaw, reached[2])

    def test_direct_entry_crossed_target_moves_immediately_in_both_wrap_directions(self):
        for version, mission_type in DIRECT_MISSIONS:
            for sign in (-1, 1):
                with self.subTest(version=version, direction=sign):
                    config = self.configuration('stepbystep')
                    config['entry']['heading_offset_deg'] = sign * 5.
                    start = math.radians(sign * 179.)
                    mission = self.make_mission('stepbystep', step_mission, config,
                                                start_yaw=start, mission_type=mission_type)
                    self.feed(mission, (0., 0., start), 10.)
                    self.assertGreater(sign * mission.step(10.)[1], 0.)
                    crossed = (0., 0., math.radians(-sign * 174.))
                    self.feed(mission, crossed, 10.05, angular=sign * .10)
                    linear, angular, _ = mission.step(10.05)
                    self.assertGreater(linear, 0., mission.reason)
                    self.assertAlmostEqual(angular, 0.)
                    self.assertEqual(mission.entry_phase, 'DRIVE')
                    self.assertAlmostEqual(mission.entry_drive_yaw, crossed[2])

    def test_direct_entry_transition_keeps_freshness_body_and_front_stop_gates(self):
        for version, mission_type in DIRECT_MISSIONS:
            for gate in ('stale', 'body', 'front'):
                with self.subTest(version=version, gate=gate):
                    mission = self.make_mission('stepbystep', step_mission, mission_type=mission_type)
                    self.feed(mission, (0., 0., 0.), 10.)
                    self.assertGreater(mission.step(10.)[1], 0.)
                    reached = (0., 0., math.radians(5.))
                    if gate == 'stale':
                        mission.update_odometry(reached, 0., .08, 10.5)
                        self.assertEqual(mission.step(10.5), (0., 0., False))
                        self.assertIn('waiting for fresh', mission.reason)
                    else:
                        self.feed(mission, reached, 10.05, angular=.08)
                        if gate == 'body':
                            grid = self.box_grid(step_mission, (-.01, -.01, .01, .01))
                            with patch.object(mission, '_grid', return_value=grid):
                                self.assertEqual(mission.step(10.05), (0., 0., False))
                            self.assertIn('current body overlaps', mission.reason)
                        else:
                            mission.front_clearance = mission.config['entry']['obstacle_clearance']
                            self.assertEqual(mission.step(10.05), (0., 0., False))
                            self.assertEqual(mission.state, 'PLANNING')
                            self.assertIn('front clearance', mission.reason)

    def test_initial_rotation_does_not_predict_rear_corner_overlap(self):
        for folder, module in IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                mission = self.make_mission(folder, module)
                grid = self.box_grid(module, (-.11405, -.0954, -.11365, -.0950))
                self.assertTrue(mission.planner.pose_is_collision_free(grid, (0., 0., 0.)))
                self.assertFalse(mission.planner.pose_is_collision_free(grid, (0., 0., math.radians(2.5))))
                self.feed(mission, (0., 0., 0.), 10.)
                with patch.object(mission, '_grid', return_value=grid), \
                        patch.object(mission.planner, 'rotation_is_collision_free', create=True,
                                     side_effect=AssertionError('entry predicted a rear-corner sweep')) as rotation, \
                        patch.object(mission.planner, 'primitive_is_collision_free',
                                     side_effect=AssertionError('entry predicted a moving sweep')) as moving:
                    linear, angular, _ = mission.step(10.)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)
                rotation.assert_not_called()
                moving.assert_not_called()
                self.assertEqual(mission.state, 'ENTRY')

    def test_physical_profile_keeps_unpadded_body_and_right_width_with_left_preview(self):
        for folder, module in IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                mission = self.make_mission(folder, module)
                self.assertEqual(mission.footprint.padding, 0.)
                self.assertEqual(mission.footprint.half_width, .0903)
                self.assertEqual(mission.entry_obstacle_half_width, .0903)
                self.assertEqual(mission.entry_obstacle_right_width, .0903)
                self.assertEqual(mission.entry_obstacle_left_far_width, .30)
                self.assertEqual(mission.entry_obstacle_check_distance, .80)
                grid = self.box_grid(module, (-.11, -.096, -.10, -.094))
                self.assertTrue(mission.planner.pose_is_collision_free(grid, (0., 0., 0.)))
                previous_padding = module.HybridAStarPlanner(module.RectangularFootprint(
                    mission.footprint.front, mission.footprint.rear, mission.footprint.half_width, .01))
                self.assertFalse(previous_padding.pose_is_collision_free(grid, (0., 0., 0.)))
                self.feed(mission, (0., 0., 0.), 10.)
                with patch.object(mission, '_grid', return_value=grid):
                    linear, angular, _ = mission.step(10.)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)

    def test_current_physical_rear_body_overlap_still_stops_rotation_and_drive(self):
        for folder, module in IMPLEMENTATIONS:
            for phase in ('ALIGN', 'DRIVE'):
                with self.subTest(folder=folder, phase=phase):
                    mission = self.make_mission(folder, module)
                    mission.entry_phase, mission.entry_drive_yaw = phase, 0.
                    grid = self.box_grid(module, (-.11, -.089, -.105, -.085))
                    self.assertFalse(mission.planner.pose_is_collision_free(grid, (0., 0., 0.)))
                    self.feed(mission, (0., 0., 0.), 10.)
                    self.assertTrue(math.isinf(mission.front_clearance))
                    with patch.object(mission, '_grid', return_value=grid):
                        self.assertEqual(mission.step(10.), (0., 0., False))
                    self.assertIn('current body overlaps', mission.reason)

    def test_drive_and_braking_do_not_use_predicted_entry_sweeps(self):
        for folder, module in IMPLEMENTATIONS:
            for braking in (False, True):
                with self.subTest(folder=folder, braking=braking):
                    mission = self.make_mission(folder, module)
                    mission.entry_phase, mission.entry_drive_yaw = 'DRIVE', 0.
                    pose = module.Pose2D(.2, 0., 0.)
                    self.feed(mission, (pose.x, pose.y, pose.yaw), 10., .15)
                    mission.last_linear = .15
                    grid = self.box_grid(module, (.275, -.01, .277, .01))
                    self.assertTrue(mission.planner.pose_is_collision_free(grid, pose))
                    self.assertFalse(mission.planner.pose_is_collision_free(grid, (.21, 0., 0.)))
                    if braking:
                        mission.front_clearance = mission.config['entry']['obstacle_clearance'] + .005
                    with patch.object(mission, '_grid', return_value=grid), \
                            patch.object(mission.planner, 'primitive_is_collision_free',
                                         side_effect=AssertionError('entry predicted a moving sweep')) as sweep:
                        linear, angular, _ = mission._enter(pose, .05, 10.)
                    sweep.assert_not_called()
                    self.assertGreater(linear, 0.)
                    self.assertAlmostEqual(linear, .13 if braking else
                                           .15 + mission.entry_linear_acceleration * .05)
                    self.assertAlmostEqual(angular, 0.)
                    self.assertEqual(mission.state, 'ENTRY')

    def test_front_lidar_stop_extends_left_preview_but_keeps_right_body_width(self):
        for folder, module in IMPLEMENTATIONS:
            for lateral in (-.095, -.085, .085, .095):
                with self.subTest(folder=folder, lateral=lateral):
                    mission = self.make_mission(folder, module)
                    mission.entry_phase, mission.entry_drive_yaw = 'DRIVE', 0.
                    mission.update_odometry((0., 0., 0.), 0., 0., 10.)
                    mission.update_cloud([[.16, lateral]], (0., 0., 0.), (0., 0., 0.), 10.)
                    linear, angular, _ = mission.step(10.)
                    if lateral >= -mission.footprint.half_width:
                        self.assertEqual((linear, angular), (0., 0.))
                        self.assertEqual(mission.state, 'PLANNING')
                        self.assertIn('front clearance', mission.reason)
                    else:
                        self.assertGreater(linear, 0.)
                        self.assertEqual(mission.state, 'ENTRY')

    def test_left_preview_excludes_side_rear_and_returns_beyond_its_far_edge(self):
        for folder, module in IMPLEMENTATIONS:
            config = self.configuration(folder)
            front = config['footprint']['front']
            far_x = front + config['entry']['obstacle_check_distance']
            middle_x = .5 * far_x
            middle_left = .5 * (.0903 + .30)
            cases = (
                ('ahead-left', (.20, .12), True),
                ('same-distance-right', (.20, -.12), False),
                ('side-left', (.02, .12), False),
                ('behind-left', (-.15, .05), False),
                ('inside-left-boundary', (middle_x, middle_left - .0001), True),
                ('outside-left-boundary', (middle_x, middle_left + .0001), False),
                ('inside-right-boundary', (middle_x, -.0902), True),
                ('outside-right-boundary', (middle_x, -.0904), False),
                ('inside-far-boundary', (far_x - .0001, .20), True),
                ('outside-far-boundary', (far_x + .0001, .20), False),
            )
            for name, (x, y), detected in cases:
                with self.subTest(folder=folder, case=name):
                    mission = self.make_mission(folder, module, copy.deepcopy(config))
                    mission.update_scan([math.hypot(x, y)], math.atan2(y, x), 1., .05, 3.,
                                        (0., 0., 0.), (0., 0., 0.), 10.)
                    if detected:
                        self.assertAlmostEqual(mission.front_clearance, x - front)
                    else:
                        self.assertEqual(mission.front_clearance, math.inf)

    def test_left_preview_stops_for_ahead_left_point_without_changing_body_geometry(self):
        for folder, module in IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                mission = self.make_mission(folder, module)
                mission.entry_phase, mission.entry_drive_yaw = 'DRIVE', 0.
                mission.update_odometry((0., 0., 0.), 0., 0., 10.)
                mission.update_scan([math.hypot(.20, .12)], math.atan2(.12, .20), 1., .05, 3.,
                                    (0., 0., 0.), (0., 0., 0.), 10.)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertEqual(mission.state, 'PLANNING')
                self.assertIn('front clearance', mission.reason)
                self.assertEqual(mission.plans, 0)
                self.assertEqual(mission.footprint.padding, 0.)
                self.assertEqual(mission.planner.footprint, mission.footprint)

    def test_left_preview_does_not_remove_excluded_returns_from_scan_or_map(self):
        points = np.array(((.45, .17), (.10, .25), (1., .20), (.45, -.17), (-.20, .12)))
        for folder, module in IMPLEMENTATIONS:
            with self.subTest(folder=folder):
                wide_config = self.configuration(folder)
                legacy_config = copy.deepcopy(wide_config)
                legacy_config['entry']['obstacle_left_far_width'] = .0903
                wide = self.make_mission(folder, module, wide_config)
                legacy = self.make_mission(folder, module, legacy_config)
                for mission in (wide, legacy):
                    for index in range(mission.config['grid']['mark_observations']):
                        stamp = 10. + .05 * index
                        mission.update_odometry((0., 0., 0.), 0., 0., stamp)
                        mission.update_cloud(points, (0., 0., 0.), (0., 0., 0.), stamp)
                self.assertTrue(math.isfinite(wide.front_clearance))
                self.assertEqual(legacy.front_clearance, math.inf)
                self.assertEqual(np.count_nonzero(np.isfinite(wide.scan[0])), len(points))
                np.testing.assert_array_equal(wide.scan[0], legacy.scan[0])
                np.testing.assert_array_equal(wide.costmap._dynamic_occupied,
                                              legacy.costmap._dynamic_occupied)
                np.testing.assert_array_equal(wide.costmap.to_occupancy_data(),
                                              legacy.costmap.to_occupancy_data())
                # Each separated return survives into a confirmed hard cell.
                # Angular-bin centres can place that cell beside the raw XY.
                self.assertEqual(sum(wide.costmap._dynamic_occupied), len(points))

    def test_equal_left_preview_width_preserves_legacy_unbounded_strip(self):
        for folder, module in IMPLEMENTATIONS:
            for omit_far_width in (False, True):
                with self.subTest(folder=folder, default=omit_far_width):
                    config = self.configuration(folder)
                    if omit_far_width:
                        config['entry'].pop('obstacle_left_far_width', None)
                    else:
                        config['entry']['obstacle_left_far_width'] = .0903
                    mission = self.make_mission(folder, module, config)
                    mission.update_scan([1.2], 0., 1., .05, 3.,
                                        (0., 0., 0.), (0., 0., 0.), 10.)
                    self.assertAlmostEqual(mission.front_clearance, 1.2 - mission.footprint.front)

    def test_left_preview_parameters_reject_nonfinite_or_invalid_bounds(self):
        for folder, module in IMPLEMENTATIONS:
            config = self.configuration(folder)
            cases = {
                'obstacle_left_far_width': (True, False, np.bool_(True), None, '.3', math.nan,
                                            math.inf, -math.inf, -.1, .0902),
                'obstacle_check_distance': (True, False, np.bool_(True), None, '.8', math.nan,
                                            math.inf, -math.inf, 0., -.1,
                                            config['entry']['obstacle_clearance'],
                                            config['sensors']['range_max'] + .001),
            }
            for key, values in cases.items():
                for value in values:
                    candidate = copy.deepcopy(config)
                    candidate['entry'][key] = value
                    with self.subTest(folder=folder, key=key, value=value), \
                            self.assertRaisesRegex(ValueError, 'entry.*' + key):
                        module.TunnelMission(candidate)

    def test_absent_offset_preserves_legacy_heading_feedback_without_sweep(self):
        for folder, module in IMPLEMENTATIONS:
            commands = []
            for explicit_zero in (False, True):
                config = self.configuration(folder)
                config['entry'].pop('heading_offset_deg', None)
                config['entry'].pop('maximum_curvature', None)
                if explicit_zero:
                    config['entry']['heading_offset_deg'] = 0.
                mission = self.make_mission(folder, module, config)
                pose = module.Pose2D(.2, 0., .05)
                self.feed(mission, (pose.x, pose.y, pose.yaw), 10.)
                with patch.object(mission, '_grid', side_effect=AssertionError('legacy entry changed')):
                    commands.append(mission._enter(pose, .05, 10.))
            with self.subTest(folder=folder):
                self.assertEqual(commands[0], commands[1])
                self.assertGreater(commands[0][0], 0.)
                self.assertLess(commands[0][1], 0.)

    def test_entry_heading_parameters_reject_invalid_values(self):
        for folder, module in IMPLEMENTATIONS:
            config = self.configuration(folder)
            cases = {
                'heading_offset_deg': (True, None, '5', math.nan, math.inf, -math.inf, -15.1, 15.1),
                'maximum_curvature': (True, None, '.35', math.nan, math.inf, 0., -.1,
                                      1. / config['planner']['minimum_turning_radius'] + .01),
                'alignment_angular_velocity': (True, None, '.1', math.nan, math.inf, 0., -.1,
                                               config['control']['maximum_angular_velocity'] + .01),
                'alignment_angular_acceleration': (True, None, '.25', math.nan, math.inf, 0., -.1,
                                                   config['control']['angular_acceleration'] + .01),
                'alignment_tolerance_deg': (True, None, '.3', math.nan, math.inf, 0., -.1, 1.01),
            }
            for key, values in cases.items():
                for value in values:
                    candidate = copy.deepcopy(config)
                    candidate['entry'][key] = value
                    if key == 'maximum_curvature':
                        candidate['entry']['heading_offset_deg'] = 0.
                    with self.subTest(folder=folder, key=key, value=value), \
                            self.assertRaisesRegex(ValueError, 'entry.*' + key):
                        module.TunnelMission(candidate)


if __name__ == '__main__':
    unittest.main()
