"""FAST source-profile regressions; ideal entry motion is not Gazebo evidence."""

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes/stepbystep'))

from tunnel.mission import TunnelMission, load_config


CONFIG = ROOT / 'codes/stepbystep/tunnel.yaml'


class TunnelProfileTests(unittest.TestCase):
    def test_normal_is_the_default_and_keeps_validated_low_speed_values(self):
        normal = load_config(CONFIG, 'normal')
        self.assertEqual(load_config(CONFIG), normal)
        expected = {'cruise_velocity': .075, 'minimum_velocity': .035,
                    'entry_velocity': .04, 'exit_velocity': .025,
                    'maximum_angular_velocity': .55,
                    'maximum_lateral_acceleration': .03,
                    'linear_acceleration': .15, 'linear_deceleration': .40,
                    'angular_acceleration': .55,
                    'safety_linear_deceleration': .15,
                    'safety_angular_deceleration': .55,
                    'safety_reaction_time': .10}
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(normal['control'][name], value)
        self.assertEqual(normal['entry']['cruise_velocity'], .04)
        self.assertEqual(normal['grid']['endpoint_clear_guard_radius'], .030)

    def test_fast_values_match_source_simulation_profile(self):
        fast = load_config(CONFIG, 'fast')
        # Source: custom_autorace_bringup/config/fast_gazebo.yaml, tunnel.
        expected = {'cruise_velocity': .32, 'minimum_velocity': .035,
                    'entry_velocity': .035, 'exit_velocity': .32,
                    'maximum_angular_velocity': 1.4,
                    'maximum_lateral_acceleration': .30,
                    'linear_acceleration': .35, 'angular_acceleration': 8.,
                    'safety_linear_deceleration': .40,
                    'safety_reaction_time': .15}
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(fast['control'][name], value)
        self.assertEqual(fast['entry']['cruise_velocity'], .32)
        self.assertEqual(fast['grid']['endpoint_clear_guard_radius'], 0.)
        self.assertAlmostEqual(fast['planner']['goal_heading_tolerance'], math.radians(10.))
        # Source exit apron world x_max=.45 in the frozen staging frame.
        self.assertAlmostEqual(fast['planner']['keep_in_rectangles'][-1][-1], 2.1975895)

    def test_profile_merge_does_not_mutate_normal_or_duplicate_calibration(self):
        normal = load_config(CONFIG)
        before = copy.deepcopy(normal)
        fast = load_config(CONFIG, 'fast')
        self.assertEqual(normal, before)
        self.assertEqual(load_config(CONFIG), before)
        for key in ('footprint', 'sensors', 'goal', 'exit_connector'):
            self.assertEqual(fast[key], normal[key])
        self.assertEqual(fast['grid']['static_map'], normal['grid']['static_map'])
        self.assertTrue(Path(fast['grid']['static_map']['image']).is_file())
        self.assertNotIn('profiles', normal)
        self.assertNotIn('profiles', fast)

    def test_unknown_profile_is_rejected_instead_of_silently_using_normal(self):
        with self.assertRaisesRegex(ValueError, 'profile'):
            load_config(CONFIG, 'typo-fast')

    def entry_run(self, profile='normal', obstacle=None, config=None):
        # Isolate the velocity law from wall alignment and real map geometry.
        config = copy.deepcopy(config) if config is not None else load_config(CONFIG, profile)
        config.pop('localization', None)
        config['grid'].pop('static_map')
        config['grid'].pop('static_hit_exclusion_radius')
        # This one-dimensional fixture isolates speed/braking. The physical
        # heading bias and swept body checks have separate 2-D entry tests.
        config['entry']['heading_offset_deg'] = 0.
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        pose, linear, now = 0., 0., 10.
        dt = config['control']['period']
        acceleration = config['entry'].get('linear_acceleration', config['control']['linear_acceleration'])
        trace = []
        with patch.object(mission, '_grid', side_effect=AssertionError('entry grid guard')):
            for _ in range(700):
                mission.update_odometry((pose, 0., 0.), linear, 0., now)
                measured_range = math.inf if obstacle is None else obstacle-pose
                mission.update_scan([measured_range], 0., 1., .05, 40.,
                                    (0., 0., 0.), (pose, 0., 0.), now)
                command, angular, complete = mission.step(now)
                trace.append((pose, linear, command, mission.state))
                self.assertEqual(angular, 0.)
                self.assertFalse(complete)
                self.assertLessEqual(command-linear, acceleration*dt+1e-9)
                self.assertLessEqual(linear-command, config['control']['linear_deceleration']*dt+1e-9)
                if mission.state == 'PLANNING':
                    self.assertEqual(command, 0.)
                    break
                self.assertEqual(mission.state, 'ENTRY')
                self.assertLessEqual(command, config['entry']['cruise_velocity']+1e-9)
                linear, pose, now = command, pose+command*dt, now+dt
        self.assertEqual(mission.state, 'PLANNING', mission.reason)
        self.assertEqual(mission.plans, 0)
        self.assertLessEqual(pose, config['entry']['distance'])
        if obstacle is not None:
            self.assertGreaterEqual(obstacle-pose-mission.footprint.front,
                                    config['entry']['obstacle_clearance'])
        # PLANNING must still wait for actual measured rest, not treat its
        # zero command as proof the moving body has already stopped.
        if linear > config['control']['planning_stopped_linear']:
            with patch.object(mission.planner, 'plan') as search:
                mission.step(now+dt)
                search.assert_not_called()
        return config, np.asarray([row[2] for row in trace]), trace

    def test_fast_straight_entry_accelerates_and_brakes_before_distance_goal(self):
        config, commands, trace = self.entry_run('fast')
        self.assertNotIn('linear_acceleration', config['entry'])
        self.assertAlmostEqual(commands[0], .35 * config['control']['period'])
        self.assertGreater(max(commands), .25)
        self.assertLess(commands[-2], .04)
        self.assertGreater(np.count_nonzero(np.diff(commands) < -1e-6), 5)
        self.assertLess(len(trace)*config['control']['period'], 8.)

    def test_fast_entry_brakes_before_lidar_front_margin_when_obstacle_is_nearer(self):
        config, commands, trace = self.entry_run('fast', obstacle=.85)
        self.assertGreater(max(commands), .20)
        self.assertLess(commands[-2], .04)
        self.assertLess(trace[-1][0], config['entry']['distance']-.3)

    def test_normal_entry_keeps_its_previous_speed_cap(self):
        config, commands, _ = self.entry_run('normal')
        self.assertNotIn('linear_acceleration', config['entry'])
        self.assertAlmostEqual(commands[0], .15 * config['control']['period'])
        self.assertAlmostEqual(max(commands), .04)

    def test_physical_entry_uses_gentler_acceleration_and_still_brakes(self):
        for directory in ('stepbystep', 'final_codes'):
            physical = load_config(ROOT / 'codes' / directory / 'tunnel.real.yaml')
            for obstacle in (None, .85):
                with self.subTest(directory=directory, obstacle=obstacle):
                    config, commands, trace = self.entry_run(config=physical, obstacle=obstacle)
                    dt = config['control']['period']
                    self.assertAlmostEqual(commands[0], .10 * dt)
                    self.assertLessEqual(max(np.diff(np.r_[0., commands])), .10 * dt + 1e-9)
                    self.assertLessEqual(max(commands), .25 + 1e-9)
                    self.assertEqual(config['control']['cruise_velocity'], .32)
                    self.assertEqual(config['control']['linear_acceleration'], .15)
                    self.assertGreater(np.count_nonzero(np.diff(commands) < -1e-6), 5)
                    if obstacle is None:
                        self.assertAlmostEqual(max(commands), .25)
                    else:
                        self.assertLess(trace[-1][0], config['entry']['distance'] - .3)

    def test_entry_distance_ramp_uses_its_own_acceleration(self):
        config = load_config(CONFIG, 'fast')
        config['entry']['linear_acceleration'] = .10
        config['grid'].pop('static_map')
        config['grid'].pop('static_hit_exclusion_radius')
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        dt = config['control']['period']
        command = 0.
        # Odometry advances 5 cm then stays there. Let the command rate limit
        # settle so the distance-based reference, rather than that limit,
        # determines the eventual speed (about .106 m/s at a=.10 m/s²).
        for tick in range(60):
            now = 10. + tick * dt
            pose = (0. if tick == 0 else .05, 0., 0.)
            mission.update_odometry(pose, command, 0., now)
            mission.update_scan([math.inf], 0., 1., .05, 40.,
                                (0., 0., 0.), pose, now)
            command, angular, complete = mission.step(now)
            self.assertEqual(mission.state, 'ENTRY')
            self.assertEqual(angular, 0.)
            self.assertFalse(complete)
        self.assertGreater(command, .10)
        self.assertLess(command, .11)

    def test_invalid_entry_acceleration_is_rejected(self):
        for value in (0., -.10, math.nan, math.inf, None, True, '0.10'):
            with self.subTest(value=value):
                config = load_config(CONFIG)
                config['entry']['linear_acceleration'] = value
                with self.assertRaisesRegex(ValueError, 'entry.*linear_acceleration'):
                    TunnelMission(config)

    def goal_mission(self):
        config = load_config(CONFIG, 'fast')
        config['grid'].pop('static_map')
        config['grid'].pop('static_hit_exclusion_radius')
        config['planner']['keep_in_rectangles'] = None
        config['goal']['pose'] = [1., 0., 0.]
        config['goal'].pop('planning_pose')
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([math.inf], 0., 1., .05, 40.,
                            (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        x = np.linspace(0., 1., 51)
        mission.path = mission._tracking_path(x, x*0., x*0., x*0.)
        mission.state, mission.last_linear, mission.last_angular = 'FOLLOWING', .08, .6
        mission.update_odometry((.98, 0., 0.), .08, .6, 10.05)
        return mission

    def test_outside_goal_decelerates_then_confirms_rest_without_lane_frames(self):
        mission = self.goal_mission()
        dt, linear, angular, x = mission.control['period'], .08, .6, .98
        for tick in range(10):
            now = 10.05+tick*dt
            mission.update_odometry((x, 0., 0.), linear, angular, now)
            mission.update_scan([math.inf], 0., 1., .05, 40.,
                                (0., 0., 0.), (x, 0., 0.), now)
            command, yaw, complete = mission.step(now)
            self.assertFalse(complete)
            self.assertLessEqual(linear-command, mission.control['linear_deceleration']*dt+1e-9)
            self.assertLessEqual(abs(angular-yaw), mission.control['angular_acceleration']*dt+1e-9)
            if mission.state == 'WAIT_STOP':
                self.assertEqual((command, yaw), (0., 0.))
                self.assertEqual(mission.completion_started, now)
                break
            self.assertEqual(mission.state, 'FOLLOWING')
            self.assertIn('stopping at outside goal', mission.reason)
            self.assertIsNone(mission.completion_started)
            linear, angular, x = command, yaw, x+command*dt
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.assertLessEqual(abs(x-mission.goal.x), mission.config['goal']['position_tolerance'])
        self.assertGreater(tick, 1)
        # A zero command is not evidence of measured rest. The same odometry
        # cannot complete the handoff; a fresh stopped sample can, without
        # any camera callbacks or lane confirmations.
        self.assertEqual(mission.step(now), (0., 0., False))
        now += dt
        mission.update_odometry((x, 0., 0.), 0., 0., now)
        mission.update_scan([math.inf], 0., 1., .05, 40.,
                            (0., 0., 0.), (x, 0., 0.), now)
        self.assertEqual(mission.step(now), (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')

    def test_outside_goal_current_footprint_conflict_still_blocks_motion(self):
        mission = self.goal_mission()
        # A measured point within the body stays a hard obstacle even when
        # goal distance/orientation are already within completion tolerance.
        for i in range(mission.config['grid']['mark_observations']):
            stamp = 10.05 + i*.05
            mission.update_odometry((.98, 0., 0.), .08, .6, stamp)
            mission.update_scan([.055], 0., 1., .05, 40.,
                                (0., 0., 0.), (.98, 0., 0.), stamp)
        self.assertEqual(mission.step(stamp), (0., 0., False))
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIsNone(mission.completion_started)
        self.assertIn('localized footprint', mission.reason)

    def test_fast_curvature_profile_keeps_yaw_lateral_and_wheel_acceleration_bounds(self):
        config = load_config(CONFIG, 'fast')
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        station = np.linspace(0., 2., 201)
        curvature = np.zeros_like(station)
        curvature[60:130] = 1./config['planner']['minimum_turning_radius']
        path = mission._tracking_path(station, station*0., station*0., curvature)
        cfg = config['control']
        self.assertGreater(max(path.speed), .30)
        self.assertLessEqual(max(path.speed), cfg['cruise_velocity'])
        self.assertLessEqual(max(abs(path.speed*curvature)), cfg['maximum_angular_velocity']+1e-9)
        self.assertLessEqual(max(path.speed**2*abs(curvature)), cfg['maximum_lateral_acceleration']+1e-9)
        elapsed = 2.*np.diff(path.station)/(path.speed[:-1]+path.speed[1:])
        yaw_acceleration = abs(np.diff(path.speed*curvature))/elapsed
        self.assertLessEqual(max(yaw_acceleration), cfg['angular_acceleration']*(1.+1e-9))
        # Source custom model: wheel separation .1482 m, acceleration 1 m/s².
        # This source budget check is not a physical motor capability claim.
        self.assertLessEqual(max(cfg['linear_acceleration'], cfg['linear_deceleration'])
                             + .1482/2.*cfg['angular_acceleration'], 1.)


if __name__ == '__main__':
    unittest.main()
