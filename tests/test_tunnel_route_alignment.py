"""Route-change triggers and command-timed rotation through map overlaps."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.geometry import OccupancyGrid
from tunnel.mission import RouteUpdate, TunnelMission
from tunnel.planner import GlobalRoute
from tunnel.tracking import LimitedCommand
from test_tunnel_mission import configuration


class RouteAlignmentTest(unittest.TestCase):
    def make_mission(self, yaw=0., velocity=.4):
        config = configuration(goal=(2.5, 0., 0.))
        config['grid']['bounds'] = [-3., -3., 3., 3.]
        config['control'].update(cruise_velocity=.4,
                                 maximum_angular_velocity=1.4,
                                 angular_acceleration=8.)
        config['entry']['alignment_angular_velocity'] = .3
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        self.feed(mission, 10., yaw=yaw, velocity=velocity)
        mission.update_scan([math.inf], 0., 1., .1, 4.,
                            (0., 0., 0.), (0., 0., yaw), 10.)
        mission._initialize((0., 0., 0.))
        mission.global_path = self.route(yaw)
        mission.state = 'FOLLOWING'
        mission.last_linear = velocity
        mission.last_time = mission.last_plan_attempt = 10.
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        return mission

    @staticmethod
    def route(heading):
        return GlobalRoute(np.array([[0., 0.],
                                     [2.*math.cos(heading), 2.*math.sin(heading)]]))

    @staticmethod
    def feed(mission, stamp, yaw=0., velocity=0., angular=0.):
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return mission.update_odometry((0., 0., yaw), velocity, angular, stamp)

    def request_alignment(self, mission, yaw=0., offset=math.radians(137.), stamp=10.05):
        mission.pending_route = RouteUpdate(self.route(yaw+offset), '', .01)
        self.assertEqual(mission.step(stamp), (0., 0., False))

    def begin_turn(self, mission, yaw=0., offset=math.radians(137.)):
        self.request_alignment(mission, yaw, offset)
        self.feed(mission, 10.10, yaw)
        result = mission.step(10.10)
        self.assertEqual(result[0], 0.)
        self.assertNotEqual(result[1], 0., mission.reason)
        self.assertFalse(result[2])
        return math.copysign(1., result[1])

    def finish_sent_turn(self, mission, start_stamp=10.10, period=.05):
        """Observe output commands and elapsed time without moving odometry."""
        target = mission.route_turn.target_angle
        stamp, sent_angle = start_stamp, mission.route_turn.angle
        outputs = []
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, .1)) as local:
            for _ in range(1000):
                previous_angular = mission.last_angular
                outputs.append(previous_angular)
                sent_angle += abs(previous_angular) * period
                stamp += period
                command = mission.step(stamp)
                if mission.route_turn is None:
                    self.assertGreater(command[0], 0.)
                    local.assert_called_once()
                    self.assertGreaterEqual(sent_angle + 1e-9, target)
                    self.assertLessEqual(sent_angle - target,
                                         abs(previous_angular) * period + 1e-9)
                    return stamp, outputs
                local.assert_not_called()
                self.assertEqual(command[0], 0.)
                self.assertNotEqual(command[1], 0., mission.reason)
        self.fail('rotation did not hand off after the sent command angle')

    def test_changed_route_sends_immediate_zero_despite_previous_cruise_command(self):
        mission = self.make_mission()
        with patch.object(mission.local_planner, 'local_command') as local:
            self.request_alignment(mission)
        local.assert_not_called()
        self.assertEqual((mission.last_linear, mission.last_angular), (0., 0.))

    def test_rotation_starts_next_tick_without_waiting_for_measured_stop(self):
        mission = self.make_mission(velocity=.4)
        self.request_alignment(mission)
        self.assertEqual(mission.odom[-1], 10.)
        linear, angular, _ = mission.step(10.10)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0., mission.reason)

    def test_sent_137_degree_turn_finishes_with_completely_unchanged_odometry(self):
        mission = self.make_mission()
        sign = self.begin_turn(mission)
        self.assertEqual(sign, 1.)
        odometry = mission.odom
        self.finish_sent_turn(mission)
        self.assertEqual(mission.odom, odometry)

    def test_requested_turn_handles_both_directions_across_pi_wrap(self):
        for sign in (1., -1.):
            with self.subTest(direction=sign):
                start = sign*math.radians(170.)
                mission = self.make_mission(yaw=start)
                self.assertEqual(self.begin_turn(mission, start,
                                                 sign*math.radians(137.)), sign)
                _, outputs = self.finish_sent_turn(mission)
                self.assertTrue(all(sign * value > 0. for value in outputs))

    def test_route_changed_mid_turn_stops_before_retargeting(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        mission.pending_route = RouteUpdate(self.route(-math.pi/2.), '', .01)
        self.feed(mission, 10.15, math.radians(30.), angular=.3)
        self.assertEqual(mission.step(10.15), (0., 0., False))
        self.feed(mission, 10.20, math.radians(30.), angular=.1)
        linear, angular, _ = mission.step(10.20)
        self.assertEqual(linear, 0.)
        self.assertLess(angular, 0., mission.reason)
        self.finish_sent_turn(mission, start_stamp=10.20)

    def test_small_route_update_while_waiting_preserves_requested_direction(self):
        mission = self.make_mission()
        self.request_alignment(mission)
        turn = mission.route_turn
        mission.pending_route = RouteUpdate(self.route(math.pi/2.), '', .01)
        self.feed(mission, 10.10)
        linear, angular, _ = mission.step(10.10)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)
        self.assertIs(mission.route_turn, turn)
        self.assertAlmostEqual(turn.target_yaw, math.radians(137.))
        self.assertAlmostEqual(turn.target_angle, math.radians(137.))
        self.finish_sent_turn(mission)

    def test_large_route_change_retargets_even_if_odometry_already_reached_old_target(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        self.feed(mission, 10.15, math.radians(137.), angular=.1)
        self.assertGreater(mission.step(10.15)[1], 0.)
        mission.pending_route = RouteUpdate(self.route(0.), '', .01)
        self.feed(mission, 10.20, math.radians(137.))
        self.assertEqual(mission.step(10.20), (0., 0., False))
        self.feed(mission, 10.25, math.radians(137.))
        linear, angular, _ = mission.step(10.25)
        self.assertEqual(linear, 0.)
        self.assertLess(angular, 0., mission.reason)

    def test_small_route_updates_during_turn_never_replace_the_requested_target(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        turn = mission.route_turn
        target_yaw, target_angle = turn.target_yaw, turn.target_angle
        # Every accepted route changes by less than 90 degrees, even though
        # their accumulated change eventually exceeds the original target.
        for index, degrees in enumerate((127., 147., 117., 137., 157., 177., 197., 217., 237., 257., 277.)):
            stamp = 10.15+index*.05
            mission.pending_route = RouteUpdate(self.route(math.radians(degrees)), '', .01)
            self.feed(mission, stamp, math.radians(10.+index*8.), angular=.3)
            linear, angular, complete = mission.step(stamp)
            self.assertEqual(linear, 0.)
            self.assertGreater(angular, 0., mission.reason)
            self.assertFalse(complete)
            self.assertIs(mission.route_turn, turn)
            self.assertEqual(turn.phase, 'TURN')
            self.assertEqual((turn.target_yaw, turn.target_angle), (target_yaw, target_angle))

    def test_ninety_degree_route_change_mid_turn_explicitly_stops_and_retargets(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        turn = mission.route_turn
        mission.pending_route = RouteUpdate(self.route(math.radians(47.)), '', .01)
        self.feed(mission, 10.15, math.radians(30.), angular=.3)
        self.assertEqual(mission.step(10.15), (0., 0., False))
        self.assertIsNot(mission.route_turn, turn)
        self.feed(mission, 10.20, math.radians(30.))
        linear, angular, _ = mission.step(10.20)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)
        self.assertAlmostEqual(mission.route_turn.target_yaw, math.radians(47.))
        self.assertAlmostEqual(mission.route_turn.target_angle, math.radians(17.))

    def test_sub_ninety_degree_updates_during_stop_do_not_replace_saved_target(self):
        for offset in (-89.9, -60., -30., -8., 8., 30., 60., 89.9):
            with self.subTest(offset=offset):
                mission = self.make_mission()
                self.request_alignment(mission)
                turn = mission.route_turn
                mission.pending_route = RouteUpdate(self.route(math.radians(137.+offset)), '', .01)
                self.feed(mission, 10.10)
                linear, angular, _ = mission.step(10.10)
                self.assertEqual(linear, 0.)
                self.assertGreater(angular, 0.)
                self.assertIs(mission.route_turn, turn)
                self.assertAlmostEqual(turn.target_yaw, math.radians(137.))
                self.assertAlmostEqual(turn.target_angle, math.radians(137.))

    def test_sub_ninety_degree_latest_route_after_turn_does_not_start_another_turn(self):
        for degrees in (8., 30., 60., 89.9):
            with self.subTest(degrees=degrees):
                mission = self.make_mission()
                self.begin_turn(mission)
                stamp, _ = self.finish_sent_turn(mission)
                mission.pending_route = RouteUpdate(self.route(math.radians(137.+degrees)), '', .01)
                self.feed(mission, stamp+.05, math.radians(137.))
                with patch.object(mission.local_planner, 'local_command',
                                  return_value=LimitedCommand(.2, .1)) as local:
                    self.assertGreater(mission.step(stamp+.05)[0], 0.)
                local.assert_called_once()
                self.assertIsNone(mission.route_turn)

    def test_rotation_uses_fixed_speed_without_slowing_near_requested_angle(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        _, outputs = self.finish_sent_turn(mission)
        speed = mission.config['entry']['alignment_angular_velocity']
        np.testing.assert_allclose(outputs[-5:], speed)
        self.assertTrue(all(value <= speed+1e-9 for value in outputs))

    def test_rotation_acceleration_limit_applies_before_fixed_speed(self):
        mission = self.make_mission()
        mission.control['angular_acceleration'] = .5
        self.begin_turn(mission)
        self.assertAlmostEqual(mission.last_angular, .5 * .05)
        _, outputs = self.finish_sent_turn(mission)
        increments = np.diff([0., *outputs])
        self.assertTrue(np.all(increments <= .5 * .05 + 1e-9))
        np.testing.assert_allclose(outputs[-5:], .3)


    def test_odometry_reaching_target_early_does_not_end_the_command_budget(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        self.feed(mission, 10.15, math.radians(137.), velocity=.4, angular=.5)
        with patch.object(mission.local_planner, 'local_command') as local:
            linear, angular, _ = mission.step(10.15)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)
        local.assert_not_called()
        self.assertIsNotNone(mission.route_turn)

    def test_command_budget_uses_actual_elapsed_time_instead_of_nominal_period(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        stamp, outputs = self.finish_sent_turn(mission, period=.20)
        self.assertLess(stamp-10.10, math.radians(137.)/.3 + .25)
        self.assertGreater(len(outputs), 1)

    def test_no_stop_command_is_inserted_at_command_budget_completion(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        self.feed(mission, 10.11, angular=.5)
        self.finish_sent_turn(mission)
        self.assertIsNone(mission.route_turn)
        self.assertEqual((mission.last_linear, mission.last_angular), (.2, .1))

    def test_map_heading_correction_does_not_complete_sent_rotation(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        mission.anchor = (0., 0., -math.pi)
        self.feed(mission, 10.15, 0., angular=.1)
        result = mission.step(10.15)
        self.assertEqual(result[0], 0.)
        self.assertGreater(result[1], 0., mission.reason)

    def test_forward_route_refresh_keeps_driving_without_stop(self):
        mission = self.make_mission()
        mission.pending_route = RouteUpdate(self.route(math.radians(5.)), '', .01)
        self.feed(mission, 10.05, velocity=.4)
        with patch.object(mission.local_planner, 'local_command',
                          wraps=mission.local_planner.local_command) as local:
            result = mission.step(10.05)
        self.assertGreater(result[0], 0., mission.reason)
        local.assert_called_once()

    def test_sub_ninety_degree_replacement_routes_use_local_driving(self):
        for degrees in (8., 30., 60., 89.9):
            for sign in (1., -1.):
                with self.subTest(degrees=sign*degrees):
                    mission = self.make_mission()
                    mission.pending_route = RouteUpdate(self.route(math.radians(sign*degrees)), '', .01)
                    self.feed(mission, 10.05, velocity=.4)
                    with patch.object(mission.local_planner, 'local_command',
                                      return_value=LimitedCommand(.2, .1)) as local:
                        self.assertGreater(mission.step(10.05)[0], 0.)
                    local.assert_called_once()
                    self.assertIsNone(mission.route_turn)

    def test_exact_ninety_degree_first_and_replacement_routes_stop_across_pi_wrap(self):
        for first in (False, True):
            for start in (0., math.radians(170.), math.radians(-170.)):
                for sign in (1., -1.):
                    with self.subTest(first=first, start=start, direction=sign):
                        mission = self.make_mission(yaw=start)
                        if first:
                            mission.global_path = None
                        with patch.object(mission.local_planner, 'local_command') as local:
                            self.request_alignment(mission, yaw=start, offset=sign*math.pi/2.)
                        local.assert_not_called()
                        self.assertEqual((mission.last_linear, mission.last_angular), (0., 0.))

    def test_sub_ninety_degree_route_to_route_change_does_not_preempt_local_control(self):
        mission = self.make_mission()
        mission.global_path = self.route(math.radians(100.))
        mission.pending_route = RouteUpdate(self.route(math.radians(120.)), '', .01)
        self.feed(mission, 10.05, velocity=.4)
        with patch.object(mission.local_planner, 'local_command',
                          return_value=LimitedCommand(.2, .1)) as local:
            self.assertGreater(mission.step(10.05)[0], 0.)
        local.assert_called_once()
        self.assertIsNone(mission.route_turn)

    def test_first_route_aligns_even_when_heading_change_is_below_ninety_degrees(self):
        for degrees in (8., 30., 60., 89.9):
            for sign in (1., -1.):
                with self.subTest(degrees=sign*degrees):
                    mission = self.make_mission(velocity=0.)
                    mission.global_path = None
                    with patch.object(mission.local_planner, 'local_command') as local:
                        self.request_alignment(mission, offset=math.radians(sign*degrees))
                    local.assert_not_called()
                    self.feed(mission, 10.10)
                    linear, angular, _ = mission.step(10.10)
                    self.assertEqual(linear, 0.)
                    self.assertGreater(sign*angular, 0.)
                    self.assertAlmostEqual(mission.route_turn.target_angle, math.radians(degrees))

    def test_first_already_aligned_route_drives_next_tick_without_extra_rotation(self):
        mission = self.make_mission(velocity=0.)
        mission.global_path = None
        self.request_alignment(mission, offset=0.)
        self.feed(mission, 10.10)
        self.assertGreater(mission.step(10.10)[0], 0., mission.reason)
        self.assertIsNone(mission.route_turn)

    def test_route_change_compares_both_routes_at_the_same_current_position(self):
        mission = self.make_mission()
        mission.global_path = GlobalRoute(np.array(((0., 0.), (1., 0.), (1., 1.))))
        mission.pending_route = RouteUpdate(
            GlobalRoute(np.array(((0., 0.), (1., 0.), (1., -1.)))), '', .01)
        mission.update_odometry((.95, 0., 0.), .4, 0., 10.05)
        with patch.object(mission.local_planner, 'local_command') as local:
            self.assertEqual(mission.step(10.05), (0., 0., False))
        local.assert_not_called()
        self.assertLess(mission.route_turn.target_yaw, 0.)
        self.assertLess(abs(mission.route_turn.target_yaw), math.pi/2.)

    def test_route_change_threshold_is_loaded_in_degrees(self):
        mission = self.make_mission()
        self.assertAlmostEqual(mission.route_change_alignment_angle, math.pi/2.)

    def test_first_route_requires_alignment_before_forward_motion(self):
        mission = self.make_mission(velocity=0.)
        mission.global_path = None
        with patch.object(mission.local_planner, 'local_command') as local:
            self.request_alignment(mission)
        local.assert_not_called()
        self.feed(mission, 10.10)
        linear, angular, _ = mission.step(10.10)
        self.assertEqual(linear, 0.)
        self.assertGreater(angular, 0.)

    def test_behind_targets_without_route_change_do_not_request_rotation(self):
        for degrees in (120., 180.):
            with self.subTest(degrees=degrees):
                mission = self.make_mission()
                mission.global_path = self.route(math.radians(degrees))
                with patch.object(mission.local_planner, 'local_command',
                                  wraps=mission.local_planner.local_command) as local:
                    self.assertEqual(mission.step(10.05), (0., 0., False))
                local.assert_called_once()
                self.assertIsNone(mission.route_turn)
                self.assertIn('no local command', mission.reason)

    def test_same_route_recalculation_does_not_interrupt_a_forward_curve(self):
        mission = self.make_mission(yaw=math.radians(-20.))
        mission.global_path = self.route(0.)
        mission.pending_route = RouteUpdate(self.route(0.), '', .01)
        self.feed(mission, 10.05, math.radians(-20.), velocity=.4)
        self.assertGreater(mission.step(10.05)[0], 0., mission.reason)
        self.assertIsNone(mission.route_turn)

    def test_turn_uses_old_lidar_without_restoring_sensor_age_stop(self):
        mission = self.make_mission()
        self.request_alignment(mission)
        self.feed(mission, 15.)
        result = mission.step(15.)
        self.assertEqual(mission.scan[-1], 10.)
        self.assertEqual(result[0], 0.)
        self.assertGreater(result[1], 0., mission.reason)

    def test_current_body_map_overlap_does_not_interrupt_sent_turn(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        data = np.zeros((600, 600), dtype=np.int8)
        data[300, 300] = 100
        grid = OccupancyGrid(data, .01, -3., -3.)
        self.feed(mission, 10.15, angular=.1)
        with patch.object(mission, '_grid', return_value=grid):
            linear, angular, complete = mission.step(10.15)
            self.assertEqual(linear, 0.)
            self.assertGreater(angular, 0.)
            self.assertFalse(complete)
            self.finish_sent_turn(mission, start_stamp=10.15)
        self.assertIsNone(mission.route_turn)



    def test_requested_angle_stays_shortest_turn_despite_surrounding_map_cells(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                mission = self.make_mission()
                self.request_alignment(mission, offset=sign*math.radians(137.))
                self.feed(mission, 10.10)
                grid = OccupancyGrid(np.full((600, 600), 100, dtype=np.int8), .01, -3., -3.)
                with patch.object(mission, '_grid', return_value=grid):
                    linear, angular, complete = mission.step(10.10)
                self.assertEqual(linear, 0.)
                self.assertGreater(sign*angular, 0., mission.reason)
                self.assertFalse(complete)
                self.assertAlmostEqual(mission.route_turn.target_angle, math.radians(137.))


    def test_reset_clears_in_progress_alignment(self):
        mission = self.make_mission()
        self.begin_turn(mission)
        mission.reset()
        self.feed(mission, 11.)
        mission._initialize((0., 0., 0.))
        mission.global_path = self.route(0.)
        mission.state = 'FOLLOWING'
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        self.assertGreater(mission.step(11.)[0], 0., mission.reason)


if __name__ == '__main__':
    unittest.main()
