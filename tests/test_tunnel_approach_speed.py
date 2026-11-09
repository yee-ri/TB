"""Inside-goal slowing without an added stop-confirmation phase."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_tunnel_exit_stages as stages
from test_tunnel_exit_stages import CONFIG, INSIDE
from test_tunnel_planner import CONTROL, grid_with
from tunnel.geometry import Pose2D
from tunnel.mission import TunnelMission, load_config
from tunnel.planner import GlobalLocalPlanner, GlobalRoute


class ApproachSpeedTest(unittest.TestCase):
    def mission(self, **kwargs):
        helper = stages.ExitStagesTest()
        self.addCleanup(helper.doCleanups)
        return helper, helper.mission(**kwargs)

    def run_approach(self, arrival_speed):
        x, y = INSIDE[0]-.8, INSIDE[1]
        helper, mission = self.mission(pose=(x, y, 0.))
        mission.exit_approach_velocity = arrival_speed
        mission.global_path = GlobalRoute(np.array(((x, y), INSIDE)))
        mission.worker = SimpleNamespace(is_alive=lambda: True)
        mission.state = 'FOLLOWING'
        mission.last_linear = .4
        stamp, velocity = 10., .4
        commands = []
        for _ in range(120):
            stamp += .05
            helper.feed(mission, (x, y, 0.), stamp, linear=velocity)
            linear, angular, complete = mission.step(stamp)
            self.assertFalse(complete)
            if mission.route_goal is mission.goal:
                self.assertEqual((linear, angular), (0., 0.))
                self.assertIsNone(mission.route_turn)
                return commands, stamp-10.
            self.assertGreater(linear, 0., 'approach must not wait before reaching the point')
            self.assertGreaterEqual(linear, velocity-.4*.05-1e-9)
            x += linear*.05
            velocity = linear
            commands.append(linear)
        self.fail('inside waypoint was never reached')

    def test_approach_reaches_inside_point_at_lower_speed_without_waiting(self):
        baseline, baseline_time = self.run_approach(.4)
        slower, slower_time = self.run_approach(.2)
        self.assertAlmostEqual(baseline[-1], .4)
        self.assertLessEqual(slower[-1], .23)
        self.assertGreaterEqual(slower[-1], .2)
        self.assertAlmostEqual(slower[0], baseline[0])
        self.assertLess(slower_time-baseline_time, 1.)

    def test_limit_uses_inside_point_not_local_lookahead_or_waypoint(self):
        _, mission = self.mission()
        pose = Pose2D(INSIDE[0]-.19, INSIDE[1], 0.)
        expected = mission._approach_speed_limit(pose)
        for points in (((1.2, 1.48), INSIDE),
                       ((1.2, 1.48), (1.4, 1.3), INSIDE)):
            mission.global_path = GlobalRoute(np.array(points))
            for lookahead in (.1, .3, .5):
                mission.local_planner.local_lookahead = lookahead
                self.assertEqual(mission._approach_speed_limit(pose), expected)
        self.assertEqual(expected, .2)
        self.assertIsNone(mission._approach_speed_limit(Pose2D(INSIDE[0]-.21, INSIDE[1], 0.)))
        boundary = Pose2D(INSIDE[0]-.035, INSIDE[1], 0.)
        self.assertAlmostEqual(mission._approach_speed_limit(boundary), .2)

    def test_outside_and_unstaged_goals_have_no_approach_limit(self):
        _, mission = self.mission()
        mission.route_goal = mission.goal
        self.assertIsNone(mission._approach_speed_limit(Pose2D(*INSIDE, 0.)))
        _, unstaged = self.mission(staged=False)
        self.assertIsNone(unstaged._approach_speed_limit(Pose2D(*INSIDE, 0.)))

    def test_speed_target_keeps_curve_and_actual_deceleration_limit(self):
        _, mission = self.mission()
        planner = GlobalLocalPlanner(mission.geometry)
        route = GlobalRoute(np.array(((0., 0.), (1., .05))))
        control = dict(CONTROL, cruise_velocity=.4, safety_linear_deceleration=.15)
        with patch.object(planner, '_clearance_score', return_value=(0, 0.)):
            normal = planner.local_command(grid_with(), (0., 0., 0.), route,
                                           control, .2, 0., .05)
            slow = planner.local_command(grid_with(), (0., 0., 0.), route,
                                         control, .2, 0., .05, speed_limit=.2)
            ramp = planner.local_command(grid_with(), (0., 0., 0.), route,
                                         control, .4, 0., .05, speed_limit=.2)
        self.assertLess(slow.linear_velocity, normal.linear_velocity)
        self.assertAlmostEqual(slow.angular_velocity/slow.linear_velocity,
                               normal.angular_velocity/normal.linear_velocity)
        self.assertAlmostEqual(ramp.linear_velocity, .4-.4*.05)
        self.assertLessEqual(abs(slow.linear_velocity*slow.angular_velocity),
                             control['maximum_lateral_acceleration'])

    def test_missing_setting_preserves_cruise_and_lower_profile_stays_lower(self):
        config = load_config(CONFIG)
        config['goal'].pop('approach_velocity')
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        mission.route_goal = mission.exit_approach_goal
        self.assertIsNone(mission._approach_speed_limit(Pose2D(*INSIDE, 0.)))
        planner = mission.local_planner
        control = dict(mission.control, cruise_velocity=.08)
        route = GlobalRoute(np.array(((0., 0.), (1., 0.))))
        command = planner.local_command(grid_with(), (0., 0., 0.), route,
                                        control, .08, 0., .05, speed_limit=.15)
        self.assertAlmostEqual(command.linear_velocity, .08)

    def test_invalid_arrival_speed_is_rejected_when_loading_config(self):
        for speed in (True, 0., -.1, math.nan, math.inf):
            with self.subTest(speed=speed):
                config = load_config(CONFIG)
                config['goal']['approach_velocity'] = speed
                with self.assertRaisesRegex(ValueError, 'goal/approach_velocity'):
                    TunnelMission(config)


if __name__ == '__main__':
    unittest.main()
