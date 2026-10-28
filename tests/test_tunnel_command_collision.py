"""Real rectangular-body checks over the next command tick, without ROS."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.mission import TunnelMission
from tunnel.geometry import OccupancyGrid, Pose2D
from tunnel.planner import GlobalRoute
from tunnel.tracking import LimitedCommand
from test_tunnel_mission import configuration


class LocalCommandGeometryTest(unittest.TestCase):
    def make_mission(self):
        config = configuration()
        config['footprint']['padding'] = 0.
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        return mission

    def test_rotating_corner_is_checked_when_linear_speed_is_zero(self):
        mission = self.make_mission()
        values = np.zeros((400, 400), dtype=np.int8)
        values[292:296, 258:266] = 100
        grid = OccupancyGrid(values, .001, -.2, -.2)
        pose = Pose2D(0., 0., 0.)
        self.assertTrue(mission.geometry.pose_is_collision_free(grid, pose))
        self.assertFalse(mission.geometry.pose_is_collision_free(grid, Pose2D(0., 0., .05)))
        self.assertFalse(mission.local_planner.command_is_collision_free(
            grid, pose, LimitedCommand(0., 1.), mission.control))

    def test_obstacle_beyond_next_tick_does_not_extend_the_command_check(self):
        mission = self.make_mission()
        values = np.zeros((200, 400), dtype=np.int8)
        values[:, 325] = 100  # x=.125m; current front is only .067645m.
        grid = OccupancyGrid(values, .001, -.2, -.1)
        pose, command = Pose2D(0., 0., 0.), LimitedCommand(.2, 0.)
        self.assertTrue(mission.geometry.primitive_is_collision_free(
            grid, pose, 0., command.linear_velocity*mission.control['period']))
        self.assertTrue(mission.local_planner.command_is_collision_free(
            grid, pose, command, mission.control))
        self.assertFalse(mission.geometry.primitive_is_collision_free(
            grid, pose, 0., mission.local_planner.local_horizon))

    def test_zero_command_checks_current_body_without_extrapolating_motion(self):
        mission = self.make_mission()
        values = np.zeros((200, 400), dtype=np.int8)
        values[:, 325] = 100
        grid = OccupancyGrid(values, .001, -.2, -.1)
        pose = Pose2D(0., 0., 0.)
        self.assertTrue(mission.geometry.pose_is_collision_free(grid, pose))
        self.assertTrue(mission.local_planner.command_is_collision_free(
            grid, pose, LimitedCommand(0., 0.), mission.control))
        values[:, 265] = 100  # x=.065m now intersects the actual front.
        occupied = OccupancyGrid(values, .001, -.2, -.1)
        self.assertFalse(mission.local_planner.command_is_collision_free(
            occupied, pose, LimitedCommand(0., 0.), mission.control))

    def test_rate_limited_command_checks_corner_even_when_candidate_is_straight(self):
        mission = self.make_mission()
        mission.control.update(maximum_angular_velocity=1.4, angular_acceleration=.5)
        values = np.zeros((600, 800), dtype=np.int8)
        values[292:296, 258:266] = 100
        grid = OccupancyGrid(values, .001, -.2, -.2)
        pose = Pose2D(0., 0., 0.)
        route = GlobalRoute(np.array(((0., 0.), (.5, 0.))))
        # The straight candidate clears this cell, but angular acceleration
        # cannot cancel last tick's positive turn before its corner touches it.
        self.assertTrue(mission.geometry.primitive_is_collision_free(
            grid, pose, 0., mission.local_planner.local_horizon))
        command = mission.local_planner.local_command(
            grid, pose, route, mission.control, .01, 1., mission.control['period'])
        self.assertIsNone(command)


if __name__ == '__main__':
    unittest.main()
