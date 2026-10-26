"""Real rectangular-body command and braking sweeps, without ROS."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))
from tunnel.mission_v2 import TunnelMissionV2
from tunnel.planner import OccupancyGrid, Pose2D
from tunnel.tracking import LimitedCommand
from test_tunnel_mission import configuration


class LocalCommandGeometryTest(unittest.TestCase):
    def make_mission(self):
        config = configuration()
        config['footprint']['padding'] = 0.
        mission = TunnelMissionV2(config)
        self.addCleanup(mission.close)
        return mission

    def test_rotating_corner_is_checked_when_linear_speed_is_zero(self):
        mission = self.make_mission()
        values = np.zeros((400, 400), dtype=np.int8)
        values[292:296, 258:266] = 100
        grid = OccupancyGrid(values, .001, -.2, -.2)
        pose = Pose2D(0., 0., 0.)
        self.assertTrue(mission.planner.pose_is_collision_free(grid, pose))
        self.assertFalse(mission.planner.pose_is_collision_free(grid, Pose2D(0., 0., .05)))
        self.assertFalse(mission.local_planner.command_is_safe(
            grid, pose, LimitedCommand(0., 1.), mission.control))

    def test_obstacle_beyond_next_tick_still_blocks_braking_sweep(self):
        mission = self.make_mission()
        values = np.zeros((200, 400), dtype=np.int8)
        values[:, 325] = 100  # x=.125m; current front is only .067645m.
        grid = OccupancyGrid(values, .001, -.2, -.1)
        pose, command = Pose2D(0., 0., 0.), LimitedCommand(.2, 0.)
        self.assertTrue(mission.planner.primitive_is_collision_free(
            grid, pose, 0., command.linear_velocity*mission.control['period']))
        self.assertFalse(mission.local_planner.command_is_safe(
            grid, pose, command, mission.control))

    def test_measured_motion_cannot_be_erased_by_a_zero_command(self):
        mission = self.make_mission()
        values = np.zeros((200, 400), dtype=np.int8)
        values[:, 325] = 100
        grid = OccupancyGrid(values, .001, -.2, -.1)
        pose = Pose2D(0., 0., 0.)
        self.assertTrue(mission.planner.pose_is_collision_free(grid, pose))
        self.assertTrue(mission.local_planner.command_is_safe(
            grid, pose, LimitedCommand(0., 0.), mission.control))
        self.assertFalse(mission.local_planner.command_is_safe(
            grid, pose, LimitedCommand(0., 0.), mission.control,
            measured_linear=.2, measured_angular=0.))


if __name__ == '__main__':
    unittest.main()
