"""Local command generation keeps rate limits without a map-overlap veto."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from tunnel.mission import TunnelMission
from tunnel.geometry import OccupancyGrid, Pose2D
from tunnel.planner import GlobalRoute
from test_tunnel_body_parts import FRONT, REAR, HALF_WIDTH, PARTS
from test_tunnel_mission import configuration


class LocalCommandOverlapTest(unittest.TestCase):
    def make_mission(self, components=False):
        config = configuration()
        config['footprint'] = dict(front=.067645, rear=REAR,
                                   half_width=HALF_WIDTH, padding=0.)
        if components:
            config['footprint'].update(front=FRONT, parts=PARTS)
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        return mission

    def test_rate_limited_turn_is_returned_when_its_corner_overlaps_map(self):
        for components in (False, True):
            with self.subTest(components=components):
                mission = self.make_mission(components)
                mission.control.update(maximum_angular_velocity=1.4, angular_acceleration=.5)
                values = np.zeros((600, 800), dtype=np.int8)
                if components:
                    values[281:284, 242:246] = 100
                else:
                    values[292:296, 258:266] = 100
                grid = OccupancyGrid(values, .001, -.2, -.2)
                pose = Pose2D(0., 0., 0.)
                route = GlobalRoute(np.array(((0., 0.), (.5, 0.))))
                self.assertTrue(mission.geometry.pose_is_collision_free(grid, pose))
                self.assertFalse(mission.geometry.pose_is_collision_free(grid, Pose2D(0., 0., .05)))
                command = mission.local_planner.local_command(
                    grid, pose, route, mission.control, .01, 1., mission.control['period'])
                self.assertIsNotNone(command)
                self.assertGreater(command.linear_velocity, 0.)
                self.assertGreaterEqual(command.angular_velocity, 1.-.5*mission.control['period'])
                self.assertLessEqual(command.angular_velocity, mission.control['maximum_angular_velocity'])

    def test_endpoint_command_remains_zero_even_when_map_overlaps_body(self):
        mission = self.make_mission(components=True)
        values = np.zeros((600, 800), dtype=np.int8)
        values[200, 700] = 100
        grid = OccupancyGrid(values, .001, -.2, -.2)
        pose = Pose2D(.5, 0., 0.)
        route = GlobalRoute(np.array(((0., 0.), (.5, 0.))))
        self.assertFalse(mission.geometry.pose_is_collision_free(grid, pose))
        command = mission.local_planner.local_command(
            grid, pose, route, mission.control, 0., 0., mission.control['period'])
        self.assertIsNotNone(command)
        self.assertEqual((command.linear_velocity, command.angular_velocity), (0., 0.))


if __name__ == '__main__':
    unittest.main()
