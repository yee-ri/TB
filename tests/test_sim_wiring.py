"""Keep simulator wiring isolated from both hardware and original controllers."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SimulationWiringTests(unittest.TestCase):
    def test_roslaunch_entrypoint_uses_noetic_python3(self):
        entrypoint = ROOT / 'codes/stepbystep/try_maze.py'
        self.assertEqual(entrypoint.read_text().splitlines()[0], '#!/usr/bin/env python3')
        self.assertTrue(entrypoint.stat().st_mode & 0o111)

    def test_container_has_no_external_ros_network_or_writable_source(self):
        service = yaml.safe_load((ROOT / 'compose.gazebo.yaml').read_text())['services']['gazebo']
        self.assertEqual(service['network_mode'], 'none')
        self.assertNotIn('ports', service)
        self.assertNotIn('privileged', service)
        self.assertNotIn('devices', service)
        source = [v for v in service['volumes'] if ':/workspace:' in v]
        self.assertEqual(len(source), 1)
        self.assertTrue(source[0].endswith(':/workspace:ro'))
        self.assertIn('.:/target:ro', service['volumes'])

    def test_launch_reuses_only_environment_and_one_tb_controller(self):
        launch = ET.parse(ROOT / 'launch/tunnel_gazebo.launch').getroot()
        includes = launch.findall('include')
        self.assertEqual(len(includes), 1)
        self.assertEqual(includes[0].get('file'),
                         '$(find custom_autorace_description)/launch/gazebo_autorace.launch')
        nodes = launch.findall('.//node')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].get('type'), 'try_maze.py')
        params = {p.get('name'): p.get('value') for p in nodes[0].findall('param')}
        self.assertEqual(params['start_step'], '9')
        self.assertEqual(params['lidar_type'], 'laserscan')
        self.assertEqual(params['lidar_topic'], '/scan_mid360_raw')
        self.assertEqual(params['tunnel_profile'], '$(arg tunnel_profile)')
        self.assertEqual(launch.find("arg[@name='tunnel_profile']").get('default'), 'normal')
        self.assertEqual(nodes[0].find('remap').attrib,
                         {'from': '/odom', 'to': '/odometry/filtered'})
        args = {a.get('name'): a.get('value') for a in includes[0].findall('arg')}
        self.assertEqual(args['fuse_imu'], 'true')
        self.assertEqual(args['odometry_source'], 'world')
        self.assertEqual(args['tunnel_obstacle_layout'], '$(arg tunnel_obstacle_layout)')
        self.assertEqual(launch.find("arg[@name='tunnel_obstacle_layout']").get('default'),
                         'layout_a')

    def test_compose_exposes_fast_as_an_opt_in_profile_not_another_node(self):
        service = yaml.safe_load((ROOT / 'compose.gazebo.yaml').read_text())['services']['gazebo']
        self.assertIn('tunnel_profile:=${TB_TUNNEL_PROFILE:-normal}', service['command'][-1])

    def test_compose_selects_existing_layouts_without_changing_the_default(self):
        service = yaml.safe_load((ROOT / 'compose.gazebo.yaml').read_text())['services']['gazebo']
        self.assertIn('tunnel_obstacle_layout:=${TB_TUNNEL_LAYOUT:-layout_a}',
                      service['command'][-1])


if __name__ == '__main__':
    unittest.main()
