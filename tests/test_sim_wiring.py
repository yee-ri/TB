"""Keep simulator wiring isolated from both hardware and original controllers."""
from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIM_PACKAGE = ROOT / 'ros/tb_tunnel_sim'


class SimulationWiringTests(unittest.TestCase):
    def test_roslaunch_entrypoint_uses_noetic_python3(self):
        entrypoint = ROOT / 'codes/stepbystep/try_maze.py'
        self.assertEqual(entrypoint.read_text().splitlines()[0], '#!/usr/bin/env python3')
        self.assertTrue(entrypoint.stat().st_mode & 0o111)

    def test_launch_resolves_the_stepbystep_controller_unambiguously(self):
        launch = ET.parse(SIM_PACKAGE / 'launch/tunnel_gazebo.launch').getroot()
        node_type = launch.find('node').get('type')
        matches = [path for path in SIM_PACKAGE.rglob(node_type)
                   if path.is_file() and path.stat().st_mode & 0o111]
        self.assertEqual(matches, [SIM_PACKAGE / 'scripts/tb_tunnel_controller.py'])
        self.assertTrue(matches[0].is_symlink())
        self.assertEqual(matches[0].resolve(), ROOT / 'codes/stepbystep/try_maze.py')

    def test_simulation_package_links_to_the_existing_launch_and_config(self):
        self.assertEqual(ET.parse(SIM_PACKAGE / 'package.xml').getroot().findtext('name'),
                         'tb_tunnel_sim')
        self.assertFalse((ROOT / 'package.xml').exists())
        for relative, target in (
                ('launch/tunnel_gazebo.launch', 'launch/tunnel_gazebo.launch'),
                ('config/tunnel.yaml', 'codes/stepbystep/tunnel.yaml')):
            with self.subTest(relative=relative):
                link = SIM_PACKAGE / relative
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), ROOT / target)

    def test_ros_resolver_excludes_recorded_controllers(self):
        import roslib.packages
        import rospkg

        with tempfile.TemporaryDirectory(prefix='tunnel checkout ') as directory, \
                patch.dict(os.environ, {'CMAKE_PREFIX_PATH': ''}):
            checkout = Path(directory)
            package = checkout / 'ros/tb_tunnel_sim'
            shutil.copytree(SIM_PACKAGE, package, symlinks=True)
            controller = checkout / 'codes/stepbystep/try_maze.py'
            controller.parent.mkdir(parents=True)
            shutil.copy2(ROOT / 'codes/stepbystep/try_maze.py', controller)
            archived_package = checkout / 'records/old_run/code'
            archived = archived_package / 'scripts/tb_tunnel_controller.py'
            archived.parent.mkdir(parents=True)
            shutil.copy2(SIM_PACKAGE / 'package.xml', archived_package / 'package.xml')
            shutil.copy2(controller, archived)
            recorded_bytes = archived.read_bytes()
            recorded_mode = archived.stat().st_mode

            rospack = rospkg.RosPack(ros_paths=[str(checkout / 'ros')])
            self.assertEqual(rospack.get_path('tb_tunnel_sim'), str(package))
            self.assertEqual(roslib.packages.find_node(
                'tb_tunnel_sim', 'tb_tunnel_controller.py', rospack=rospack),
                [str(controller)])
            self.assertEqual(archived.read_bytes(), recorded_bytes)
            self.assertEqual(archived.stat().st_mode, recorded_mode)

            # A duplicate inside the active package must still be detected.
            duplicate = package / 'duplicate/tb_tunnel_controller.py'
            duplicate.parent.mkdir()
            shutil.copy2(controller, duplicate)
            self.assertCountEqual(roslib.packages.find_node(
                'tb_tunnel_sim', 'tb_tunnel_controller.py', rospack=rospack),
                [str(controller), str(duplicate)])

    def test_compose_limits_ros_package_discovery_to_ros_directory(self):
        service = yaml.safe_load((ROOT / 'compose.gazebo.yaml').read_text())['services']['gazebo']
        self.assertIn('export ROS_PACKAGE_PATH=/target/ros:$$ROS_PACKAGE_PATH',
                      service['command'][-1])

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
        self.assertEqual(nodes[0].get('type'), 'tb_tunnel_controller.py')
        params = {p.get('name'): p.get('value') for p in nodes[0].findall('param')}
        self.assertEqual(params['start_step'], '9')
        self.assertEqual(params['lidar_type'], 'laserscan')
        self.assertEqual(params['lidar_topic'], '/scan_mid360_raw')
        self.assertEqual(params['tunnel_profile'], '$(arg tunnel_profile)')
        self.assertEqual(params['tunnel_config'], '$(find tb_tunnel_sim)/config/tunnel.yaml')
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
