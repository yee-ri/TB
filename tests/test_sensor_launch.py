"""Sensor driver configuration tests without hardware access."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class SensorLaunchTests(unittest.TestCase):
    def test_camera_stream_and_verified_manual_exposure(self):
        launch = ET.parse(ROOT / 'launch/sensors.launch').getroot()
        camera = launch.find("include[@file='$(find realsense2_camera)/launch/rs_camera.launch']")
        self.assertIsNotNone(camera)
        arguments = {arg.get('name'): arg.get('value') for arg in camera.findall('arg')}
        for name, value in {'color_width': '640', 'color_height': '480',
                            'color_fps': '30', 'infra_rgb': 'true',
                            'publish_tf': 'true'}.items():
            self.assertEqual(arguments[name], value)
        parameters = {param.get('name'): param.get('value') for param in launch.iter('param')}
        for name, value in {'enable_auto_exposure': 'false', 'exposure': '20000',
                            'gain': '16'}.items():
            self.assertEqual(parameters['camera/stereo_module/'+name], value)

    def test_depth_is_optional_and_forwarded_to_the_camera(self):
        launch = ET.parse(ROOT / 'launch/sensors.launch').getroot()
        self.assertEqual(launch.find("arg[@name='enable_depth']").get('default'), 'false')
        camera = launch.find("include[@file='$(find realsense2_camera)/launch/rs_camera.launch']")
        self.assertEqual(camera.find("arg[@name='enable_depth']").get('value'),
                         '$(arg enable_depth)')
