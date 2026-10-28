"""Sensor driver configuration tests without hardware access."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class SensorLaunchTests(unittest.TestCase):
    def test_camera_stream_does_not_force_exposure_or_gain(self):
        launch = ET.parse(ROOT / 'launch/sensors.launch').getroot()
        camera = launch.find("include[@file='$(find realsense2_camera)/launch/rs_camera.launch']")
        self.assertIsNotNone(camera)
        arguments = {arg.get('name'): arg.get('value') for arg in camera.findall('arg')}
        for name, value in {'color_width': '640', 'color_height': '480',
                            'color_fps': '30', 'infra_rgb': 'true',
                            'publish_tf': 'true'}.items():
            self.assertEqual(arguments[name], value)
        overrides = {'enable_auto_exposure', 'exposure', 'gain'}
        for setting in list(launch.iter('param')) + list(launch.iter('arg')):
            self.assertNotIn(setting.get('name', '').rsplit('/', 1)[-1], overrides)

    def test_depth_is_optional_and_forwarded_to_the_camera(self):
        launch = ET.parse(ROOT / 'launch/sensors.launch').getroot()
        self.assertEqual(launch.find("arg[@name='enable_depth']").get('default'), 'false')
        camera = launch.find("include[@file='$(find realsense2_camera)/launch/rs_camera.launch']")
        self.assertEqual(camera.find("arg[@name='enable_depth']").get('value'),
                         '$(arg enable_depth)')
