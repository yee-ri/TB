"""Version selection without ROS, publishers, or robot motion."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class TunnelVersionTests(unittest.TestCase):
    def selector(self, directory):
        name = 'version_test_' + directory
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / 'codes' / directory / 'tunnel')]
        original = types.ModuleType(name + '.mission')
        original.TunnelMission = Mock()
        experimental = types.ModuleType(name + '.mission_v2')
        experimental.TunnelMissionV2 = Mock()
        modules = {name: package, original.__name__: original,
                   experimental.__name__: experimental}
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)
        spec = importlib.util.spec_from_file_location(
            name + '.versions', Path(package.__path__[0]) / 'versions.py')
        selected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selected)
        return selected, original, experimental

    def test_default_uses_original_without_importing_v2(self):
        for directory in ('stepbystep', 'final_codes'):
            with self.subTest(directory=directory):
                selector, original, experimental = self.selector(directory)
                config, clock = {'sensors': {'maximum_age': .3}}, object()
                with patch.dict(sys.modules, {experimental.__name__: None}):
                    result = selector.create_mission(config, clock=clock)
                self.assertIs(result, original.TunnelMission.return_value)
                original.TunnelMission.assert_called_once_with(config, clock=clock)
                experimental.TunnelMissionV2.assert_not_called()

    def test_v2_is_explicit_and_receives_the_same_config_and_clock(self):
        for directory in ('stepbystep', 'final_codes'):
            with self.subTest(directory=directory):
                selector, original, experimental = self.selector(directory)
                config, clock = {'sensors': {'maximum_age': .3}}, object()
                result = selector.create_mission(config, version='v2', clock=clock)
                self.assertIs(result, experimental.TunnelMissionV2.return_value)
                experimental.TunnelMissionV2.assert_called_once_with(config, clock=clock)
                original.TunnelMission.assert_not_called()

    def test_invalid_choice_never_constructs_either_version(self):
        for directory in ('stepbystep', 'final_codes'):
            selector, original, experimental = self.selector(directory)
            for version in ('V2', 'v3', '', None, 2, True):
                with self.subTest(directory=directory, version=version):
                    with self.assertRaisesRegex(ValueError, 'tunnel_version must be v1 or v2'):
                        selector.create_mission({}, version=version)
            original.TunnelMission.assert_not_called()
            experimental.TunnelMissionV2.assert_not_called()


if __name__ == '__main__':
    unittest.main()
