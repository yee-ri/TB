"""Tunnel configuration/start guards; never connect to ROS or the drivetrain."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / 'codes/stepbystep/tunnel.real.yaml'
COMPETITION = ROOT / 'codes/stepbystep/tunnel.competition.yaml'
GAZEBO_MAP = ROOT / 'codes/stepbystep/maps/autorace_gazebo.pgm'


def load_cli():
    spec = importlib.util.spec_from_file_location('tb_nuc_trial_test', ROOT / 'scripts/nuc_trial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NucTrialSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli()
        temporary = tempfile.TemporaryDirectory(prefix='tb-tunnel-guard-test-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.practice = self.cli.read_config(PRACTICE)
        self.practice['grid']['static_map']['image'] = str(
            (PRACTICE.parent / self.practice['grid']['static_map']['image']).resolve())

    def config_file(self, config):
        path = self.directory / 'tunnel.yaml'
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        return path

    def assert_start_refused_before_runtime(self, path, reason):
        with patch.object(self.cli, 'current_trial') as trial, \
                patch.object(self.cli, 'graph') as graph, \
                patch.object(self.cli, 'check_remote') as remote, \
                patch.object(self.cli.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, reason):
                self.cli.start(path)
            trial.assert_not_called()
            graph.assert_not_called()
            remote.assert_not_called()
            process.assert_not_called()

    def test_practice_passes_without_physical_setup_declarations(self):
        config, map_path = self.cli.check_config(self.config_file(self.practice))
        self.assertNotIn('physical_setup', config)
        self.assertTrue(map_path.is_file())

    def test_static_valid_config_still_requires_active_recording(self):
        with patch.object(self.cli, 'current_trial', return_value=(self.directory, {'recording_active': False})), \
                patch.object(self.cli, 'graph') as graph, \
                patch.object(self.cli, 'check_remote') as remote, \
                patch.object(self.cli.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'active owned rosbag'):
                self.cli.start(self.config_file(self.practice))
            graph.assert_not_called()
            remote.assert_not_called()
            process.assert_not_called()

    def test_invalid_geometry_and_fast_profile_block_before_runtime(self):
        for section, key, value, reason in (
                ('footprint', 'front', None, 'footprint.front'),
                ('goal', 'pose', [None, 1, 90], 'goal.pose'),
                ('planner', 'keep_in_rectangles', [[-2, -2, 4, 4]], 'inside grid.bounds')):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.practice)
                config[section][key] = value
                self.assert_start_refused_before_runtime(self.config_file(config), reason)
        config = copy.deepcopy(self.practice)
        config['profiles']['fast'] = {}
        self.assert_start_refused_before_runtime(self.config_file(config), 'only profiles')

    def test_yaml_speeds_have_no_duplicate_launcher_ceiling(self):
        self.practice['control']['cruise_velocity'] = 0.2
        self.practice['entry']['cruise_velocity'] = 0.05
        config, _ = self.cli.check_config(self.config_file(self.practice))
        self.assertEqual(config['control']['cruise_velocity'], 0.2)
        self.assertEqual(config['entry']['cruise_velocity'], 0.05)

    def test_invalid_cruise_speeds_block_before_runtime(self):
        for section in ('control', 'entry'):
            for value in (0, -0.01, float('nan'), float('inf'), None, True, '0.2'):
                with self.subTest(section=section, value=value):
                    config = copy.deepcopy(self.practice)
                    config[section]['cruise_velocity'] = value
                    self.assert_start_refused_before_runtime(
                        self.config_file(config), section + r'\.cruise_velocity')

    def test_original_gazebo_map_is_rejected_even_if_renamed(self):
        copied = self.directory / 'renamed_physical_map.pgm'
        shutil.copyfile(GAZEBO_MAP, copied)
        for map_path in (GAZEBO_MAP, copied):
            with self.subTest(map_path=map_path):
                config = copy.deepcopy(self.practice)
                config['grid']['static_map']['image'] = str(map_path)
                self.assert_start_refused_before_runtime(self.config_file(config), 'original Gazebo wall map')

    def test_incomplete_competition_rejects_static_layout_and_start(self):
        config = self.cli.read_config(COMPETITION)
        self.assertIsNone(config['grid']['static_map']['image'])
        self.assertTrue(any(value is None for value in config['goal']['pose']))
        self.assert_start_refused_before_runtime(COMPETITION, 'goal.pose')

    def test_environment_selects_config_without_bypassing_missing_geometry(self):
        with patch.dict(os.environ, {'TB_TUNNEL_CONFIG': str(COMPETITION)}):
            cli = load_cli()
            self.assertEqual(cli.DEFAULT_CONFIG, COMPETITION)
            with self.assertRaisesRegex(ValueError, 'goal.pose'):
                cli.check_config(cli.DEFAULT_CONFIG)

    def start_metadata(self, path):
        (self.directory / 'source_sha256.json').write_text('{}')
        return {'recording_active': True,
                'config_sha256': self.cli.digest(path),
                'wall_map_sha256': self.cli.digest(Path(self.practice['grid']['static_map']['image']))}

    def test_live_sensor_failure_blocks_controller(self):
        path = self.config_file(self.practice)
        with patch.object(self.cli, 'current_trial', return_value=(self.directory, self.start_metadata(path))), \
                patch.object(self.cli, 'owned_process', return_value=123), \
                patch.object(self.cli, 'check_remote') as remote, \
                patch.object(self.cli, 'graph') as graph, \
                patch.object(self.cli.subprocess, 'run') as diagnostic, \
                patch.object(self.cli.subprocess, 'Popen') as process:
            diagnostic.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'Sensor/TF check failed'):
                self.cli.start(path)
            remote.assert_called_once_with(self.directory)
            graph.assert_not_called()
            process.assert_not_called()

    def test_other_cmd_vel_publisher_blocks_controller(self):
        path = self.config_file(self.practice)
        report = {'topics': {'/odom': {'age_at_receipt_seconds': {'max': 0.01},
                                      'age_of_last_message_now_seconds': 0.01}}}
        (self.directory / 'prestart_diagnostics.json').write_text(json.dumps(report))
        with patch.object(self.cli, 'current_trial', return_value=(self.directory, self.start_metadata(path))), \
                patch.object(self.cli, 'owned_process', return_value=123), \
                patch.object(self.cli, 'check_remote'), \
                patch.object(self.cli, 'graph', return_value={'publishers': {'/cmd_vel': ['/teleop']}}), \
                patch.object(self.cli.subprocess, 'run') as diagnostic, \
                patch.object(self.cli.subprocess, 'Popen') as process:
            diagnostic.return_value.returncode = 0
            with self.assertRaisesRegex(RuntimeError, 'Existing /cmd_vel publisher'):
                self.cli.start(path)
            process.assert_not_called()

    def test_physical_start_enables_torque_only_in_controller_after_preflight(self):
        path = self.config_file(self.practice)
        metadata = self.start_metadata(path)
        report = {'topics': {'/odom': {'age_at_receipt_seconds': {'max': 0.01},
                                      'age_of_last_message_now_seconds': 0.01}}}
        (self.directory / 'prestart_diagnostics.json').write_text(json.dumps(report))
        events = []
        process = Mock(pid=234, stdout=[], returncode=0)
        process.poll.return_value = 0
        def launch(command, **kwargs):
            events.append('controller')
            self.assertIn('_enable_motor_torque:=true', command)
            self.assertIn('_start_step:=9', command)
            self.assertIn('_tunnel_profile:=normal', command)
            return process
        def diagnostics(*args, **kwargs):
            events.append('diagnostics')
            return SimpleNamespace(returncode=0)
        state = {'publishers': {}, 'subscribers': {'/cmd_vel': ['/turtlebot3_core']}}
        with patch.object(self.cli, 'current_trial', return_value=(self.directory, metadata)), \
                patch.object(self.cli, 'owned_process', return_value=123), \
                patch.object(self.cli, 'check_remote', side_effect=lambda trial: events.append('remote')), \
                patch.object(self.cli, 'graph', return_value=state), \
                patch.object(self.cli.subprocess, 'run', side_effect=diagnostics), \
                patch.object(self.cli.subprocess, 'Popen', side_effect=launch), \
                patch.object(self.cli, 'process_identity', return_value={'pid': 234}), \
                patch.object(self.cli.signal, 'signal'), patch('builtins.print'):
            self.cli.start(path)
        self.assertEqual(events, ['remote', 'diagnostics', 'controller'])
        saved = json.loads((self.directory / 'trial.json').read_text())
        self.assertEqual(saved['controller_status'], 'exited')
        self.assertIn('_enable_motor_torque:=true', saved['controller_command'])


class TorqueStopTests(unittest.TestCase):
    """Mock the complete ROS boundary: no master, serial port, or hardware IO."""

    def setUp(self):
        self.cli = load_cli()
        self.state = {
            'topic_types': {'/motor_power': 'std_msgs/Bool', '/sensor_state': 'turtlebot3_msgs/SensorState'},
            'subscribers': {'/motor_power': ['/turtlebot3_core']},
            'publishers': {'/sensor_state': ['/turtlebot3_core']}}
        self.now = 100.0
        self.callback = None
        self.reply = {'callerid': '/turtlebot3_core', 'torque': False, 'stamp_offset': 0}
        self.publisher = Mock()
        self.publisher.get_num_connections.return_value = 1
        self.publisher.publish.side_effect = self.on_publish
        self.subscriber = Mock()
        self.rospy = SimpleNamespace(
            init_node=Mock(), signal_shutdown=Mock(),
            Publisher=Mock(return_value=self.publisher),
            Subscriber=Mock(side_effect=self.subscribe),
            Time=SimpleNamespace(now=lambda: self.stamp(self.now)))
        self.modules = {
            'rospy': self.rospy,
            'rosnode': SimpleNamespace(kill_nodes=Mock()),
            'std_msgs': SimpleNamespace(),
            'std_msgs.msg': SimpleNamespace(Bool=lambda data: SimpleNamespace(data=data)),
            'turtlebot3_msgs': SimpleNamespace(),
            'turtlebot3_msgs.msg': SimpleNamespace(SensorState=object)}
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules', self.modules).start()
        patch.object(self.cli.time, 'sleep').start()
        event = self.cli.threading.Event()
        # Messages are synchronously dispatched by the mock publisher. Avoid
        # real timeout waits without changing how set()/clear() are tested.
        event.wait = lambda timeout: event.is_set()
        patch.object(self.cli.threading, 'Event', return_value=event).start()

    @staticmethod
    def stamp(value):
        return SimpleNamespace(to_sec=lambda: value)

    def subscribe(self, topic, message_type, callback, **kwargs):
        self.assertEqual(topic, '/sensor_state')
        self.callback = callback
        return self.subscriber

    def on_publish(self, message):
        self.assertIs(message.data, False)
        self.now += .05
        if self.callback and self.reply is not None:
            self.callback(SimpleNamespace(
                _connection_header={'callerid': self.reply['callerid']},
                torque=self.reply['torque'],
                header=SimpleNamespace(stamp=self.stamp(self.now + self.reply['stamp_offset']))))

    def test_stop_reports_feedback_and_is_repeatable_without_controller(self):
        with patch.object(self.cli, 'graph', return_value=self.state), patch('builtins.print') as output:
            self.cli.stop()
            self.cli.stop()
        self.assertEqual(self.publisher.publish.call_count, 2)
        self.modules['rosnode'].kill_nodes.assert_not_called()
        text = '\n'.join(str(call) for call in output.call_args_list)
        self.assertIn('OpenCR reports torque=false', text)
        self.assertIn('not physical braking', text)
        self.assertIn('Keep the PC on', text)
        self.rospy.Publisher.assert_called_with('/motor_power', self.modules['std_msgs.msg'].Bool, queue_size=1, latch=False)

    def test_missing_motor_power_input_fails_without_publishing(self):
        self.state['subscribers'] = {}
        with self.assertRaisesRegex(RuntimeError, 'NOT requested'):
            self.cli.request_torque_disable(self.state)
        self.rospy.Publisher.assert_not_called()

    def test_controller_shutdown_precedes_torque_disable(self):
        active = copy.deepcopy(self.state)
        active['publishers']['/cmd_vel'] = ['/tb_controller']
        order = []

        def shutdown(nodes):
            order.append('controller shutdown')
            self.assertEqual(nodes, ['/tb_controller'])
            return nodes, []

        self.modules['rosnode'].kill_nodes.side_effect = shutdown
        with patch.object(self.cli, 'graph', side_effect=[active, self.state]), \
                patch.object(self.cli, 'request_torque_disable', side_effect=lambda state: order.append('torque disable')), \
                patch('builtins.print'):
            self.cli.stop()
        self.assertEqual(order, ['controller shutdown', 'torque disable'])

    def test_missing_feedback_still_requests_disable_but_fails(self):
        self.state['publishers'] = {}
        with self.assertRaisesRegex(RuntimeError, 'feedback is unavailable'):
            self.cli.request_torque_disable(self.state)
        self.publisher.publish.assert_called_once()
        self.publisher.unregister.assert_called_once()
        self.rospy.Subscriber.assert_not_called()

    def test_no_feedback_is_not_a_success(self):
        self.reply = None
        with self.assertRaisesRegex(RuntimeError, 'no fresh torque=false'):
            self.cli.request_torque_disable(self.state)
        self.publisher.unregister.assert_called_once()
        self.subscriber.unregister.assert_called_once()

    def test_wrong_caller_stale_future_and_torque_on_are_ignored(self):
        for changes in (
                {'callerid': '/unrelated_node'},
                {'stamp_offset': -1},
                {'stamp_offset': 1},
                {'stamp_offset': float('nan')},
                {'torque': True}):
            with self.subTest(changes=changes):
                self.reply = dict(callerid='/turtlebot3_core', torque=False, stamp_offset=0)
                self.reply.update(changes)
                with self.assertRaisesRegex(RuntimeError, 'no fresh torque=false'):
                    self.cli.request_torque_disable(self.state)


if __name__ == '__main__':
    unittest.main()
