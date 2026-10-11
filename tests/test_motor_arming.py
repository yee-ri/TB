"""Physical-start handshake tests with fake ROS, fake time and no hardware."""
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))
from tunnel import motor
import test_tunnel_wiring as wiring


class FakeROS:
    def __init__(self, fault=None):
        self.now = 0.
        self.fault = fault
        self.callbacks = {}
        self.command_messages = []
        self.power_messages = []
        self.torque = False
        self.cancelled = False
        self.graph_calls = 0
        self.events = []
        self.cmd = self.publisher('/cmd_vel')
        self.module = types.ModuleType('rospy')
        self.module.Publisher = lambda topic, kind, **kw: self.publisher(topic)
        self.module.Subscriber = self.subscribe
        self.module.get_name = lambda: '/tb_controller'
        self.module.is_shutdown = lambda: False
        self.module.get_param = lambda name, default: fault == 'sim_time'
        self.module.Time = types.SimpleNamespace(now=lambda: wiring.Stamp(self.now+100.))
        self.module.loginfo = Mock()

    def publisher(self, topic):
        def publish(message):
            if topic == '/cmd_vel':
                self.command_messages.append((self.now, message.linear.x, message.angular.z))
            else:
                self.events.append(('power',message.data))
                self.power_messages.append((self.now, message.data))
                if self.fault != 'no_ack':
                    self.torque = message.data
        peer = '/tb_trial_recorder' if self.fault == 'recorder_only' else motor.CORE
        return types.SimpleNamespace(publish=publish, unregister=Mock(),
            impl=types.SimpleNamespace(get_stats_info=lambda: [(1,peer,'o','TCPROS',topic,True)]))

    def subscribe(self, topic, kind, callback, **kwargs):
        self.callbacks[topic] = callback
        return types.SimpleNamespace(unregister=Mock())

    def sleep(self, duration):
        self.now += duration
        source = '/wrong_node' if self.fault == 'wrong_source' else motor.CORE
        sensor = types.SimpleNamespace(
            _connection_header={'callerid':source},button=1 if self.fault == 'button' else 0,
            torque=self.torque,
            header=types.SimpleNamespace(stamp=wiring.Stamp(self.now+100.-(1. if self.fault == 'stale' else 0.))))
        remote = wiring.Twist()
        remote._connection_header = {'callerid':source}
        if self.fault == 'remote':
            remote.linear.x = .1
        if '/sensor_state' in self.callbacks:
            self.callbacks['/sensor_state'](sensor)
            self.callbacks['/cmd_vel_rc100'](remote)

    def graph(self):
        self.graph_calls += 1
        if self.fault == 'cancel_before_request' and self.graph_calls == 2:
            self.cancelled = True
        if self.fault == 'stale_after_ack' and self.graph_calls == 3:
            self.now += .21
        if self.fault == 'off_after_ack' and self.graph_calls == 3:
            self.torque = False
            self.sleep(.05)
        pubs = [['/cmd_vel',['/tb_controller']],['/motor_power',['/tb_controller']],
                ['/sensor_state',[motor.CORE]],['/cmd_vel_rc100',[motor.CORE]]]
        if self.fault == 'other_publisher':
            pubs[0][1].append('/teleop')
        return pubs,[['/cmd_vel',[motor.CORE]],['/motor_power',[motor.CORE]]],[]

    def topic_types(self):
        return [['/cmd_vel','bad/type' if self.fault == 'type' else 'geometry_msgs/Twist'],
                ['/motor_power','std_msgs/Bool'],['/sensor_state','turtlebot3_msgs/SensorState'],
                ['/cmd_vel_rc100','geometry_msgs/Twist']]

    @contextmanager
    def installed(self):
        master = types.SimpleNamespace(getSystemState=self.graph,getTopicTypes=self.topic_types)
        class Bool:
            def __init__(self, data):
                self.data = data
        modules = {'rospy':self.module,
                   'rosgraph':wiring.module('rosgraph',Master=lambda name:master),
                   'geometry_msgs.msg':wiring.module('geometry_msgs.msg',Twist=wiring.Twist),
                   'std_msgs.msg':wiring.module('std_msgs.msg',Bool=Bool),
                   'turtlebot3_msgs.msg':wiring.module('turtlebot3_msgs.msg',SensorState=object)}
        with patch.dict(sys.modules,modules), patch.object(motor,'time',types.SimpleNamespace(
                monotonic=lambda:self.now,sleep=self.sleep)):
            yield


class MotorArmingTests(unittest.TestCase):
    def test_confirmed_start_uses_existing_zero_publisher_and_gates_both_adapter_paths(self):
        ros = FakeROS()
        with ros.installed():
            with motor.arm_motor_torque(ros.cmd,wiring.Twist,lambda:False,lambda:None,threading.RLock()):
                self.assertTrue(ros.torque)
                self.assertGreaterEqual(ros.now,.4-1e-12)
        self.assertEqual([value for _,value in ros.power_messages],[True])
        self.assertGreaterEqual(ros.power_messages[0][0],.3)
        self.assertTrue(all(v == 0 and w == 0 for _,v,w in ros.command_messages))

        fixture = wiring.TunnelWiringTests()
        fixture.setUp()
        try:
            for start_step in (8, 9):
                with self.subTest(start_step=start_step):
                    fixture.params.update({'~enable_motor_torque':True,'~start_step':start_step})
                    @contextmanager
                    def handshake(publisher,zero,cancelled,abort,lock):
                        node = fixture.subscribers[-1][1].__self__
                        self.assertEqual(node.step, start_step)
                        self.assertIsNone(node.tunnel_timer)
                        self.assertFalse(node.drive_ready)
                        node.move(.1,.2,tunnel=start_step == 9)
                        node.step = 0
                        node.move(.1,.2)
                        node.publish_velocity(10.,20.)
                        self.assertTrue(all(call.args[0].linear.x == call.args[0].angular.z == 0
                                            for call in publisher.publish.call_args_list))
                        node.step = start_step
                        yield
                        self.assertTrue(node.drive_ready)
                        self.assertIsNotNone(node.tunnel_timer)
                    with patch.object(fixture.target,'arm_motor_torque',handshake):
                        node = fixture.target.turtlebot()
                    self.assertEqual(node.step, start_step)
        finally:
            fixture.doCleanups()

    def test_failures_interruption_and_external_step_roll_back_without_enabling_motion(self):
        for fault in ('no_ack','stale','remote','button','wrong_source','other_publisher',
                      'type','sim_time','recorder_only','interrupt','cancel_after_activation',
                      'cancel_before_request','stale_after_ack','off_after_ack','abort_error'):
            with self.subTest(fault=fault):
                ros = FakeROS(fault)
                def abort():
                    ros.events.append(('gate',False))
                    if fault == 'abort_error':
                        raise RuntimeError('shutdown publisher failed after gate closed')
                with ros.installed():
                    with self.assertRaises((RuntimeError,KeyboardInterrupt)):
                        with motor.arm_motor_torque(ros.cmd,wiring.Twist,lambda:ros.cancelled,abort,threading.RLock()):
                            if fault == 'cancel_after_activation':
                                ros.cancelled = True
                            else:
                                raise KeyboardInterrupt()
                self.assertLessEqual(ros.now,2.85)
                self.assertFalse(ros.power_messages[-1][1])
                self.assertGreaterEqual(sum(not value for _,value in ros.power_messages),5)
                self.assertTrue(all(v == 0 and w == 0 for _,v,w in ros.command_messages))
                self.assertLess(ros.events.index(('gate',False)),ros.events.index(('power',False)))
                if fault == 'cancel_before_request':
                    self.assertFalse(any(value for _,value in ros.power_messages))

        fixture = wiring.TunnelWiringTests()
        fixture.setUp()
        try:
            fixture.params.update({'~enable_motor_torque':True,'~start_step':9})
            @contextmanager
            def cancelled_handshake(publisher,zero,cancelled,abort,lock):
                node = fixture.subscribers[-1][1].__self__
                node.step_callback(types.SimpleNamespace(data=3))
                self.assertTrue(cancelled())
                node.move(.2,.2)
                node.publish_velocity(10.,20.)
                node.shutdown()  # Must work before its timer exists.
                self.assertIsNone(node.tunnel_timer)
                self.assertFalse(node.drive_ready)
                self.assertTrue(all(call.args[0].linear.x == call.args[0].angular.z == 0
                                    for call in publisher.publish.call_args_list))
                raise RuntimeError('cancelled')
                yield
            with patch.object(fixture.target,'arm_motor_torque',cancelled_handshake):
                with self.assertRaises(RuntimeError):
                    fixture.target.turtlebot()
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
