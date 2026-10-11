"""TB adapter tests with mocked ROS transport; not a robot-driving test."""
import importlib.util
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2  # Load native bindings before per-test sys.modules restoration.


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "codes/stepbystep/try_maze.py"


class Stamp:
    def __init__(self, seconds=0.0):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds

    @classmethod
    def from_sec(cls, seconds):
        return cls(seconds)

    @classmethod
    def now(cls):
        return cls(42.0)


def vector(x=0.0, y=0.0, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z)


def quaternion(yaw=0.0):
    return types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def transform(x=0.0, y=0.0, yaw=0.0, z=0.0):
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=vector(x, y, z), rotation=quaternion(yaw)))


def quaternion_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0],
        [0, 0, 0, 1],
    ])


class Twist:
    def __init__(self):
        self.linear, self.angular = vector(), vector()


class Mission:
    def __init__(self, config, clock=None):
        self.config = config
        self.clock = clock
        self.state, self.reason = "ENTRY", "test"
        self.anchor = None
        self.reset = Mock()
        self.close = Mock()
        self.step = Mock(return_value=(0.06, 0.02, False))
        self.update_odometry = Mock()
        self.update_cloud = Mock()
        self.update_scan = Mock()
        self.update_lane = Mock()


def module(name, **members):
    result = types.ModuleType(name)
    result.__dict__.update(members)
    return result


class TunnelWiringTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "control": {"period": 0.05},
            "sensors": {"base_frame": "base_footprint", "z_min": 0.03,
                        "z_max": 0.30, "range_max": 4.0, "minimum_lane_pixels": 30},
        }
        self.publishers, self.subscribers, self.timers = [], [], []
        self.params = {}
        self.tf_buffer = types.SimpleNamespace(lookup_transform=Mock(return_value=transform()))

        def publisher(topic, message_type, **kwargs):
            value = types.SimpleNamespace(topic=topic, publish=Mock())
            self.publishers.append(value)
            return value

        def subscriber(topic, message_type, callback, **kwargs):
            # A subscriber may execute a callback immediately when constructed.
            node = callback.__self__
            for field in ("step", "bridge", "tf_buffer", "tunnel", "command_lock",
                          "odom_frame", "initial_x", "gate_state", "turn_direction"):
                self.assertTrue(hasattr(node, field), field)
            self.subscribers.append((topic, callback, kwargs, message_type))
            return object()

        def timer(period, callback):
            self.timers.append((period.to_sec(), callback))
            return types.SimpleNamespace(shutdown=Mock())

        rospy = module("rospy", init_node=Mock(), Publisher=publisher,
                       Subscriber=subscriber, Timer=timer, Time=Stamp, Duration=Stamp,
                       get_param=lambda name, default: self.params.get(name, default),
                       on_shutdown=Mock(), loginfo=Mock(), logwarn=Mock(),
                       loginfo_throttle=Mock(), logwarn_throttle=Mock())
        tf_math = module("tf.transformations", euler_from_quaternion=lambda q:
                         (0.0, 0.0, 2.0 * math.atan2(q[2], q[3])), quaternion_matrix=quaternion_matrix)
        cloud_reader = module("sensor_msgs.point_cloud2", read_points=lambda cloud, **kwargs: cloud.points)
        def transform_to_kdl(tf):
            q = tf.transform.rotation
            return types.SimpleNamespace(M=quaternion_matrix((q.x,q.y,q.z,q.w))[:3,:3])
        cloud_tf = module("tf2_sensor_msgs.tf2_sensor_msgs", do_transform_cloud=lambda cloud, tf: cloud,
                          transform_to_kdl=transform_to_kdl)
        self.load_config = Mock(return_value=self.config)
        mission = module("tunnel.mission", TunnelMission=Mission, load_config=self.load_config)
        modules = {
            "rospy": rospy,
            "tf": module("tf", transformations=tf_math), "tf.transformations": tf_math,
            "sensor_msgs": module("sensor_msgs", point_cloud2=cloud_reader),
            "sensor_msgs.msg": module("sensor_msgs.msg", Image=object,
                                      PointCloud2=type("PointCloud2", (), {}), LaserScan=type("LaserScan", (), {})),
            "sensor_msgs.point_cloud2": cloud_reader,
            "geometry_msgs": module("geometry_msgs"),
            "geometry_msgs.msg": module("geometry_msgs.msg", Twist=Twist),
            "nav_msgs": module("nav_msgs"), "nav_msgs.msg": module("nav_msgs.msg", Odometry=object),
            "std_msgs": module("std_msgs"), "std_msgs.msg": module("std_msgs.msg", UInt8=object),
            "cv_bridge": module("cv_bridge", CvBridge=lambda: types.SimpleNamespace(
                imgmsg_to_cv2=lambda data, *args, **kwargs: data.array)),
            "tf2_ros": module("tf2_ros", Buffer=lambda: self.tf_buffer, TransformListener=lambda _: object()),
            "tf2_sensor_msgs": module("tf2_sensor_msgs", tf2_sensor_msgs=cloud_tf),
            "tf2_sensor_msgs.tf2_sensor_msgs": cloud_tf,
            "tunnel": module("tunnel", mission=mission,
                             __path__=[str(ROOT / 'codes/stepbystep/tunnel')]), "tunnel.mission": mission,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("tb_wiring_test_target", SOURCE)
        self.target = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.target)
        self.node = self.target.turtlebot()

    def image(self, visible=False):
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)
        if visible:
            pixels[320:460, 100:110] = (0, 255, 255)
            pixels[320:460, 530:540] = (255, 255, 255)
        return types.SimpleNamespace(array=pixels, header=types.SimpleNamespace(stamp=Stamp(41.9)))

    def odometry(self, x=2.0, y=2.0, yaw=math.pi / 2):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="odom", stamp=Stamp(41.8)),
            pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=vector(x, y), orientation=quaternion(yaw))),
            twist=types.SimpleNamespace(twist=types.SimpleNamespace(linear=vector(0.12), angular=vector(z=0.3))),
        )

    def cloud(self, points):
        return types.SimpleNamespace(points=points,
            header=types.SimpleNamespace(frame_id="livox_frame", stamp=Stamp(41.7)))

    def timed_cloud(self, points, offsets):
        message = self.cloud(points)
        message.fields = [types.SimpleNamespace(name=name, offset=offset, datatype=kind, count=1)
                          for name, offset, kind in [('x',0,7), ('y',4,7), ('z',8,7),
                                                     ('intensity',12,7), ('tag',16,2),
                                                     ('line',17,2), ('timestamp',18,8)]]
        message.is_bigendian = False
        message.point_step = 26
        message.width, message.height = len(points), 1
        message.row_step = message.width * message.point_step
        message.data = b''.join(struct.pack('<ffffBBd', *point, 15., 0, 0,
                                           (41.7 + offset) * 1e9)
                                for point, offset in zip(points, offsets))
        return message

    def laser(self, ranges, angle_min=0.0, angle_increment=math.pi / 2):
        return types.SimpleNamespace(ranges=ranges, angle_min=angle_min,
            angle_increment=angle_increment, range_min=0.05, range_max=4.0,
            header=types.SimpleNamespace(frame_id="base_scan", stamp=Stamp(41.7)))

    def test_startup_initializes_before_callbacks_and_uses_one_publisher(self):
        self.assertEqual(self.node.step, 0)
        self.assertEqual([p.topic for p in self.publishers], ["/cmd_vel"])
        self.assertEqual(len(self.subscribers), 5)
        self.assertEqual(self.node.lidar_type, "pointcloud2")
        self.assertEqual(self.subscribers[2][0], "/livox/lidar")
        self.assertEqual(self.subscribers[2][3].__name__, "PointCloud2")
        self.assertEqual(self.subscribers[2][2], {'queue_size': 1, 'buff_size': 4*1024*1024})
        self.assertEqual(self.timers[0][0], 0.05)
        self.assertEqual(self.node.tunnel.clock(), 42.0)
        self.assertEqual(self.node.lturn_template.shape, (100, 100))
        self.assertEqual(self.node.rturn_template.shape, (100, 100))

    def test_explicit_start_step_is_configurable(self):
        self.params["~start_step"] = 9
        self.assertEqual(self.target.turtlebot().step, 9)

    def test_normal_profile_is_default_and_fast_uses_same_single_publisher(self):
        self.assertEqual(self.load_config.call_args.args[1], 'normal')
        for profile in ('normal', 'fast'):
            with self.subTest(profile=profile):
                self.params['~tunnel_profile'] = profile
                before = len(self.publishers)
                node = self.target.turtlebot()
                self.assertEqual(self.load_config.call_args.args[1], profile)
                self.assertEqual([p.topic for p in self.publishers[before:]], ['/cmd_vel'])
                node.step = 9
                node.tunnel_tick(None)
                node.tunnel.step.assert_called_once_with(42.0)
                self.assertEqual(node.cmd_pub.publish.call_count, 1)

    def test_invalid_profile_raises_before_creating_a_command_publisher(self):
        self.params['~tunnel_profile'] = 'typo-fast'
        self.load_config.side_effect = ValueError('unknown tunnel profile')
        before = len(self.publishers)
        with self.assertRaisesRegex(ValueError, 'profile'):
            self.target.turtlebot()
        self.assertEqual(len(self.publishers), before)

    def test_laserscan_input_selects_one_subscriber_with_topic_override(self):
        self.params.update({"~lidar_type": "laserscan", "~lidar_topic": "/sim/scan"})
        self.target.turtlebot()
        scan = self.subscribers[-3]
        self.assertEqual(scan[0], "/sim/scan")
        self.assertEqual(scan[3].__name__, "LaserScan")
        self.assertEqual(scan[1].__name__, "laser_callback")

    def test_laserscan_default_topic_and_invalid_type(self):
        self.params["~lidar_type"] = "laserscan"
        self.assertEqual(self.target.turtlebot().lidar_topic, "/scan_mid360_raw")
        self.params["~lidar_type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "lidar_type"):
            self.target.turtlebot()

    def test_timer_does_not_touch_non_tunnel_missions(self):
        for step in (0, 1, 3, 5, 8, 11):
            self.node.step = step
            self.node.tunnel_tick(None)
        self.node.tunnel.step.assert_not_called()
        self.node.cmd_pub.publish.assert_not_called()

    def test_timer_drives_entry_without_a_camera_or_blind_advance(self):
        self.node.step = 9
        self.node.move_time_yaw_hold = Mock(side_effect=AssertionError("blind advance"))
        self.node.tunnel_tick(None)
        self.node.tunnel.reset.assert_called_once()
        self.node.tunnel.step.assert_called_once_with(42.0)
        cmd = self.node.cmd_pub.publish.call_args.args[0]
        self.assertAlmostEqual(cmd.linear.x, 0.06)
        self.assertEqual(self.node.step, 9)
        self.node.tunnel.state = "FOLLOWING"
        self.node.tunnel_tick(None)
        self.assertEqual(self.node.step, 10)
        self.node.tunnel.reset.assert_called_once()

    def test_tunnel_images_supply_pixels_only_and_do_not_publish(self):
        self.node.lane_tracking = Mock(side_effect=AssertionError("lane command during tunnel"))
        for step in (9, 10):
            self.node.step = step
            self.node.img_callback(self.image(True))
            self.node.tunnel.update_lane.assert_called_with(True, 41.9)
            self.node.img_callback(self.image(False))
            self.node.tunnel.update_lane.assert_called_with(False, 41.9)
        self.node.cmd_pub.publish.assert_not_called()

    def test_lane_detection_requires_both_real_lines(self):
        image = self.image(True).array
        self.assertTrue(self.node.lane_detect(image))
        image[:, 530:540] = 0
        self.assertFalse(self.node.lane_detect(image))
        self.assertFalse(self.node.lane_detect(image[:100]))

    def test_entry_transition_frame_does_not_emit_lane_command(self):
        self.node.step, self.node.maze_count = 8, 9
        self.node.lane_tracking = Mock(side_effect=AssertionError("transition command"))
        self.node.img_callback(self.image(False))
        self.assertEqual(self.node.step, 8.5)
        self.assertEqual(self.node.tunnel_generation, 0)
        self.node.cmd_pub.publish.assert_not_called()

    def test_timed_approach_finishes_before_tunnel_activation(self):
        self.node.step = 8.5
        self.node.move_time_yaw_hold = Mock()
        self.node.lane_tracking = Mock(side_effect=AssertionError('lane during handoff'))
        self.node.img_callback(self.image(False))
        self.node.move_time_yaw_hold.assert_called_once_with(.06, -1.565, 2.5)
        self.assertEqual(self.node.step, 9)
        self.assertEqual(self.node.tunnel_generation, 1)
        self.assertFalse(self.node.tunnel_active)
        command = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., 0.))

    def test_fresh_step8_captures_lane_end_heading_in_rebased_odometry(self):
        for raw_yaw, lane_turn in ((0., 0.), (1.2, 0.), (1.2, .3)):
            with self.subTest(raw_yaw=raw_yaw, lane_turn=lane_turn):
                self.params['~start_step'] = 8
                node = self.target.turtlebot()
                node.odom_callback(self.odometry(yaw=raw_yaw))
                self.assertAlmostEqual(node.yaw, 0.)
                node.lane_tracking = Mock()
                node.move_time_yaw_hold = Mock()
                node.img_callback(self.image(True))
                node.odom_callback(self.odometry(yaw=raw_yaw + lane_turn))
                for _ in range(9):
                    node.img_callback(self.image(False))
                self.assertEqual(node.step, 8)
                node.img_callback(self.image(False))
                self.assertEqual(node.step, 8.5)
                self.assertAlmostEqual(node.gate_approach_yaw, lane_turn)
                # A later pose update must not change the captured target.
                node.odom_callback(self.odometry(yaw=raw_yaw + lane_turn + .1))
                node.img_callback(self.image(False))
                args = node.move_time_yaw_hold.call_args.args
                self.assertEqual((args[0], args[2]), (.06, 2.5))
                self.assertAlmostEqual(args[1], lane_turn)
                self.assertEqual(node.step, 9)
                self.assertFalse(node.tunnel_active)

    def test_fresh_step8_waits_for_first_odometry_and_motor_startup(self):
        for odom_received, drive_ready in ((False, True), (True, False)):
            with self.subTest(odom_received=odom_received, drive_ready=drive_ready):
                self.params['~start_step'] = 8
                node = self.target.turtlebot()
                if odom_received:
                    node.odom_callback(self.odometry())
                node.drive_ready = drive_ready
                node.lane_tracking = Mock()
                for _ in range(11):
                    node.img_callback(self.image(False))
                self.assertEqual(node.step, 8)
                self.assertEqual(node.maze_count, 0)
                node.lane_tracking.assert_not_called()
                self.assertTrue(all(call.args[0].linear.x == call.args[0].angular.z == 0
                                    for call in node.cmd_pub.publish.call_args_list))
                node.odom_callback(self.odometry())
                node.drive_ready = True
                node.img_callback(self.image(True))
                node.lane_tracking.assert_called_once()

    def test_step8_lane_drives_over_headless_ssh_without_opening_debug_window(self):
        self.params['~start_step'] = 8
        node = self.target.turtlebot()
        node.odom_callback(self.odometry())
        with patch.dict(self.target.os.environ, {}, clear=True), \
                patch.object(self.target.cv2, 'imshow') as show, \
                patch.object(self.target.cv2, 'waitKey') as wait:
            node.img_callback(self.image(True))
        show.assert_not_called()
        wait.assert_not_called()
        command = node.cmd_pub.publish.call_args.args[0]
        self.assertGreater(command.linear.x, 0.)
        self.assertEqual(node.step, 8)

    def test_lane_debug_window_is_explicit_opt_in(self):
        self.params.update({'~start_step': 8, '~show_lane_debug': True})
        node = self.target.turtlebot()
        node.odom_callback(self.odometry())
        with patch.object(self.target.cv2, 'imshow') as show, \
                patch.object(self.target.cv2, 'waitKey') as wait:
            node.img_callback(self.image(True))
        show.assert_called_once()
        wait.assert_called_once_with(1)

    def test_upstream_lane_speeds_and_turn_limits_reach_existing_publisher(self):
        centered = self.image()
        centered.array[320:460, 96:105] = (0, 255, 255)
        centered.array[320:460, 536:545] = (255, 255, 255)
        far_right = self.image()
        far_right.array[320:460, 620:630] = (0, 255, 255)
        for step, speed, forward, turn_limit in ((8, 0, .1045, .629),
                                                (11, 0, .1045, .629),
                                                (7, 1, .06175, .333),
                                                (3, 2, .09025, .592),
                                                (2, 0, .08075, .629)):
            with self.subTest(step=step, speed=speed):
                node = self.target.turtlebot()
                node.step, node.speed = step, speed
                node.lane_tracking(centered.array)
                cmd = node.cmd_pub.publish.call_args.args[0]
                self.assertAlmostEqual(cmd.linear.x, forward)
                self.assertAlmostEqual(cmd.angular.z, 0.)
                node.prev_target_x = None
                node.prev_error = 0.
                node.lane_tracking(far_right.array)
                cmd = node.cmd_pub.publish.call_args.args[0]
                self.assertAlmostEqual(cmd.linear.x, forward)
                self.assertAlmostEqual(cmd.angular.z, -turn_limit)

    def test_gate_approach_slows_then_releases_speed_state_on_open(self):
        node = self.node
        node.step = 7
        node.lane_tracking = Mock()
        image = self.image()
        image.array[0:20, 0:50] = (0, 0, 150)
        node.img_callback(image)
        self.assertEqual(node.speed, 1)
        node.gate_state, node.gate_open_count = 1, 19
        with patch.object(self.target.rospy, 'sleep', Mock(), create=True):
            node.img_callback(self.image())
        self.assertEqual(node.step, 8)
        self.assertEqual(node.speed, 0)
        node.speed = 1
        node.step_callback(types.SimpleNamespace(data=8))
        self.assertEqual(node.speed, 0)

    def test_obstacle_lane_speed_state_resets_after_clearance(self):
        node = self.node
        node.step, node.obstacle_move_count = 3, 20
        node.check_obstacle_start = Mock(return_value=False)
        node.check_obstacle_end = Mock(return_value=False)
        node.lane_tracking = Mock()
        node.img_callback(self.image(True))
        self.assertEqual(node.speed, 2)
        node.check_obstacle_end.return_value = True
        node.img_callback(self.image(True))
        self.assertEqual(node.speed, 0)

    def test_wall_alignment_stays_on_step_nine_with_zero_command(self):
        self.node.step = 9
        self.node.tunnel.state = 'LOCALIZING'
        self.node.tunnel.step.return_value = (0., 0., False)
        self.node.tunnel_tick(None)
        self.assertEqual(self.node.step, 9)
        command = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., 0.))

    def test_physical_environment_config_is_used_for_full_course_too(self):
        with patch.dict(self.target.os.environ, {'TB_TUNNEL_CONFIG': '/survey/tunnel.real.yaml'}):
            self.target.turtlebot()
        self.assertEqual(self.load_config.call_args.args[0], '/survey/tunnel.real.yaml')

    def test_explicit_ros_config_overrides_environment(self):
        self.params['~tunnel_config'] = '/simulation/tunnel.yaml'
        with patch.dict(self.target.os.environ, {'TB_TUNNEL_CONFIG': '/survey/tunnel.real.yaml'}):
            self.target.turtlebot()
        self.assertEqual(self.load_config.call_args.args[0], '/simulation/tunnel.yaml')

    def test_completion_releases_to_lane_on_next_image(self):
        self.node.step = 10
        self.node.prev_error = 100.0
        self.node.prev_target_x = 480.0
        self.node.prev_v_l, self.node.prev_v_r = 25.0, 19.0
        self.node.tunnel.step.return_value = (0.0, 0.0, True)
        self.node.tunnel.state = "COMPLETE"
        self.node.tunnel_tick(None)
        self.assertEqual(self.node.step, 11)
        self.assertFalse(self.node.tunnel_active)
        self.assertEqual(self.node.prev_error, 0.0)
        self.assertIsNone(self.node.prev_target_x)
        self.assertEqual((self.node.prev_v_l, self.node.prev_v_r), (0.0, 0.0))
        self.node.tunnel.update_lane.assert_not_called()
        self.node.cmd_pub.publish.assert_called_once()
        command = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0.0, 0.0))

        # No image means no new lane command; a blank image must not replay
        # the wheel command stored before entering the tunnel.
        self.node.tunnel_tick(None)
        self.node.cmd_pub.publish.assert_called_once()
        for _ in range(4):
            self.node.img_callback(self.image(False))
            command = self.node.cmd_pub.publish.call_args.args[0]
            self.assertEqual((command.linear.x, command.angular.z), (0.0, 0.0))
        self.node.tunnel.update_lane.assert_not_called()

    def test_first_single_line_image_drives_after_stopped_tunnel_handoff(self):
        for color, removed_columns in (("yellow", slice(530, 540)),
                                       ("white", slice(100, 110))):
            with self.subTest(color=color):
                node = self.target.turtlebot()
                node.step = 10
                node.tunnel.state = "COMPLETE"
                node.tunnel.step.return_value = (0.0, 0.0, True)
                node.tunnel_tick(None)
                self.assertEqual(node.step, 11)
                node.cmd_pub.publish.reset_mock()

                image = self.image(True)
                image.array[:, removed_columns] = 0
                self.assertFalse(node.lane_detect(image.array))
                node.img_callback(image)

                # The ordinary lane controller can use either color alone
                # immediately; there is no three-frame, two-line gate here.
                node.cmd_pub.publish.assert_called_once()
                command = node.cmd_pub.publish.call_args.args[0]
                self.assertGreater(command.linear.x, 0.0)
                self.assertEqual(node.step, 11)
                node.tunnel.update_lane.assert_not_called()

    def test_external_step_change_cancels_pending_timer_command(self):
        self.node.step = 10

        def change_step(_now):
            self.node.step_callback(types.SimpleNamespace(data=11))
            return 0.2, 0.5, False

        self.node.tunnel.step.side_effect = change_step
        self.node.tunnel_tick(None)
        self.assertEqual(self.node.step, 11)
        self.node.cmd_pub.publish.assert_called_once()
        cmd = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0.0, 0.0))

    def test_shutdown_stops_existing_publisher_and_rejects_later_commands(self):
        self.node.step = 10
        self.node.shutdown()
        self.node.tunnel_timer.shutdown.assert_called_once()
        self.node.tunnel.close.assert_called_once()
        self.node.shutdown()
        self.node.tunnel_tick(None)
        self.node.move(0.2, 0.3, tunnel=True)
        self.node.step = 11
        self.node.move(0.2, 0.3)
        self.node.publish_velocity(10.0, 20.0)
        self.node.tunnel.step.assert_not_called()
        self.node.cmd_pub.publish.assert_called_once()
        cmd = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0.0, 0.0))

    def test_shutdown_cancels_inflight_timer_command(self):
        self.node.step = 10

        def shutdown_inflight(_now):
            self.node.shutdown()
            return 0.2, 0.5, False

        self.node.tunnel.step.side_effect = shutdown_inflight
        self.node.tunnel_tick(None)
        self.node.cmd_pub.publish.assert_called_once()
        cmd = self.node.cmd_pub.publish.call_args.args[0]
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0.0, 0.0))

    def test_old_lane_or_mission_command_cannot_override_tunnel(self):
        self.node.step = 10
        self.node.move(0.5, 1.0)
        self.node.stop()
        self.node.publish_velocity(10.0, 20.0)
        self.node.cmd_pub.publish.assert_not_called()
        self.node.move(0.1, 0.2, tunnel=True)
        self.node.cmd_pub.publish.assert_called_once()

    def test_odometry_preserves_tb_rebased_coordinates_stamp_and_twist(self):
        self.node.odom_callback(self.odometry())
        self.node.odom_callback(self.odometry(y=3.0))
        pose, v, w, stamp = self.node.tunnel.update_odometry.call_args.args
        np.testing.assert_allclose(pose, (1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual((v, w, stamp), (0.12, 0.3, 41.8))

    def test_cloud_uses_acquisition_tf_and_preserves_rear_side_points(self):
        self.node.step = 9
        self.node.odom_callback(self.odometry())
        self.tf_buffer.lookup_transform.side_effect = [transform(-0.033, 0.0), transform(2.0, 3.0, math.pi / 2)]
        cloud = self.cloud([(0.5, 0.0, 0.1), (-0.5, 0.0, 0.1), (0.0, 0.6, 0.1),
                            (0.5, 0.0, 0.01), (0.5, 0.0, 0.5), (float("nan"), 0.0, 0.1)])
        self.node.scan_callback(cloud)
        calls = self.tf_buffer.lookup_transform.call_args_list
        self.assertEqual(calls[0].args[:3], ("base_footprint", "livox_frame", cloud.header.stamp))
        self.assertEqual(calls[1].args[:3], ("odom", "base_footprint", cloud.header.stamp))
        points, sensor, pose, stamp = self.node.tunnel.update_cloud.call_args.args
        self.assertEqual(points, [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.6)])
        np.testing.assert_allclose(pose, (1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual(sensor, (-0.033, 0.0, 0.0))
        self.assertEqual(stamp, 41.7)
        self.node.cmd_pub.publish.assert_not_called()

    def test_missing_acquisition_tf_does_not_forge_a_fresh_scan(self):
        self.node.step = 10
        self.node.odom_callback(self.odometry())
        self.tf_buffer.lookup_transform.side_effect = [transform(), RuntimeError("missing")]
        self.node.scan_callback(self.cloud([(0.5, 0.0, 0.1)]))
        self.node.tunnel.update_cloud.assert_not_called()

    def test_cloud_transform_failure_keeps_previous_sensor_stamp(self):
        self.node.step = 10
        with patch.object(self.target.tf2_sensor_msgs.tf2_sensor_msgs, "do_transform_cloud",
                          side_effect=RuntimeError("invalid cloud")):
            self.node.scan_callback(self.cloud([(0.5, 0.0, 0.1)]))
        self.node.tunnel.update_cloud.assert_not_called()

    def test_float32_cloud_fast_path_preserves_acquisition_pose_without_reserializing(self):
        self.node.step = 9
        self.node.odom_callback(self.odometry())
        self.tf_buffer.lookup_transform.side_effect = [transform(-0.033, 0., z=.19),
                                                      transform(2.,3.,math.pi/2)]
        message = self.cloud([(0.5,0.,0.),(-0.5,0.,0.),(0.,.6,0.),
                              (.5,0.,.5),(float('nan'),0.,0.)])
        message.fields = [types.SimpleNamespace(name=name,offset=offset,datatype=7,count=1)
                          for name,offset in [('x',0),('y',4),('z',8)]]
        message.is_bigendian = False
        message.point_step = 12
        message.width, message.height = len(message.points), 1
        message.row_step = message.width*12
        with patch.object(self.target.tf2_sensor_msgs.tf2_sensor_msgs, 'do_transform_cloud') as old:
            self.node.scan_callback(message)
        old.assert_not_called()
        points,sensor,pose,stamp = self.node.tunnel.update_cloud.call_args.args
        np.testing.assert_allclose(points, [[.467,0.],[-.533,0.],[-.033,.6]],atol=1e-7)
        self.assertEqual(sensor,(-.033,0.,0.))
        np.testing.assert_allclose(pose,(1.,0.,0.),atol=1e-12)
        self.assertEqual(stamp,41.7)
        self.node.cmd_pub.publish.assert_not_called()

    def test_motion_compensation_waits_once_for_last_point_tf_and_uses_header_pose(self):
        self.node.step = 9
        self.node.odom_callback(self.odometry())
        self.config['sensors']['livox_motion_compensation'] = True
        message = self.timed_cloud([[.5, 0., 0.], [.5, 0., 0.]], [.1, 0.])
        def lookup(target, source, stamp, timeout):
            if source == 'livox_frame':
                return transform(-.033, 0., z=.19)
            # Translate in world x while heading world +y; a stationary
            # header frame sees later returns shifted to its right.
            return transform(2. + stamp.to_sec() - 41.7, 2., math.pi / 2)
        self.tf_buffer.lookup_transform.side_effect = lookup
        with patch.object(self.target.tf2_sensor_msgs.tf2_sensor_msgs, 'do_transform_cloud') as fallback:
            self.node.scan_callback(message)
        fallback.assert_not_called()
        calls = self.tf_buffer.lookup_transform.call_args_list
        self.assertAlmostEqual(calls[1].args[2].to_sec(), 41.8)
        self.assertEqual(calls[1].args[3].to_sec(), .1)
        for call in calls[2:-1]:
            self.assertEqual(call.args[3].to_sec(), 0.)
            self.assertGreaterEqual(call.args[2].to_sec(), 41.7)
        self.assertEqual(calls[-1].args[2], message.header.stamp)
        self.assertEqual(calls[-1].args[3].to_sec(), 0.)
        self.assertLessEqual(len(calls), 15)
        points, sensor, pose, stamp = self.node.tunnel.update_cloud.call_args.args
        np.testing.assert_allclose(points, [[.467, -.1], [.467, 0.]], atol=1e-7)
        np.testing.assert_allclose(pose, [0., 0., 0.], atol=1e-12)
        self.assertEqual(stamp, 41.7)
        self.assertEqual(sensor, (-.033, 0., 0.))
        self.node.cmd_pub.publish.assert_not_called()

    def test_motion_compensation_missing_point_tf_or_bad_time_never_refreshes_scan(self):
        self.node.step = 9
        self.node.odom_callback(self.odometry())
        self.config['sensors']['livox_motion_compensation'] = True
        message = self.timed_cloud([[.5, 0., .1]], [.1])
        for failure_index in (1, 3):
            with self.subTest(failure_index=failure_index):
                calls = []
                def lookup(*args):
                    calls.append(args)
                    if len(calls) == failure_index + 1:
                        raise RuntimeError('missing point acquisition TF')
                    return transform()
                self.tf_buffer.lookup_transform.side_effect = lookup
                with patch.object(self.target.tf2_sensor_msgs.tf2_sensor_msgs, 'do_transform_cloud') as fallback:
                    self.node.scan_callback(message)
                fallback.assert_not_called()
                self.node.tunnel.update_cloud.assert_not_called()
        self.tf_buffer.lookup_transform.side_effect = None
        with patch.object(self.target.tf2_sensor_msgs.tf2_sensor_msgs, 'do_transform_cloud') as fallback:
            self.node.scan_callback(self.timed_cloud([[.5, 0., .1]], [float('nan')]))
        fallback.assert_not_called()
        self.node.tunnel.update_cloud.assert_not_called()
        self.node.cmd_pub.publish.assert_not_called()

    def test_laserscan_preserves_raw_ranges_and_acquisition_pose(self):
        self.node.step = 9
        self.node.odom_callback(self.odometry())
        self.tf_buffer.lookup_transform.side_effect = [
            transform(-0.033, 0.02, math.pi / 2, 0.1439), transform(2.0, 3.0, math.pi / 2)]
        scan = self.laser([0.5, 0.6, 0.7, 0.8, float("inf"), float("nan"), 0.01, 4.1])
        self.node.laser_callback(scan)
        calls = self.tf_buffer.lookup_transform.call_args_list
        self.assertEqual(calls[0].args[:3], ("base_footprint", "base_scan", scan.header.stamp))
        self.assertEqual(calls[1].args[:3], ("odom", "base_footprint", scan.header.stamp))
        ranges, angle_min, increment, range_min, range_max, sensor, pose, stamp = self.node.tunnel.update_scan.call_args.args
        np.testing.assert_allclose(ranges, scan.ranges, equal_nan=True)
        self.assertEqual((angle_min, increment, range_min, range_max), (0.0, math.pi / 2, 0.05, 4.0))
        np.testing.assert_allclose(sensor, (-0.033, 0.02, math.pi / 2), atol=1e-12)
        np.testing.assert_allclose(pose, (1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual(stamp, 41.7)
        self.node.tunnel.update_cloud.assert_not_called()
        self.node.cmd_pub.publish.assert_not_called()

    def test_laserscan_invalid_returns_never_invent_observations(self):
        self.node.step = 10
        for ranges in ([], [float("nan"), -float("inf")], [0.01, 4.1]):
            self.node.laser_callback(self.laser(ranges))
        self.node.tunnel.update_scan.assert_not_called()
        self.node.tunnel.update_cloud.assert_not_called()
        self.tf_buffer.lookup_transform.assert_not_called()

    def test_laserscan_all_infinity_keeps_real_clearing_evidence(self):
        self.node.step = 10
        self.node.odom_callback(self.odometry())
        self.node.laser_callback(self.laser([float("inf"), float("inf")]))
        self.node.tunnel.update_scan.assert_called_once()
        self.assertTrue(np.all(np.isposinf(self.node.tunnel.update_scan.call_args.args[0])))
        self.node.tunnel.update_cloud.assert_not_called()

    def test_laserscan_invalid_geometry_and_missing_tf_do_not_refresh_cloud(self):
        self.node.step = 10
        self.node.laser_callback(self.laser([0.5], angle_increment=0.0))
        self.tf_buffer.lookup_transform.assert_not_called()
        self.tf_buffer.lookup_transform.side_effect = RuntimeError("missing acquisition TF")
        self.node.laser_callback(self.laser([0.5]))
        self.node.tunnel.update_cloud.assert_not_called()
        self.node.tunnel.update_scan.assert_not_called()

    def test_laserscan_uses_full_sensor_rotation_before_height_filter(self):
        sensor = transform(z=0.4)
        sensor.transform.rotation = types.SimpleNamespace(x=0.0, y=math.sin(0.1), z=0.0, w=math.cos(0.1))
        self.tf_buffer.lookup_transform.return_value = sensor
        self.node.lidar_points = Mock()
        self.node.laser_callback(self.laser([1.0]))
        points = self.node.lidar_points.call_args.args[0]
        np.testing.assert_allclose(points, [[math.cos(0.2), 0.0, 0.4-math.sin(0.2)]], atol=1e-12)

    def test_laserscan_and_cloud_share_non_tunnel_distance_calculation(self):
        self.node.step = 3
        self.tf_buffer.lookup_transform.return_value = transform(z=0.1)
        self.node.laser_callback(self.laser([0.5, 0.2, 0.5], angle_min=-0.5, angle_increment=0.5))
        distances = (self.node.left_distance, self.node.front_distance, self.node.right_distance)
        self.node.scan_callback(self.cloud([(0.5*math.cos(0.5), -0.5*math.sin(0.5), 0.1),
                                           (0.2, 0.0, 0.1),
                                           (0.5*math.cos(0.5), 0.5*math.sin(0.5), 0.1)]))
        np.testing.assert_allclose((self.node.left_distance, self.node.front_distance, self.node.right_distance), distances)
        self.node.tunnel.update_cloud.assert_not_called()

    def test_non_tunnel_distance_statistics_are_preserved(self):
        self.node.step = 3
        self.node.scan_callback(self.cloud([(0.2, 0.0, 0.1), (0.3, 0.0, 0.1),
                                           (0.4, 0.2, 0.1), (0.4, -0.3, 0.1)]))
        self.assertAlmostEqual(self.node.front_distance, 0.21)
        self.assertAlmostEqual(self.node.left_distance, math.hypot(0.4, 0.2))
        self.assertAlmostEqual(self.node.right_distance, 0.5)
        self.node.tunnel.update_cloud.assert_not_called()


if __name__ == "__main__":
    unittest.main()
