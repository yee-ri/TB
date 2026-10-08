"""TB adapter tests with mocked ROS transport; not a robot-driving test."""
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "codes/stepbystep/try_maze.py"


class Stamp:
    def __init__(self, seconds=0.0):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds

    @classmethod
    def now(cls):
        return cls(42.0)


def vector(x=0.0, y=0.0, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z)


def quaternion(yaw=0.0):
    return types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def transform(x=0.0, y=0.0, yaw=0.0):
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=vector(x, y), rotation=quaternion(yaw)))


class Twist:
    def __init__(self):
        self.linear, self.angular = vector(), vector()


class Mission:
    def __init__(self, config, clock=None):
        self.config = config
        self.clock = clock
        self.state, self.reason = "ENTRY", "test"
        self.reset = Mock()
        self.close = Mock()
        self.step = Mock(return_value=(0.06, 0.02, False))
        self.update_odometry = Mock()
        self.update_cloud = Mock()
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
            self.subscribers.append((topic, callback, kwargs))
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
                         (0.0, 0.0, 2.0 * math.atan2(q[2], q[3])))
        cloud_reader = module("sensor_msgs.point_cloud2", read_points=lambda cloud, **kwargs: cloud.points)
        cloud_tf = module("tf2_sensor_msgs.tf2_sensor_msgs", do_transform_cloud=lambda cloud, tf: cloud)
        mission = module("tunnel.mission", TunnelMission=Mission, load_config=lambda path: self.config)
        modules = {
            "rospy": rospy,
            "tf": module("tf", transformations=tf_math), "tf.transformations": tf_math,
            "sensor_msgs": module("sensor_msgs", point_cloud2=cloud_reader),
            "sensor_msgs.msg": module("sensor_msgs.msg", Image=object, PointCloud2=object),
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
            "tunnel": module("tunnel", mission=mission), "tunnel.mission": mission,
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

    def test_startup_initializes_before_callbacks_and_uses_one_publisher(self):
        self.assertEqual(self.node.step, 0)
        self.assertEqual([p.topic for p in self.publishers], ["/cmd_vel"])
        self.assertEqual(len(self.subscribers), 5)
        self.assertEqual(self.timers[0][0], 0.05)
        self.assertEqual(self.node.tunnel.clock(), 42.0)
        self.assertEqual(self.node.lturn_template.shape, (100, 100))
        self.assertEqual(self.node.rturn_template.shape, (100, 100))

    def test_explicit_start_step_is_configurable(self):
        self.params["~start_step"] = 9
        self.assertEqual(self.target.turtlebot().step, 9)

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
        self.assertEqual(self.node.step, 9)
        self.assertEqual(self.node.tunnel_generation, 1)
        self.node.cmd_pub.publish.assert_not_called()

    def test_completion_releases_to_lane_on_next_image(self):
        self.node.step = 10
        self.node.tunnel.step.return_value = (0.0, 0.0, True)
        self.node.tunnel.state = "COMPLETE"
        self.node.tunnel_tick(None)
        self.assertEqual(self.node.step, 11)
        self.assertEqual(self.node.prev_v_l, 0.0)
        self.node.lane_tracking = Mock()
        self.node.img_callback(self.image(True))
        self.node.lane_tracking.assert_called_once()

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
