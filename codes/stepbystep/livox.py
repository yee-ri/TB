#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


class LidarDistance:
    def __init__(self):
        rospy.init_node('lidar_distance', anonymous=True)

        self.robot_half_width = 0.08
        self.safety_margin = 0.03
        self.side_limit = 0.55

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber('/livox/lidar', PointCloud2, self.lidar_callback, queue_size=1, buff_size=4*1024*1024)

    def lidar_callback(self, data):
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', data.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
            cloud = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(data, transform)
        except Exception as e:
            rospy.logwarn("TF ERROR: %s", e)
            return

        left_ranges = []
        front_ranges = []
        right_ranges = []
        safe_width = self.robot_half_width + self.safety_margin

        for x, y, z in point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True):
            if x <= 0.15 or z > 0.30:
                continue

            if abs(y) < safe_width:
                front_ranges.append(x)
            elif safe_width <= y < self.side_limit:
                left_ranges.append(math.hypot(x, y))
            elif -self.side_limit < y <= -safe_width:
                right_ranges.append(math.hypot(x, y))

        left = np.percentile(left_ranges, 10) if left_ranges else float('inf')
        front = min(front_ranges) if front_ranges else float('inf')
        right = np.percentile(right_ranges, 10) if right_ranges else float('inf')

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f", left, front, right)


if __name__ == '__main__':
    controller = LidarDistance()
    rospy.spin()