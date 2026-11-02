
import math

import numpy as np
import rospy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf_cloud
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs import point_cloud2
from tf.transformations import quaternion_matrix


def calculate_distances(points):
    left, front, right = [], [], []
    safe_width = 0.08 + 0.03
    side_limit = 0.55
    for x, y, z in points:
        if x <= 0.15 or z > 0.30:
            continue
        if abs(y) < safe_width:
            front.append(x)
        elif safe_width <= y < side_limit:
            left.append(math.hypot(x, y))
        elif -side_limit < y <= -safe_width:
            right.append(math.hypot(x, y))
    return (float(np.percentile(left, 10)) if left else float('inf'),
            min(front) if front else float('inf'),
            float(np.percentile(right, 10)) if right else float('inf'))


class DistanceMonitor:
    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        lidar_type = rospy.get_param('~lidar_type', 'pointcloud2')
        if lidar_type not in ('pointcloud2', 'laserscan'):
            raise ValueError('lidar_type must be pointcloud2 or laserscan')
        topic = rospy.get_param('~lidar_topic', '/livox/lidar' if lidar_type == 'pointcloud2'
                                else '/scan_mid360_raw')
        msg_type, callback = ((PointCloud2, self.cloud_callback) if lidar_type == 'pointcloud2'
                              else (LaserScan, self.laser_callback))
        options = {'queue_size': 1}
        if lidar_type == 'pointcloud2':
            options['buff_size'] = 4 * 1024 * 1024
        self.subscriber = rospy.Subscriber(topic, msg_type, callback, **options)
        rospy.loginfo('Distance monitor: %s (%s), frame=base_footprint, unit=m', topic, lidar_type)

    def transform(self, header):
        try:
            return self.buffer.lookup_transform('base_footprint', header.frame_id,
                                                rospy.Time(0), rospy.Duration(0.1))
        except Exception as exc:
            rospy.logwarn_throttle(1.0, 'TF ERROR: %s', exc)
            return None

    def report(self, points):
        left, front, right = calculate_distances(points)
        rospy.loginfo('LEFT: %.2f FRONT: %.2f RIGHT: %.2f', left, front, right)

    def cloud_callback(self, data):
        transform = self.transform(data.header)
        if transform is None:
            return
        try:
            cloud = tf_cloud.do_transform_cloud(data, transform)
            self.report(point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'),
                                                 skip_nans=True))
        except Exception as exc:
            rospy.logwarn_throttle(1.0, 'PointCloud transform/read failed: %s', exc)

    def laser_callback(self, data):
        if not (math.isfinite(data.angle_min) and math.isfinite(data.angle_increment)
                and data.angle_increment != 0.0 and math.isfinite(data.range_min)
                and math.isfinite(data.range_max) and 0.0 <= data.range_min < data.range_max):
            return
        ranges = np.asarray(data.ranges, dtype=float)
        valid = np.flatnonzero(np.isfinite(ranges))
        valid = valid[(ranges[valid] >= data.range_min) & (ranges[valid] <= data.range_max)]
        if not len(valid):
            rospy.logwarn_throttle(1.0, 'LaserScan: no valid finite points; no distance update')
            return
        transform = self.transform(data.header)
        if transform is None:
            return
        angles = data.angle_min + valid * data.angle_increment
        r = ranges[valid]
        points = np.column_stack((r * np.cos(angles), r * np.sin(angles), np.zeros(len(r))))
        p, q = transform.transform.translation, transform.transform.rotation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        points = points.dot(rotation.T) + np.array([p.x, p.y, p.z])
        self.report(points)


if __name__ == '__main__':
    rospy.init_node('lidar_distance_monitor', anonymous=True)
    monitor = DistanceMonitor()
    rospy.spin()
