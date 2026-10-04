#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs

class LidarViewer:
    def __init__(self):
        rospy.init_node('lidar_viewer',anonymous=True)

        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer)

        self.robot_half_width=0.08
        self.safety_margin=0.03

        rospy.Subscriber('/livox/lidar',PointCloud2,self.scan_callback)
        rospy.spin()

    def scan_callback(self,data):
        try:
            transform=self.tf_buffer.lookup_transform('base_footprint',data.header.frame_id,rospy.Time(0),rospy.Duration(0.1))
            cloud=tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(data,transform)
        except Exception as e:
            rospy.logwarn("TF ERROR: %s",e)
            return

        left_ranges=[]
        front_ranges=[]
        right_ranges=[]

        safe_width=self.robot_half_width+self.safety_margin
        side_limit=0.55

        for point in point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True):
            x,y,z=point

            if x<=0.1:
                continue
            if z>0.30:
                continue

            if abs(y)<safe_width:
                front_ranges.append(x)
            elif safe_width<=y<side_limit:
                left_ranges.append(np.sqrt(x*x+y*y))
            elif -side_limit<y<=-safe_width:
                right_ranges.append(np.sqrt(x*x+y*y))

        left_distance=min(left_ranges) if left_ranges else float('inf')
        front_distance=min(front_ranges) if front_ranges else float('inf')
        right_distance=min(right_ranges) if right_ranges else float('inf')

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f",left_distance,front_distance,right_distance)

if __name__=='__main__':
    LidarViewer()