#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class lidar_test():
    def __init__(self):
        rospy.init_node('lidar_test',anonymous=True)

        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer)

        self.robot_half_width=0.085
        self.safety_margin=0.015

        self.scan_sub=rospy.Subscriber('/livox/lidar',PointCloud2,self.scan_callback)

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

        for point in point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True):
            x,y,z=point

            if x<=0:
                continue

            if abs(y)<safe_width:
                front_ranges.append(x)

            elif safe_width<=y<0.2:
                left_ranges.append(np.sqrt(x*x+y*y))

            elif -0.2<y<=-safe_width:
                right_ranges.append(np.sqrt(x*x+y*y))

        left=min(left_ranges) if left_ranges else float('inf')
        front=min(front_ranges) if front_ranges else float('inf')
        right=min(right_ranges) if right_ranges else float('inf')

        rospy.loginfo("LEFT: %.2f | FRONT: %.2f | RIGHT: %.2f",left,front,right)

if __name__=='__main__':
    lidar=lidar_test()
    rospy.spin()