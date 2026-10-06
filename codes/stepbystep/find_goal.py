#!/usr/bin/env python
import rospy
import math
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class MazePoseViewer:
    def __init__(self):
        rospy.init_node('maze_pose_viewer')
        self.start_x=None
        self.start_y=None
        self.start_yaw=None
        rospy.Subscriber('/odom',Odometry,self.odom_callback)
        rospy.spin()

    def odom_callback(self,data):
        x=data.pose.pose.position.x
        y=data.pose.pose.position.y

        q=data.pose.pose.orientation
        _,_,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])

        if self.start_x is None:
            self.start_x=x
            self.start_y=y
            self.start_yaw=yaw
            rospy.loginfo("ORIGIN SET -> X: 0.000 Y: 0.000 YAW: 0.0")
            return

        dx=x-self.start_x
        dy=y-self.start_y

        c=math.cos(self.start_yaw)
        s=math.sin(self.start_yaw)

        relative_x=c*dx+s*dy
        relative_y=-s*dx+c*dy
        relative_yaw=math.atan2(math.sin(yaw-self.start_yaw),math.cos(yaw-self.start_yaw))

        rospy.loginfo("X: %.3f  Y: %.3f  YAW: %.1f",relative_x,relative_y,math.degrees(relative_yaw))

if __name__=='__main__':
    MazePoseViewer()