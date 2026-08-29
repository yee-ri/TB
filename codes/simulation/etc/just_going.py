#!/usr/bin/env python
import rospy

import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
# from cv_bridge import CvBridge

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.msg = Twist()
        # self.v_l = 1
        # self.v_r = 5

        self.wheel_radius = 0.033
        self.wheel_separation = 0.16
        self.publish_velocity()
        



    def move(self,linear,angular):
        self.msg.linear.x=linear
        self.msg.linear.y=0
        self.msg.linear.z=0
        self.msg.angular.x=0
        self.msg.angular.y=0
        self.msg.angular.z=angular
        self.publish_velocity()


    def publish_velocity(self):


        
        # self.msg.linear.x = ( self.wheel_radius * (self.v_r + self.v_l) / 2.0 )

        # self.msg.angular.z = ( self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation )

        self.cmd_pub.publish(self.msg)
        

        # rospy.loginfo_throttle( 1.0, 'command L: %.3f, R: %.3f | linear.x: %.3f, angular.z: %.3f', self.v_l, self.v_r, self.msg.linear.x, self.msg.angular.z)



if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin() 