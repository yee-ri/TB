#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        
        # self.joint_sub = rospy.Subscriber('/joint_states',JointState, self.joint_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/image',Image, self.img_callback)

        self.bridge = CvBridge()

        self.msg = Twist()
        self.v_l = 0.0
        self.v_r = 0.0

        self.wheel_radius = 0.033
        self.wheel_separation = 0.16
        

        
     
    def img_callback(self,data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.crop_img = image[150:,:]
        # self.crop_img = image[100:,:]
        display_image = self.crop_img.copy()
        

        hsvFrame = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2HSV) 

        white_lower = np.array([0, 0, 200])  
        white_upper = np.array([179, 50, 255])
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([50, 255, 255]) 
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)
        h, w, _= self.crop_img.shape
        center_x = w / 2
 
        if (M_w["m00"] > 0 and M_y["m00"] > 0) :
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w['m01'] / M_w['m00'])
            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y['m01'] / M_y['m00'])

            self.cen = (self.cx_w + self.cx_y) / 2
            self.err_x = self.cen - center_x

            linear = 5
            angular = -float(self.err_x) / 2
            wheel_distance = 0.1
            self.v_l = linear - angular * wheel_distance *0.55
            self.v_r = linear + angular * wheel_distance *0.55
            
            self.publish_velocity()

        elif (M_w["m00"] > 0 and M_y["m00"] <= 0) : 
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w['m01'] / M_w['m00'])
            self.cen = self.cx_w 
            self.err_x = self.cen - center_x - 150

            linear = 5.2
            angular =-float(self.err_x) / 5
            wheel_distance = 0.2

            self.v_l = linear - angular * wheel_distance * 0.9
            self.v_r = linear + angular * wheel_distance * 0.9

            self.publish_velocity()

        elif (M_w["m00"] <= 0 and M_y["m00"] > 0) : 
            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y['m01'] / M_y['m00'])
            self.cen = self.cx_y
            self.err_x = self.cen - center_x + 135

            linear = 4.5
            angular =-float(self.err_x) / 5
            wheel_distance = 0.2
   
            self.v_l = linear - angular * wheel_distance * 0.9
            self.v_r = linear + angular * wheel_distance * 0.9

            self.publish_velocity()
        height, width = display_image.shape[:2]
        # display_image[:, width//2:] = 0
        display_image[:3*height//5, :] = 0 # 위의 3/4 안뵝ㅁ
        # cv2.circle(display_image, (self.cx_w, self.cy_w), 8, (255, 255, 255), -1)
        # cv2.circle(display_image, (self.cx_y, self.cy_y), 8, (0, 255, 255), -1)
        cv2.imshow('output', display_image)
        cv2.imshow('white mask', white_mask)
        cv2.imshow('yellow mask', yellow_mask)
        cv2.waitKey(3)

    # def joint_callback(self, data):
    #     try:
    #         r_index = data.name.index('wheel_right_joint')
    #         l_index = data.name.index('wheel_left_joint')
    #     except ValueError:
    #         return

    #     if len(data.velocity) > max(r_index, l_index):
    #         self.actual_v_r = data.velocity[r_index]
    #         self.actual_v_l = data.velocity[l_index]



    #     rospy.loginfo_throttle( 1.0, 'joint_states L: %.3f rad/s, R: %.3f rad/s', self.actual_v_l, self.actual_v_r )
        
    def publish_velocity(self):
        # self.msg.linear.x = ( self.wheel_radius * (self.v_r + self.v_l) / 2.0 )

        # self.msg.angular.z = ( self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation )

        # self.cmd_pub.publish(self.msg)

        # rospy.loginfo_throttle( 1.0, 'command L: %.3f, R: %.3f | linear.x: %.3f, angular.z: %.3f', self.v_l, self.v_r, self.msg.linear.x, self.msg.angular.z)

        rospy.loginfo("[ w CX] = %3f,[w CY] = %3f",self.cx_w,self.cy_w )

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin() 