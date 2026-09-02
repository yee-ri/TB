#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class turtlebot():
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.img_callback, queue_size=1)

        self.bridge = CvBridge()

        self.msg = Twist()
        self.v_l = 0.0
        self.v_r = 0.0

        self.cx_w = 0
        self.cy_w = 0
        self.cx_y = 0
        self.cy_y = 0

        self.wheel_radius = 0.0475
        self.wheel_separation = 0.148

         
        self.last_target_x = None

    def img_callback(self, data):
        image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return

        self.crop_img = image[300:, :]
        display_image = self.crop_img.copy()

        hsvFrame = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 150])
        white_upper = np.array([179, 50, 255])
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

        # yellow_lower = np.array([20, 100, 100])
        # yellow_upper = np.array([50, 255, 255])
        # yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        M_w = cv2.moments(white_mask)
        # M_y = cv2.moments(yellow_mask)

        height, width = self.crop_img.shape[:2]
        center_x = width / 2.0

        target_x = None

        rospy.loginfo(M_w["m00"])

        if M_w["m00"] > 0:
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w["m01"] / M_w["m00"])

            # self.cx_y = int(M_y["m10"] / M_y["m00"])
            # self.cy_y = int(M_y["m01"] / M_y["m00"])

            # measured_lane_width = abs(self.cx_w - self.cx_y)

            # if 150 < measured_lane_width < 400:
            #     self.lane_width = self.lane_width * 0.9 + measured_lane_width * 0.1

            target_x = self.cx_w - 300

            rospy.loginfo("CENTER: %f",target_x)
            rospy.loginfo("White: %f",self.cx_w)

        # elif M_w["m00"] > 0:
        #     self.cx_w = int(M_w["m10"] / M_w["m00"])
        #     self.cy_w = int(M_w["m01"] / M_w["m00"])

        #     target_x = self.cx_w - 300 #self.lane_width / 2.0

        # elif M_y["m00"] > 0:
        #     self.cx_y = int(M_y["m10"] / M_y["m00"])
        #     self.cy_y = int(M_y["m01"] / M_y["m00"])

        #     target_x = self.cx_y + 300 #self.lane_width / 2.0

        if target_x is not None:
            # if self.last_target_x is None:
            #     self.last_target_x = target_x

            # target_x = self.last_target_x * 0.7 + target_x * 0.3
            # self.last_target_x = target_x

            err_x = target_x - center_x

            linear = 8.0
            angular = -float(err_x) / 4.0
            angular = np.clip(angular, -12.0, 12.0)

            wheel_distance = 0.148

            self.v_l = linear - angular * wheel_distance * 0.8
            self.v_r = linear + angular * wheel_distance * 0.8

        else:
            self.v_l = 0.01
            self.v_r = 0.01

        self.publish_velocity()

        cv2.imshow('crop', display_image)
        cv2.waitKey(1)

    def publish_velocity(self):
        self.msg.linear.x = (self.wheel_radius * (self.v_r + self.v_l) / 2.0) * 0.1
        self.msg.angular.z = (self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation) * 0.18

        self.cmd_pub.publish(self.msg)
        rospy.loginfo("linear: %.3f angular: %.3f", self.msg.linear.x, self.msg.angular.z)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()