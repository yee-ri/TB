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

    def img_callback(self, data):
        image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return

        height, width = image.shape[:2]

        roi_start = int(height * 0.7)

        self.crop_img = image[roi_start:, :]
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

        crop_height, crop_width = self.crop_img.shape[:2]
        center_x = crop_width / 2.0

        if M_w["m00"] > 0 and M_y["m00"] > 0:
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w["m01"] / M_w["m00"])

            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y["m01"] / M_y["m00"])

            cen = (self.cx_w + self.cx_y) / 2.0
            err_x = cen - center_x

            linear = 10
            # angular = -float(err_x) / 2.0 * 0.4
            angular = -float(err_x) / 3
            wheel_distance = 0.148

            self.v_l = linear - angular * wheel_distance * 1.5
            self.v_r = linear + angular * wheel_distance * 1.5

        elif M_w["m00"] > 0:
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w["m01"] / M_w["m00"])

            cen = self.cx_w
            err_x = cen - center_x - 125

            linear = 7.0
            angular = -float(err_x) / 3
            wheel_distance = 0.148

            self.v_l = linear - angular * wheel_distance * 1.2
            self.v_r = linear + angular * wheel_distance * 1.2

        elif M_y["m00"] > 0:
            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y["m01"] / M_y["m00"])

            cen = self.cx_y
            err_x = cen - center_x + 125

            linear = 7.0
            angular = -float(err_x) / 3
            wheel_distance = 0.148

            self.v_l = linear - angular * wheel_distance * 1.2
            self.v_r = linear + angular * wheel_distance * 1.2

        else:
            self.v_l = 0.01
            self.v_r = 0.01


        # cv2.imshow('crop',self.crop_img)
        # cv2.waitKey(1)

        self.publish_velocity()

        # if M_w["m00"] > 0:
        #     cv2.circle(display_image, (self.cx_w, self.cy_w), 6, (255, 0, 0), -1)

        # if M_y["m00"] > 0:
        #     cv2.circle(display_image, (self.cx_y, self.cy_y), 6, (0, 255, 255), -1)


    def publish_velocity(self):
        self.msg.linear.x = (self.wheel_radius * (self.v_r + self.v_l) / 2.0) * 0.1
        self.msg.angular.z = (self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation) * 0.12

        self.cmd_pub.publish(self.msg)
        rospy.loginfo("linear: %.3f angular: %.3f", self.msg.linear.x, self.msg.angular.z)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()