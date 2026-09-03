#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class turtlebot():
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_callback, queue_size=1, buff_size=2**24)

        self.bridge = CvBridge()

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn1.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn1.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

        self.msg = Twist()
        self.v_l = 0.0
        self.v_r = 0.0

        self.cx_w = 0
        self.cy_w = 0
        self.cx_y = 0
        self.cy_y = 0

        self.prev_error = 0

        self.wheel_radius = 0.0475
        self.wheel_separation = 0.148

        self.turn_detect = None

    def detect_sign(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([50, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 50, 255]))
        blue_mask = cv2.inRange(hsv, np.array([90, 80, 50]), np.array([130, 255, 255]))

        mask = cv2.bitwise_or(yellow_mask, white_mask)
        mask = cv2.bitwise_or(mask, blue_mask)

        filtered = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        left_result = cv2.matchTemplate(gray, self.lturn_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, _ = cv2.minMaxLoc(left_result)

        right_result = cv2.matchTemplate(gray, self.rturn_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, _ = cv2.minMaxLoc(right_result)

        rospy.loginfo("LEFT sign: %.1f%% RIGHT sign: %.1f%%", max_val_l * 100, max_val_r * 100)

        # cv2.imshow('COLOR FILTER', filtered)

        if (max_val_l > max_val_r + 0.05 and max_val_l > 0.33) or max_val_l > 0.38:
            return 'left'

        if (max_val_r > max_val_l + 0.05 and max_val_r > 0.33) or max_val_r > 0.38:
            return 'right'

        return None

    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        if image is None:
            return

        sign = self.detect_sign(image)

        if sign == 'left' and self.turn_detect is None:
            self.turn_detect = 'left'
            rospy.loginfo("@@@@@@ LEFT SIGN DETECTED @@@@@@")

        elif sign == 'right' and self.turn_detect is None:
            self.turn_detect = 'right'
            rospy.loginfo("@@@@@@ RIGHT SIGN DETECTED @@@@@@")

        self.crop_img = image[300:, :]

        height, width = self.crop_img.shape[:2]

        hsvFrame = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 150])
        white_upper = np.array([179, 50, 255])
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([50, 255, 255])
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        if self.turn_detect == 'left':
            yellow_mask[:, 3*width//4:] = 0
            white_mask[:, :] = 0
            rospy.loginfo("@@@@@@ TURN LEFT @@@@@@@")

        elif self.turn_detect == 'right':
            white_mask[:, :width//2] = 0
            yellow_mask[:, :] = 0
            rospy.loginfo("@@@@@@ TURN RIGHT @@@@@@@")

        else:
            yellow_mask[:, :] = 0

        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)

        center_x = width / 2.0
        target_x = None

        if M_w["m00"] > 0 and M_y["m00"] > 0:
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w["m01"] / M_w["m00"])

            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y["m01"] / M_y["m00"])

            target_x = (self.cx_w + self.cx_y) / 2.0

        elif M_w["m00"] >0 and M_y["m00"]<=0:
            self.cx_w = int(M_w["m10"] / M_w["m00"])
            self.cy_w = int(M_w["m01"] / M_w["m00"])

            target_x = self.cx_w - 300

        elif M_y["m00"] > 0 and M_w["m00"]<=0 :
            self.cx_y = int(M_y["m10"] / M_y["m00"])
            self.cy_y = int(M_y["m01"] / M_y["m00"])

            target_x = self.cx_y + 300

        if target_x is not None:
            err_x = target_x - center_x
            diff_x = err_x - self.prev_error

            Kp = 0.25
            Kd = 0.1

            linear = 8.0
            angular = -(Kp * err_x + Kd * diff_x)
            angular = np.clip(angular, -18.0, 18.0)

            self.prev_error = err_x
            wheel_distance = 0.148

            self.v_l = linear - angular * wheel_distance * 1
            self.v_r = linear + angular * wheel_distance * 1

        else:
            self.v_l = 0.1
            self.v_r = 0.1

        self.publish_velocity()

        cv2.imshow('WHITE MASK', white_mask)
        cv2.imshow('YELLOW MASK', yellow_mask)
        cv2.waitKey(1)

    def publish_velocity(self):
        self.msg.linear.x = (self.wheel_radius * (self.v_r + self.v_l) / 2.0) * 0.1
        self.msg.angular.z = (self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation) * 0.2

        self.cmd_pub.publish(self.msg)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()