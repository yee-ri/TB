#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class turtlebot():
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_callback, queue_size=1, buff_size=2**24)
        self.scan_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.scan_callback)

        self.bridge = CvBridge()

        self.lturn_template = cv2.imread('/home/sj/Desktop/TB/images/lturn1.png', cv2.IMREAD_GRAYSCALE)
        self.lturn_template = cv2.resize(self.lturn_template, (100, 100))

        self.rturn_template = cv2.imread('/home/sj/Desktop/TB/images/rturn1.png', cv2.IMREAD_GRAYSCALE)
        self.rturn_template = cv2.resize(self.rturn_template, (100, 100))

        self.front_distance = float('inf')

        self.turn_detect = None
        self.turn_done = False

        self.prev_error = 0.0

        self.wheel_radius = 0.0475
        self.wheel_separation = 0.148

    def scan_callback(self, data):
        front_ranges = []

        for point in point_cloud2.read_points(data, field_names=('x', 'y', 'z'), skip_nans=True):
            x, y, z = point

            distance = np.sqrt(x * x + y * y)

            if distance <= 0:
                continue

            angle_deg = np.degrees(np.arctan2(y, x))

            if -30 <= angle_deg <= 30:
                front_ranges.append(distance)

        self.front_distance = min(front_ranges) if front_ranges else float('inf')

        rospy.loginfo('FRONT: %.2f', self.front_distance)

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

        rospy.loginfo('LEFT: %.1f%% RIGHT: %.1f%%', max_val_l * 100, max_val_r * 100)

        if (max_val_l > max_val_r + 0.03 and max_val_l > 0.27) or max_val_l > 0.32:
            return 'left'

        if (max_val_r > max_val_l + 0.03 and max_val_r > 0.27) or max_val_r > 0.32:
            return 'right'

        return None

    def turn_time(self, direction, duration, linear_speed, angular_speed):
        msg = Twist()
        msg.linear.x = linear_speed

        if direction == 'left':
            msg.angular.z = angular_speed

        elif direction == 'right':
            msg.angular.z = -angular_speed

        start_time = rospy.Time.now()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed >= duration:
                break

            self.cmd_pub.publish(msg)
            rate.sleep()

        self.cmd_pub.publish(Twist())

    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        if image is None:
            return

        sign = self.detect_sign(image)

        if sign == 'left' and self.turn_detect is None and self.front_distance<=0.45 and not self.turn_done:
            self.turn_detect = 'left'
            rospy.loginfo('@@@@@@ LEFT SIGN DETECTED @@@@@@')


        elif sign == 'right' and self.turn_detect is None and self.front_distance<=0.45  and not self.turn_done:
            
            self.turn_detect = 'right'
            rospy.loginfo('@@@@@@ RIGHT SIGN DETECTED @@@@@@')


        if self.turn_detect is not None and self.front_distance <= 0.3:
            if self.turn_detect == 'left':
                rospy.loginfo('@@@@@@ TURN LEFT @@@@@@')
                self.turn_time('left', 0.8, 0.03, 0.4)

            elif self.turn_detect == 'right':
                rospy.loginfo('@@@@@@ TURN RIGHT @@@@@@')
                self.turn_time('right', 0.8, 0.03, 0.4)

            self.turn_detect = None
            self.turn_done = True
            self.prev_error = 0.0
            return

        if self.turn_done and sign is None and self.front_distance > 1.0:
            self.turn_done = False

        crop_img = image[300:, :]
        height, width = crop_img.shape[:2]

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 50, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([50, 255, 255]))

        yellow_mask[:, :] = 0

        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)

        center_x = width / 2.0
        target_x = None

        if M_w['m00'] > 0 and M_y['m00'] > 0:
            cx_w = int(M_w['m10'] / M_w['m00'])
            cx_y = int(M_y['m10'] / M_y['m00'])

            target_x = (cx_w + cx_y) / 2.0

        elif M_w['m00'] > 0:
            cx_w = int(M_w['m10'] / M_w['m00'])
            target_x = cx_w - 300

        elif M_y['m00'] > 0:
            cx_y = int(M_y['m10'] / M_y['m00'])
            target_x = cx_y + 300

        if target_x is not None:
            err_x = target_x - center_x
            diff_x = err_x - self.prev_error

            Kp = 0.25
            Kd = 0.01
            linear = 10.0
            max_angular = 20.0

            angular = -(Kp * err_x + Kd * diff_x)
            angular = np.clip(angular, -max_angular, max_angular)

            self.prev_error = err_x

            wheel_distance = 0.148

            v_l = linear - angular * wheel_distance * 0.6
            v_r = linear + angular * wheel_distance * 0.6

            self.publish_velocity(v_l, v_r)

        else:
            self.publish_velocity(0.1, 0.1)

        mask_view = cv2.vconcat([white_mask, yellow_mask])
        cv2.imshow('MASK VIEW', mask_view)
        cv2.waitKey(1)

    def publish_velocity(self, v_l, v_r):
        msg = Twist()

        msg.linear.x = (self.wheel_radius * (v_r + v_l) / 2.0) * 0.1
        msg.angular.z = (self.wheel_radius * (v_r - v_l) / self.wheel_separation) * 0.2

        self.cmd_pub.publish(msg)


if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()