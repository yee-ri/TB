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
        self.turn_direction = None
        self.mode = 'STOP'
        self.light = 'RED'
        self.prev_error = 0.0
        self.prev_target_x = None
        self.prev_v_l = 0.0
        self.prev_v_r = 0.0
        self.wheel_radius = 0.0475
        self.wheel_separation = 0.148

    def traffic_light(self,data):
        hsv_frame = cv2.cvtColor(data,cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv_frame, np.array([45, 100, 80]), np.array([85, 255, 255]))
        green_mask = cv2.erode(green_mask,None,iterations=1)
        green_mask = cv2.dilate(green_mask,None,iterations=2)

        if cv2.countNonZero(green_mask) >= 200:
            rospy.loginfo("DETECTED GREEN LIGHT")
            rospy.sleep(1.5)
            self.mode = 'LANE'       
            return  'GREEN'
        # cv2.imshow('output', green_mask)

        return 'RED'       

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

        if (max_val_l > max_val_r + 0.05 and max_val_l > 0.3) or max_val_l > 0.4:
            return 'left'

        if (max_val_r > max_val_l + 0.05 and max_val_r > 0.3) or max_val_r > 0.4:
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
                  
        # rospy.loginfo(self.mode)
        
        if self.mode == 'STOP' :
            if self.traffic_light(image) != 'GREEN':
                return

        sign = self.detect_sign(image)

        if not self.turn_done:
            if sign == 'left' and self.turn_detect is None and self.front_distance <= 0.55:
                self.turn_detect = 'left'
                rospy.loginfo('@@@@@@ LEFT SIGN DETECTED @@@@@@')

            elif sign == 'right' and self.turn_detect is None and self.front_distance <= 0.55:
                self.turn_detect = 'right'
                rospy.loginfo('@@@@@@ RIGHT SIGN DETECTED @@@@@@')

        if self.turn_detect is not None and not self.turn_done and self.front_distance < 0.36:
            direction = self.turn_detect

            self.turn_direction = direction
            self.turn_detect = None
            self.turn_done = True
            self.prev_error = 0.0

            if direction == 'left':
                rospy.loginfo('@@@@@@ TURN LEFT @@@@@@')
                self.turn_time('left', 1, 0.03, 0.4)

            elif direction == 'right':
                rospy.loginfo('@@@@@@ TURN RIGHT @@@@@@')
                self.turn_time('right', 1, 0.03, 0.4)

            return


        crop_img = image[300:, :]
        height, width = crop_img.shape[:2]

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 50, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([50, 255, 255]))

        if not self.turn_done:
            # yellow_mask[:, :] = 0
            pass

        elif self.turn_direction == 'left':
            yellow_mask[:, width//2:] = 0
            # white_mask[:, :] = 0

        elif self.turn_direction == 'right':
            white_mask[:, :width//2] = 0
            # yellow_mask[:, :] = 0

        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)

        center_x = width / 2.0
        target_x = None

        if M_w['m00'] > 0 and M_y['m00'] > 0:
            cx_w = int(M_w['m10'] / M_w['m00'])
            cx_y = int(M_y['m10'] / M_y['m00'])

            h = white_mask.shape[0]

            white_upper = white_mask[:h//2, :]
            white_lower = white_mask[h//2:, :]
            yellow_upper = yellow_mask[:h//2, :]
            yellow_lower = yellow_mask[h//2:, :]    

            Mw_u = cv2.moments(white_upper)
            Mw_l = cv2.moments(white_lower)
            My_u = cv2.moments(yellow_upper)
            My_l = cv2.moments(yellow_lower)

            target_x = (cx_w + cx_y) / 2.0

            if Mw_u['m00'] > 0 and Mw_l['m00'] > 0 and My_u['m00'] > 0 and My_l['m00'] > 0:
                cx_w_u = int(Mw_u['m10'] / Mw_u['m00'])
                cx_w_l = int(Mw_l['m10'] / Mw_l['m00'])
                cx_y_u = int(My_u['m10'] / My_u['m00'])
                cx_y_l = int(My_l['m10'] / My_l['m00'])

                white_curve = abs(cx_w_u - cx_w_l)
                yellow_curve = abs(cx_y_u - cx_y_l)

                if white_curve > yellow_curve + 40:
                    target_x +=90

                elif yellow_curve > white_curve + 40:
                    target_x -= 90

        elif M_w['m00'] > 0:
            cx_w = int(M_w['m10'] / M_w['m00'])

            h = white_mask.shape[0]
            upper_mask = white_mask[:h//2, :]
            lower_mask = white_mask[h//2:, :]

            M_upper = cv2.moments(upper_mask)
            M_lower = cv2.moments(lower_mask)

            offset = 300

            if M_upper['m00'] > 0 and M_lower['m00'] > 0:
                cx_upper = int(M_upper['m10'] / M_upper['m00'])
                cx_lower = int(M_lower['m10'] / M_lower['m00'])

                curve = abs(cx_upper - cx_lower)

                if curve >190:  # BIG 110-145
                    offset += min((curve - 190) * 0.82, 100)

            target_x = cx_w - offset

        elif M_y['m00'] > 0:
            cx_y = int(M_y['m10'] / M_y['m00'])

            h = yellow_mask.shape[0]
            upper_mask = yellow_mask[:h//2, :]
            lower_mask = yellow_mask[h//2:, :]

            M_upper = cv2.moments(upper_mask)
            M_lower = cv2.moments(lower_mask)

            offset = 300

            if M_upper['m00'] > 0 and M_lower['m00'] > 0:
                cx_upper = int(M_upper['m10'] / M_upper['m00'])
                cx_lower = int(M_lower['m10'] / M_lower['m00'])

                curve = abs(cx_upper - cx_lower)

                if curve > 180:

                    offset += min((curve - 180) * 0.8, 100)

            target_x = cx_y + offset

        if target_x is not None:
            if self.prev_target_x is None:
                self.prev_target_x = target_x
            else:
                target_x = 0.25 * target_x + 0.75 * self.prev_target_x
                self.prev_target_x = target_x

            err_x = target_x - center_x
            diff_x = err_x - self.prev_error

            Kp = 0.15
            Kd = 0.008
            linear = 14.0 #10
            max_angular = 38.0 #25

            angular = -(Kp * err_x + Kd * diff_x)
            angular = np.clip(angular, -max_angular, max_angular)

            self.prev_error = err_x

            wheel_distance = 0.148

            v_l = linear - angular * wheel_distance * 0.5
            v_r = linear + angular * wheel_distance * 0.5

            self.prev_v_l = v_l
            self.prev_v_r = v_r

            self.publish_velocity(v_l, v_r)

        else:
            self.publish_velocity(self.prev_v_l, self.prev_v_r)

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