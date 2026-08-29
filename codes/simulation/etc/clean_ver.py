#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # self.joint_sub = rospy.Subscriber('/joint_states',JointState, self.joint_callback)
        self.image = rospy.Subscriber('/camera/image',Image,self.img_callback)

        self.bridge = CvBridge()
        self.ignore_sign_until = rospy.Time(0)

        
        self.lturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.lturn_template = cv2.resize(self.lturn_template,(64,64))

        self.msg = Twist()
        self.v_l = 0
        self.v_r = 0

        self.wheel_radius = 0.033
        self.wheel_separation = 0.16

        self.sign_detected = False
        self.sign_score = 0.0
        self.sign_box = None
        self.sign_count = 0

        self.prev_white_x = None
        self.prev_yellow_x = None

        self.lane_width = 285.0

        self.last_v_l = 0.0
        self.last_v_r = 0.0
        self.lost_count = 0
        self.mode = None
        
    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.crop_img = image[100:,:]

        sign_detected, sign_score = self.detect_lturn(image)
        

        if sign_detected == True:
            self.sign_count += 1

        elif sign_detected == False:
            self.sign_count = 0

        if self.sign_count >= 3:
            self.mode = 'turn_left'
            rospy.loginfo('@@@ TURN LEFT !!!! @@@')   
        

        height, width = image.shape[:2]

        roi_list = [
            (int(height * 0.72), height),
            (int(height * 0.55), int(height * 0.72)),
            (int(height * 0.38), int(height * 0.55))
        ]

        selected_white_mask = None
        selected_yellow_mask = None
        white_line = None
        yellow_line = None
        roi_start = 0

        for start_y, end_y in roi_list:
            crop_img = image[start_y:end_y, :]
            hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            white_mask = cv2.inRange(hsv_frame, np.array([0, 0, 200]), np.array([179, 50, 255]))
            yellow_mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([50, 255, 255]))

            if self.mode == 'turn_left':
                mask_width = yellow_mask.shape[1]
                yellow_mask[:, mask_width//2:] = 0

            white_mask = cv2.erode(white_mask, None, iterations=1)
            white_mask = cv2.dilate(white_mask, None, iterations=2)

            yellow_mask = cv2.erode(yellow_mask, None, iterations=1)
            yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)

        
            white_candidate = self.find_line(white_mask, self.prev_white_x)
            yellow_candidate = self.find_line(yellow_mask, self.prev_yellow_x)

            if white_candidate is not None or yellow_candidate is not None:
                selected_white_mask = white_mask
                selected_yellow_mask = yellow_mask
                white_line = white_candidate
                yellow_line = yellow_candidate
                roi_start = start_y
                break

        display_image = image.copy()
        center_x = width / 2.0
        target_x = None

        if white_line is not None:
            self.cx_w = white_line[0]
            self.cy_w = white_line[1] + roi_start
            self.prev_white_x = self.cx_w

            cv2.circle(display_image, (self.cx_w, self.cy_w), 6, (255, 0, 100), -1)

        if yellow_line is not None:
            self.cx_y = yellow_line[0]
            self.cy_y = yellow_line[1] + roi_start
            self.prev_yellow_x = self.cx_y

            cv2.circle(display_image, (self.cx_y, self.cy_y), 6, (255, 0, 0), -1)

        if white_line is not None and yellow_line is not None:
            measured_lane_width = abs(self.cx_w - self.cx_y)

            if 100 < measured_lane_width < width:
                self.lane_width = 0.9 * self.lane_width + 0.1 * measured_lane_width

            target_x = (self.cx_w + self.cx_y) / 2.0
            self.lost_count = 0

        elif white_line is not None:
            target_x = self.cx_w - self.lane_width / 2.0
            self.lost_count = 0

        elif yellow_line is not None:
            target_x = self.cx_y + self.lane_width / 2.0
            self.lost_count = 0

        else:
            self.lost_count += 1

        if target_x is not None:
            error_x = target_x - center_x

            linear = 5
            angular = -float(error_x) / 4.0
            wheel_distance = 0.2

            angular = np.clip(angular, -25.0, 25.0)

            self.v_l = linear - angular * wheel_distance * 1.6
            self.v_r = linear + angular * wheel_distance * 1.6

            self.last_v_l = self.v_l
            self.last_v_r = self.v_r

            target_y = int(height * 0.75)

            cv2.circle(display_image, (int(target_x), target_y), 8, (0, 255, 0), -1)
            cv2.line(display_image, (int(center_x), 0), (int(center_x), height), (0, 0, 255), 2)
            cv2.line(display_image, (int(center_x), target_y), (int(target_x), target_y), (0, 255, 255), 2)

        elif self.lost_count < 8:
            self.v_l = self.last_v_l * 0.6
            self.v_r = self.last_v_r * 0.6

        else:
            self.v_l = 0.0
            self.v_r = 0.0

        self.publish_velocity()

        cv2.imshow('output', display_image)

        # if selected_white_mask is not None:
        cv2.imshow('white mask', selected_white_mask)

        # if selected_yellow_mask is not None:
        cv2.imshow('yellow mask', selected_yellow_mask)

        cv2.waitKey(3)



    def find_line(self, mask, previous_x):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 80:
                continue

            moment = cv2.moments(contour)

            if moment['m00'] == 0:
                continue

            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])

            candidates.append((cx, cy, area))

        if len(candidates) == 0:
            return None

        if previous_x is None:
            return max(candidates, key=lambda value: value[2])

        return min(candidates, key=lambda value: abs(value[0] - previous_x))

    
    def detect_lturn(self,data):
        # image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)

        if self.lturn_template is None:
            print("template load fail")
            return

        template_h,template_w = self.lturn_template.shape
        image_h,image_w = gray.shape

        # if template_h > image_h or template_w > image_w:
        #     return

        res = cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0]+template_w,top_left[1]+template_h)

        rospy.loginfo("lturn score: %.3f",max_val)

        if max_val >= 0.275:
            cv2.rectangle(data,top_left,bottom_right,(0,0,255),2)
            rospy.loginfo("LEFT TURN DETECTED")
            return True, max_val

        return False, max_val
    
    def publish_velocity(self):
        # self.msg.linear.x = ( self.wheel_radius * (self.v_r + self.v_l) / 2.0 )
        # self.msg.angular.z = ( self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation )
        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.cmd_pub.publish(self.msg)

        # rospy.loginfo( 'command L: %.3f, R: %.3f ', self.v_l, self.v_r)



if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin() 