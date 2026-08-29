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
        self.image = rospy.Subscriber('/camera/image',Image,self.img_callback)

        self.bridge = CvBridge()
        self.msg = Twist()

        self.lturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.lturn_template = cv2.resize(self.lturn_template,(50,50))
        self.rturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/rturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.rturn_template = cv2.resize(self.rturn_template,(50,50))
        self.construct_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/repair.jpg',cv2.IMREAD_GRAYSCALE)
        self.construct_template = cv2.resize(self.construct_template,(60,60)) # 대충 0.4는 넘음

        self.v_l = 0
        self.v_r = 0

        self.wheel_radius = 0.033
        self.wheel_separation = 0.16
        self.lane_width = 285.0

        self.sign_detected = False
        self.sign = None
        self.sign_score = 0.0
        self.sign_box = None
        self.sign_count = 0

        self.prev_white_x = None
        self.prev_yellow_x = None
        self.light = None
        self.ignore_sign  = 0
        self.hide_yellow  = 0
        self.hide_white = 0
        self.last_v_l = 0.0
        self.last_v_r = 0.0
        self.lost_count = 0
        self.sign_c = False
        self.mode = 'stop' # stop,lane tracking, turn left, right, construc t
        self.c_mode = 0

    def detect_light(self,data):
        hsv_frame = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv_frame, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_mask = cv2.erode(green_mask, None, iterations=1)
        green_mask = cv2.dilate(green_mask, None, iterations=2)
        green_pixel = cv2.countNonZero(green_mask)
        # rospy.loginfo ("%.3f",green_pixel)

        if green_pixel >= 200:
            rospy.loginfo("DETECTED GREEN LIGHT")
            self.light = 'green'
            self.mode = 'lane'
            return 'green'

    def detect_sign(self, data):
        self.detect_lturn(data)
        self.detect_rturn(data)

        # rospy.loginfo("Lturn score: %.3f    Rturn score: %.3f", self.max_val_l, self.max_val_r)
        if self.max_val_l > 0.4: return 'left'
        elif self.max_val_r > 0.4: return 'right'
        
        return None
    

               
    def detect_lturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        min_val,self.max_val_l,min_loc,max_loc = cv2.minMaxLoc(res)

        return False
    
    def detect_rturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)

        if self.rturn_template is None:
            print("template load fail")
            return False, None
        res = cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        min_val,self.max_val_r,min_loc,max_loc = cv2.minMaxLoc(res)

        return False


    def scan_callback(self, data):
        front_ranges = []

        for i in range(len(data.ranges)):
            distance = data.ranges[i]

            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = data.angle_min + i * data.angle_increment
            angle_deg = np.degrees(angle)

            if angle_deg > 180:
                angle_deg -= 360

            if -15 <= angle_deg <= 15:
                front_ranges.append(distance)

        if len(front_ranges) == 0:
            return

        front_distance = min(front_ranges)

        rospy.loginfo("FRONT DISTANCE: %.2f m", front_distance)

    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.crop_img = image[50:,:]

        # rospy.loginfo("MODE IS %s", self.mode)

        if self.mode == 'stop':
            self.light = self.detect_light(image)

            if self.light == 'green':
                self.mode = 'lane'
            else:
                return

        if self.mode == 'lane':
            self.sign = self.detect_sign(image)
            self.detect_construct(image)

            # rospy.loginfo("COUNT: %d  SIGN: %s", self.sign_count, self.sign)

            if self.sign == 'left':
                self.mode = 'turn_left'
                self.c_mode = 1
                self.hide_yellow = 1
                self.sign_count = 0
                # rospy.loginfo('@@@ TURN LEFT !!!! @@@')

            elif self.sign == 'right':
                self.mode = 'turn_right'
                self.c_mode = 1
                self.hide_white = 1
                self.sign_count = 0
                # rospy.loginfo('@@@ TURN RIGHT !!!! @@@')    

            self.prev_white_x = None
            self.prev_yellow_x = None
        # rospy.loginfo("C MODE IS %s    ",self.c_mode)
        if self.c_mode == 1:
             self.sign_c = self.detect_construct(image)
             if self.sign_c  :
                self.mode = 'construct'
                self.hide_white = 2
                rospy.loginfo('@@@ CONSTRUCTION SIGN!!!! @@@')  

        height, width = image.shape[:2]

        roi_list = [
            (int(height * 0.55), height),
            (int(height * 0.45), int(height * 0.72)),
            (int(height * 0.35), int(height * 0.55))
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

            if  self.hide_yellow == 1:
                yellow_mask[:, width//2:] = 0
                # rospy.loginfo("Hide YELLOW ")
                
            elif  self.hide_white == 1:
                white_mask[:, :width//2] = 0
                # rospy.loginfo("Hide WHITE ")

            elif self.hide_white == 2:
                white_mask[:, : ] = 0

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
            angular = -float(error_x) / 3.0
            wheel_distance = 0.2

            angular = np.clip(angular, -25.0, 25.0)

            self.v_l = linear - angular * wheel_distance * 2
            self.v_r = linear + angular * wheel_distance * 2

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
        cv2.imshow('white mask', selected_white_mask)
        cv2.imshow('yellow mask', selected_yellow_mask)
        cv2.imshow('crop image',self.crop_img)
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

    def publish_velocity(self):
        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.cmd_pub.publish(self.msg)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()
