import rospy
import cv2
import numpy as np
import math

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry


class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.bridge = CvBridge()
        self.msg = Twist()

        self.lturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.lturn_template = cv2.resize(self.lturn_template,(10,10))

        self.rturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/rturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.rturn_template = cv2.resize(self.rturn_template,(10,10))

        self.construct_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/construct.png',cv2.IMREAD_GRAYSCALE)
        self.construct_template = cv2.resize(self.construct_template,(60,60))

        self.parking_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/parking.png',cv2.IMREAD_GRAYSCALE)
        self.parking_template = cv2.resize(self.parking_template,(20,20))

        self.hide = 0
        self.hide_yellow = 0
        self.hide_white = 0

        self.park_num = 0
        self.after_parking = 0
        self.parking_signal = 0

        self.v_l = 0.0
        self.v_r = 0.0
        self.last_v_l = 0.0
        self.last_v_r = 0.0

        self.wheel_radius = 0.033
        self.wheel_separation = 0.16
        self.lane_width = 285.0

        self.prev_white_x = None
        self.prev_yellow_x = None
        self.cx_w = 0
        self.cx_y = 0

        self.lost_count = 0

        self.mode = 'stop'
        self.c_mode = 0
        self.gain = 1
        self.step = 0
        self.alpha = 0

        self.sign = None
        self.sign_c = False
        self.light = None

        self.yaw = None
        self.yaw_hold = 0

        self.ready_bar = None

        self.left_distance = float('inf')
        self.front_distance = float('inf')
        self.right_distance = float('inf')

        self.cmd_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/image',Image,self.img_callback)
        self.scan_sub = rospy.Subscriber('/scan',LaserScan,self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom',Odometry,self.odom_callback)

    def detect_light(self,data):
        hsv_frame = cv2.cvtColor(data,cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv_frame,np.array([35,50,50]),np.array([85,255,255]))
        green_mask = cv2.erode(green_mask,None,iterations=1)
        green_mask = cv2.dilate(green_mask,None,iterations=2)

        if cv2.countNonZero(green_mask) >= 200:
            rospy.loginfo("DETECTED GREEN LIGHT")
            self.light = 'green'
            self.mode = 'lane'
            return 'green'

        return None

    def scan_callback(self,data):
        left_ranges = []
        front_ranges = []
        right_ranges = []

        for i in range(len(data.ranges)):
            distance = data.ranges[i]

            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = data.angle_min+i*data.angle_increment
            angle_deg = np.degrees(angle)

            if angle_deg > 180:
                angle_deg -= 360

            if 30 <= angle_deg <= 80:
                right_ranges.append(distance)

            elif -20 <= angle_deg <= 20:
                front_ranges.append(distance)

            elif -80 <= angle_deg <= -30:
                left_ranges.append(distance)

        self.left_distance = min(left_ranges) if left_ranges else float('inf')
        self.front_distance = min(front_ranges) if front_ranges else float('inf')
        self.right_distance = min(right_ranges) if right_ranges else float('inf')

    def odom_callback(self,data):
        q = data.pose.pose.orientation
        _,_,self.yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])

    def set_yaw(self,num):
        if self.yaw is None:
            return

        if num == 1:
            target_yaw = 1.545
        elif num == 2:
            target_yaw = 3.135
        elif num == 3:
            target_yaw = -0.02
        else:
            return

        while not rospy.is_shutdown():
            err = target_yaw-self.yaw
            err = math.atan2(math.sin(err),math.cos(err))

            rospy.loginfo("NUM: %d YAW: %.3f TARGET: %.3f ERR: %.3f",num,self.yaw,target_yaw,err)

            if abs(err) < 0.015:
                self.move(0,0)
                rospy.sleep(0.03)
                return 1

            v_yaw = np.clip(err*8,-1.5,1.5)
            self.move(0,v_yaw)

    def dis_y(self,image):
        crop = image[150:,:]
        hsv_frame = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))
        M_y = cv2.moments(yellow_mask)

        if M_y["m00"] > 0:
            cy_y = int(M_y["m01"]/M_y["m00"])
            rospy.loginfo("CY_Y is %d",cy_y)

            if cy_y < 43:
                self.move(0.2,0)
                return False

            self.move(0,0)
            return True

        return False

    def dis_w(self,image):
        crop = image[150:,30:]
        hsv_frame = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
        M_w = cv2.moments(white_mask)

        if M_w["m00"] > 0:
            cy_w = int(M_w["m01"]/M_w["m00"])
            rospy.loginfo("CY_W is %d",cy_w)

            if cy_w < 47:
                self.move(0.2,0)
                return False

            self.move(0,0)
            return True

        rospy.loginfo("WHITE NOT DETECTED")
        return False

    def move(self,linear,angular):
        self.msg.linear.x = linear
        self.msg.linear.y = 0
        self.msg.linear.z = 0
        self.msg.angular.x = 0
        self.msg.angular.y = 0
        self.msg.angular.z = angular
        self.publish_velocity()

    def avoid_obstacle(self,image):
        if self.step == 1:
            if self.dis_y(image):
                self.move(0,0)
                rospy.sleep(0.02)
                self.step = 2
            return

        if self.step == 2:
            if self.set_yaw(1):
                self.move(0,0)
                rospy.sleep(0.04)
                self.step = 3
            return

        if self.step == 3:
            if self.front_distance > 0.25:
                self.move(0.2,0)
            else:
                self.move(0,0)
                rospy.sleep(0.02)
                self.step = 4
            return

        if self.step == 4:
            if self.set_yaw(3):
                self.move(0,0)
                rospy.sleep(0.02)
                self.step = 5
            return

        if self.step == 5:
            if self.dis_w(image):
                self.move(0,0)
                rospy.sleep(0.02)
                self.step = 6
            return

        if self.step == 6:
            if self.set_yaw(1):
                self.move(0,0)
                rospy.sleep(0.02)
                self.step = 7
                self.gain = 0
                self.mode = 'lane'
                self.c_mode = 2
                self.hide_white = 2
                self.park_num = 1
            return

        crop_image = image[186:,:]
        height,width = crop_image.shape[:2]
        hsv_frame = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
        yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

        white_line = self.find_line(white_mask,self.prev_white_x)

        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)

        self.cx_w = int(M_w["m10"]/M_w["m00"]) if M_w["m00"] > 0 else 0
        self.cx_y = int(M_y["m10"]/M_y["m00"]) if M_y["m00"] > 0 else 0

        rospy.loginfo("DIST: %.2f YELLOW: %d WHITE: %d",self.front_distance,self.cx_y,self.cx_w)

        if self.front_distance <= 0.25 : #and self.right_distance < 0.3
            self.move(0.1,0)

            if self.front_distance <= 0.23:
                self.move(0,0)
                rospy.sleep(0.2)

                if self.cx_y < 20 and self.cx_w > 170 and self.step == 0:
                    self.move(0,0)
                    rospy.sleep(0.02)

                    if self.set_yaw(2):
                        self.move(0,0)
                        rospy.sleep(0.02)
                        self.step = 1

                    return

            return

        if white_line is None:
            self.move(0,0)
            return

        self.cx_w = white_line[0]
        self.prev_white_x = self.cx_w

        center_x = width/2.0
        target_x = self.cx_w-90
        error_x = target_x-center_x

        linear = 7
        angular = np.clip(-float(error_x)/2.5,-20.0,20.0)
        wheel_distance = 0.2

        self.v_l = linear-angular*wheel_distance
        self.v_r = linear+angular*wheel_distance

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()

    def detect_sign(self,data):
        self.detect_lturn(data)
        self.detect_rturn(data)

        rospy.loginfo("Lturn score: %.3f Rturn score: %.3f",self.max_val_l,self.max_val_r)

        if 0.1 < self.front_distance <= 0.34:
            if self.max_val_l > self.max_val_r+0.03 or self.max_val_l > 0.47:
                return 'left'

            if self.max_val_r > self.max_val_l+0.03 or self.max_val_r > 0.47:
                return 'right'

        return None

    def detect_construct(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.construct_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_c,_,_ = cv2.minMaxLoc(res)

        rospy.loginfo("CONSTRUCTION VAL IS %.3f",self.max_val_c)

        if self.max_val_c >= 0.55 and 0.92 < self.front_distance < 1.2:
            return True

        return False

    def detect_parking(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.parking_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_p,_,_ = cv2.minMaxLoc(res)

        rospy.loginfo("Parking VAL IS %.3f",self.max_val_p)

        if self.max_val_p >= 0.325 and 0.5 < self.front_distance < 0.6:
            rospy.loginfo("DETECT PARKING SIGN!!!!!")
            return True

        return False

    def finish_parking(self):
        self.move(0,0)
        rospy.sleep(0.04)

        self.c_mode = 0
        self.park_num = 0
        self.mode = 'lane'
        self.gain = 1

        self.parking_signal = 0
        self.yaw_hold = 0
        self.alpha = 0

        self.after_parking = 1

        self.hide_white = 0
        self.hide_yellow = 0

        self.prev_white_x = None
        self.prev_yellow_x = None

        self.ready_bar = 1

        rospy.loginfo("@@@ PARKING END -> TRACK YELLOW 2 @@@")

    def go_parking(self,data):
        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f",self.left_distance,self.front_distance,self.right_distance)

        if self.parking_signal == 0:
            if self.detect_parking(data):
                self.parking_signal = 1

        if self.yaw_hold == 1 and self.alpha == 0:
            if 0.85 < self.left_distance < 1.1 and self.right_distance < 0.25 and 1.5 < self.front_distance < 1.6:
                self.alpha = 1
                self.move(0,0)
                rospy.loginfo("@@@ LEFT PARKING SEQUENCE START @@@")
                rospy.sleep(0.04)
                return

            if 0.85 < self.right_distance < 1.1 and 1.5 < self.front_distance < 1.6 and 0.2 < self.left_distance < 0.25:
                self.alpha = 2
                self.move(0,0)
                rospy.loginfo("@@@ RIGHT PARKING SEQUENCE START @@@")
                rospy.sleep(0.04)
                return

            if self.yaw is not None:
                yaw_err = -1.56-self.yaw
                yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                angular = np.clip(yaw_err*3.0,-0.5,0.5)

                self.move(0.15,angular)
                rospy.loginfo("YAW HOLD: %.3f TARGET: -1.560 ERR: %.3f",self.yaw,yaw_err)

            return

        if self.alpha == 1:
            if self.left_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.02)

                if self.set_yaw(2):
                    self.move(0.15,0)
                    rospy.sleep(2)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 3

        if self.alpha == 2:
            if self.right_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.04)

                if self.set_yaw(3):
                    self.move(0.15,0)
                    rospy.sleep(2)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 4

        if self.alpha == 3 or self.alpha == 4:
            if self.set_yaw(1):
                self.finish_parking()
                return

        height,width = data.shape[:2]

        roi_list = [
            (int(height*0.55),height),
            (int(height*0.45),int(height*0.72)),
            (int(height*0.35),int(height*0.55))
        ]

        yellow_objects = []
        white_objects = []

        for start_y,end_y in roi_list:
            crop = data[start_y:end_y,:]
            hsv_frame = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)

            yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))
            white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))

            if self.parking_signal == 0:
                yellow_mask[:,:] = 0

            yellow_mask = cv2.erode(yellow_mask,None,iterations=1)
            yellow_mask = cv2.dilate(yellow_mask,None,iterations=2)

            white_mask = cv2.erode(white_mask,None,iterations=1)
            white_mask = cv2.dilate(white_mask,None,iterations=2)

            yellow_contours,_ = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            white_contours,_ = cv2.findContours(white_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            temp_yellow_objects = []
            temp_white_objects = []
            max_yellow_area = 0

            for contour in yellow_contours:
                area = cv2.contourArea(contour)
                max_yellow_area = max(max_yellow_area,area)

                if area < 40:
                    continue

                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                temp_yellow_objects.append((cx,cy,area))

            if self.parking_signal == 1:
                rospy.loginfo("YELLOW AREA: %.1f",max_yellow_area)

                if 0 < max_yellow_area < 1400:
                    self.yaw_hold = 1
                    rospy.loginfo("@@@ YELLOW AREA < 1400 -> LINE TRACKING END / YAW HOLD START @@@")

                    if self.yaw is not None:
                        yaw_err = -1.56-self.yaw
                        yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                        angular = np.clip(yaw_err*3.0,-0.5,0.5)
                        self.move(0.15,angular)

                    return

            for contour in white_contours:
                area = cv2.contourArea(contour)

                if area < 30:
                    continue

                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                temp_white_objects.append((cx,cy,area))

            temp_yellow_objects.sort(key=lambda x:x[2],reverse=True)
            temp_white_objects.sort(key=lambda x:x[2],reverse=True)

            if temp_yellow_objects or temp_white_objects:
                yellow_objects = temp_yellow_objects
                white_objects = temp_white_objects
                break

        target_x = None

        if self.parking_signal == 0:
            if len(white_objects)>=1:
                cx_w,cy_w,_ = white_objects[0]
                target_x = cx_w-80

        elif self.parking_signal == 1:
            if len(yellow_objects)>=2:
                cx1,cy1,_ = yellow_objects[0]
                cx2,cy2,_ = yellow_objects[1]
                target_x = (cx1+cx2)/2.0+20
                rospy.loginfo("TRACK YELLOW 2")

            elif len(yellow_objects)==1:
                cx_y,cy_y,_ = yellow_objects[0]
                target_x = cx_y+100
                rospy.loginfo("TRACK YELLOW 1")

        if target_x is None:
            self.move(0.1,0)
            return

        center_x = width/2.0
        err_x = target_x-center_x

        linear = 8
        angular = np.clip(-float(err_x)/4.0,-35.0,35.0)
        wheel_distance = 0.2

        self.v_l = linear-angular*wheel_distance
        self.v_r = linear+angular*wheel_distance

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/4.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()

    def detect_lturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_l,_,_ = cv2.minMaxLoc(res)

    def detect_rturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_r,_,_ = cv2.minMaxLoc(res)

    def img_callback(self,data):
        image = self.bridge.imgmsg_to_cv2(data,'bgr8')

        # rospy.loginfo("LEFT: %.2fm FRONT: %.2f RIGHT: %.2f",self.left_distance,self.front_distance,self.right_distance)
        rospy.loginfo(self.mode)

        if self.mode == 'stop':
            if self.detect_light(image) != 'green':
                return

        if self.mode == 'lane' and self.gain == 1:
            self.sign = self.detect_sign(image)

            if self.after_parking == 1:
                if self.sign == 'left':
                    self.after_parking = 2
                    self.prev_yellow_x = None
                    rospy.loginfo("@@@ LEFT SIGN DETECTED -> TRACK LEFT YELLOW ONLY @@@")

            else:
                if self.sign == 'left':
                    self.mode = 'turn_left'
                    self.c_mode = 1
                    self.hide_yellow = 1
                    rospy.loginfo("@@@ TURN LEFT !!!! @@@")

                elif self.sign == 'right':
                    self.mode = 'turn_right'
                    self.c_mode = 1
                    self.hide_white = 1
                    rospy.loginfo("@@@ TURN RIGHT !!!! @@@")

        if self.c_mode == 1:
            if self.mode != 'construct':
                self.sign_c = self.detect_construct(image)

                if self.sign_c:
                    self.mode = 'construct'
                    self.hide_white = 0
                    self.hide_yellow = 0
                    rospy.loginfo("@@@ CONSTRUCTION SIGN DETECTED @@@")

            if self.mode == 'construct':
                self.avoid_obstacle(image)
                return

        if self.park_num == 1 and self.c_mode == 2:
            self.go_parking(image)
            return

        height,width = image.shape[:2]

        roi_list = [
            (int(height*0.55),height),
            (int(height*0.45),int(height*0.72)),
            (int(height*0.35),int(height*0.55))
        ]

        selected_white_mask = None
        selected_yellow_mask = None

        white_line = None
        yellow_line = None
        yellow_lines = []

        roi_start = 0
        crop_img = image

        for start_y,end_y in roi_list:
            crop_img = image[start_y:end_y,:]
            hsv_frame = cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

            white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
            yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

            if self.after_parking == 2:
                yellow_mask[:,width//2:] = 0
                white_mask[:,:] = 0

            elif self.after_parking == 1:
                white_mask[:,:] = 0

            elif self.hide_yellow == 1:
                yellow_mask[:,width//2:] = 0

            elif self.hide_white == 1:
                white_mask[:,:width//2] = 0

            elif self.hide_white == 2:
                white_mask[:,:width//2] = 0
                yellow_mask[:,:width//4] = 0

            elif self.hide_white == 3:
                white_mask[:,width//2:] = 0

            elif self.hide == 1:
                white_mask[:,:] = 0
                yellow_mask[:2*height//3,:] = 0

            white_mask = cv2.erode(white_mask,None,iterations=1)
            white_mask = cv2.dilate(white_mask,None,iterations=2)

            yellow_mask = cv2.erode(yellow_mask,None,iterations=1)
            yellow_mask = cv2.dilate(yellow_mask,None,iterations=2)

            white_candidate = self.find_line(white_mask,self.prev_white_x)
            yellow_candidate = self.find_line(yellow_mask,self.prev_yellow_x)

            if self.after_parking == 1:
                yellow_candidates = self.find_lines(yellow_mask)

                if len(yellow_candidates) >= 2:
                    selected_white_mask = white_mask
                    selected_yellow_mask = yellow_mask
                    yellow_lines = yellow_candidates
                    roi_start = start_y
                    break

            elif white_candidate is not None or yellow_candidate is not None:
                selected_white_mask = white_mask
                selected_yellow_mask = yellow_mask
                white_line = white_candidate
                yellow_line = yellow_candidate
                roi_start = start_y
                break

        display_image = image.copy()
        center_x = width/2.0
        target_x = None

        if self.after_parking == 1:
            if len(yellow_lines) >= 2:
                target_x = (yellow_lines[0][0]+yellow_lines[1][0])/2.0
                self.lost_count = 0
                rospy.loginfo("AFTER PARKING -> TRACK YELLOW 2")

        elif self.after_parking == 2:
            if yellow_line is not None:
                self.cx_y = yellow_line[0]
                self.prev_yellow_x = self.cx_y

                target_x = self.cx_y+80
                self.lost_count = 0
                rospy.loginfo("AFTER LEFT SIGN -> TRACK LEFT YELLOW ONLY")

        else:
            if white_line is not None:
                self.cx_w = white_line[0] 
                self.prev_white_x = self.cx_w

            if yellow_line is not None:
                self.cx_y = yellow_line[0]
                self.prev_yellow_x = self.cx_y

            if white_line is not None and yellow_line is not None:
                measured_lane_width = abs(self.cx_w-self.cx_y)

                if 100 < measured_lane_width < width:
                    self.lane_width = 0.9*self.lane_width+0.1*measured_lane_width

                target_x = (self.cx_w+self.cx_y)/2.0
                self.lost_count = 0

            elif white_line is not None:
                target_x = self.cx_w-self.lane_width/2.0 
                self.lost_count = 0

            elif yellow_line is not None:
                target_x = self.cx_y+self.lane_width/2.0 
                self.lost_count = 0

        if target_x is not None:
            error_x = target_x-center_x

            linear = 6
            angular = np.clip(-float(error_x)/5,-25.0,25.0)
            wheel_distance = 0.2

            self.v_l = linear-angular*wheel_distance*3
            self.v_r = linear+angular*wheel_distance*3

            self.last_v_l = self.v_l
            self.last_v_r = self.v_r

        elif self.lost_count < 8:
            self.lost_count += 1
            self.v_l = self.last_v_l*0.6
            self.v_r = self.last_v_r*0.6

        else:
            self.v_l = 0.0
            self.v_r = 0.0

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()

        if selected_white_mask is not None:
            white_view = cv2.cvtColor(selected_white_mask,cv2.COLOR_GRAY2BGR)
        else:
            white_view = np.zeros_like(display_image)

        if selected_yellow_mask is not None:
            yellow_view = cv2.cvtColor(selected_yellow_mask,cv2.COLOR_GRAY2BGR)
        else:
            yellow_view = np.zeros_like(display_image)

        view_w = 320
        view_h = 240

        output_view = cv2.resize(display_image,(view_w,view_h))
        crop_view = cv2.resize(crop_img,(view_w,view_h))
        white_view = cv2.resize(white_view,(view_w,view_h))
        yellow_view = cv2.resize(yellow_view,(view_w,view_h))

        top = cv2.hconcat([output_view,crop_view])
        bottom = cv2.hconcat([yellow_view,white_view])
        total_view = cv2.vconcat([top,bottom])

        cv2.imshow('ALL VIEW',total_view)
        cv2.waitKey(3)

    def find_line(self,mask,previous_x):
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 80:
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            candidates.append((cx,cy,area))

        if not candidates:
            return None

        if previous_x is None:
            return max(candidates,key=lambda x:x[2])

        return min(candidates,key=lambda x:abs(x[0]-previous_x))

    def find_lines(self,mask):
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 80:
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            candidates.append((cx,cy,area))

        return sorted(candidates,key=lambda x:x[2],reverse=True)

    def publish_velocity(self):
        self.cmd_pub.publish(self.msg)


if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()