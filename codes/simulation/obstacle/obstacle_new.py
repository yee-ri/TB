import rospy
import cv2
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)


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
        self.front_distance = 0
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
        self.scan_active = False
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image = rospy.Subscriber('/camera/image',Image,self.img_callback)
        self.scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom = rospy.Subscriber('/odom',Odometry,self.odom_callback)

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

    def scan_callback(self, data):

        front_ranges = []

        for i in range(len(data.ranges)):
            distance = data.ranges[i]

            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = data.angle_min + i * data.angle_increment
            angle_deg = np.degrees(angle)

            if angle_deg > 180:angle_deg -= 360

            if -15 <= angle_deg <= 15:front_ranges.append(distance)

        if len(front_ranges) == 0:
            return 0

        self.front_distance = min(front_ranges)

        
    def odom_callback(self,data):
        q = data.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _,_,yaw =  euler_from_quaternion(quaternion)
        self.yaw = yaw
        # rospy.loginfo("YAW is %5f",self.yaw)

    def set_yaw(self, num):
        if self.yaw is None:
            return

        if num == 1:
            target_yaw = 1.57

        elif num == 2:
            target_yaw = 3.14

        elif num == 3:
            target_yaw = 0.0

        else:
            return

        while not rospy.is_shutdown():
            err = target_yaw - self.yaw
            err = math.atan2(math.sin(err), math.cos(err))

            rospy.loginfo(
                "NUM: %d  YAW: %.3f  TARGET: %.3f  ERR: %.3f",
                num, self.yaw, target_yaw, err
            )

            if abs(err) < 0.02:
                self.move(0,0)
                return

            v_yaw = err * 5
            v_yaw = np.clip(v_yaw,-0.5,0.5)

            self.move(0,v_yaw)

            rospy.sleep(0.02)
    def dis_y(self,image):

        crop = image[150:,:]
        hsvFrame = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([50, 255, 255]) 
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
        M_y = cv2.moments(yellow_mask)
        if  M_y["m00"] > 0 :
            cy_y = int(M_y['m01'] / M_y['m00'])
        if cy_y < 63: 
            self.move(0.1,0) 
            if cy_y == 63 : return
    
    def dis_w(self,image):

        crop = image[150:,:]
        hsvFrame = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 
        white_lower = np.array([0, 0, 200])  
        white_upper = np.array([179, 50, 255])
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)
        M_y = cv2.moments(white_mask)
        if  M_y["m00"] > 0 :
            cy_w = int(M_y['m01'] / M_y['m00'])
        if cy_w < 55: 
            self.move(0.1,0) 
            if cy_w >= 55 : return




    def move(self,linear,angular):
        self.msg.linear.x=linear
        self.msg.linear.y=0
        self.msg.linear.z=0
        self.msg.angular.x=0
        self.msg.angular.y=0
        self.msg.angular.z=angular
        self.publish_velocity()
    
    def avoid_obstacle(self,image):
        
        if self.front_distance is None:
            return

        crop_imgage=image[186:,:]
        height,width=crop_imgage.shape[:2]
        hsvFrame=cv2.cvtColor(crop_imgage,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsvFrame,np.array([0,0,200]),np.array([179,50,255]))
        yellow_mask=cv2.inRange(hsvFrame,np.array([20,100,100]),np.array([50,255,255]))

        white_line=self.find_line(white_mask,self.prev_white_x)

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

        if M_w["m00"]==0:
            self.cx_w=0
        else:
            self.cx_w=int(M_w["m10"]/M_w["m00"])

        if M_y["m00"]==0:
            self.cx_y=0
        else:
            self.cx_y=int(M_y["m10"]/M_y["m00"])
        
        rospy.loginfo("DIST: %.2f  YELLOW: %d  WHITE: %d",self.front_distance,self.cx_y,self.cx_w)
        if self.front_distance<=0.4:
            self.move(0.1,0)
            if self.front_distance<=0.4:
                self.move(0.1,0)
                if self.front_distance<=0.2 :
                    self.move(0,0)
                    rospy.sleep(0.5)
                    
                    if 100 < self.cx_y and 170 <self.cx_w< 200:
                        # rospy.loginfo("DIST: %.2f  YELLOW: %d  WHITE: %d",self.front_distance,self.cx_y,self.cx_w)
                        rospy.loginfo(">?>>>???????????/")
                        # self.msg.linear.x=0
                        # self.msg.angular.z=0
                        # self.publish_velocity()
                        # rospy.sleep(5)

                        # self.msg.linear.x=0
                        # self.msg.angular.z=-1
                        # self.publish_velocity()
                        # rospy.sleep(2)

                        # self.msg.linear.x=3
                        # self.msg.angular.z=0
                        # self.publish_velocity()
                        # rospy.sleep(1.5)

                        # self.msg.linear.x=0
                        # self.msg.angular.z=1
                        # self.publish_velocity()
                        # rospy.sleep(2.0)

                        # self.msg.linear.x=1
                        # self.msg.angular.z=0
                        # self.publish_velocity()
                        # rospy.sleep(2)
                        # return

                    elif self.cx_y<20 and 170<self.cx_w :
                        # rospy.loginfo("DIST: %.2f  YELLOW: %d  WHITE: %d",self.front_distance,self.cx_y,self.cx_w)

                        self.move(0,0)
                        rospy.sleep(0.06)

                        yaw_dir = 2
                        self.set_yaw(yaw_dir)
                        self.dis_y(image)

                        # self.move(0.1,0) 
                        # rospy.sleep(2.6)

                        self.move(0,0)
                        rospy.sleep(0.02)
                        yaw_dir = 1
                        self.set_yaw(yaw_dir)

                        self.move(0.1,0)
                        rospy.sleep(4.5)

                        self.move(0,0)
                        rospy.sleep(0.06)
                        yaw_dir = 3
                        self.set_yaw(yaw_dir)  

                        self.move(0.1,0) 
                        rospy.sleep(2.7)

                        self.move(0,0)
                        rospy.sleep(0.06)
                        yaw_dir = 1
                        self.set_yaw(yaw_dir)
                                              
                        return

            return
        if white_line is not None:
            self.cx_w = white_line[0]
            self.prev_white_x = self.cx_w

            center_x = width / 2.0
            target_x = self.cx_w - 80

            error_x = target_x - center_x

            linear = 7
            angular = -float(error_x) / 2.5
            wheel_distance = 0.2

            angular = np.clip(angular, -20.0, 20.0)

            self.v_l = linear - angular * wheel_distance
            self.v_r = linear + angular * wheel_distance

            self.msg.linear.x = self.wheel_radius * (self.v_r + self.v_l) / 2.0
            self.msg.angular.z = self.wheel_radius * (self.v_r - self.v_l) / self.wheel_separation

            self.publish_velocity()

        else:
            self.msg.linear.x = 0
            self.msg.angular.z = 0
            self.publish_velocity()

    def detect_sign(self, data):
        self.detect_lturn(data)
        self.detect_rturn(data)

        rospy.loginfo("Lturn score: %.3f    Rturn score: %.3f", self.max_val_l, self.max_val_r)
        # if self.max_val_l > 0.3: return 'left'
        # elif self.max_val_r > 0.39: return 'right'
        if 0.1<self.front_distance <= 0.34:
            if self.max_val_l > self.max_val_r + 0.03 : return 'left'
            elif self.max_val_r > self.max_val_l+ 0.03 : return 'right'
                
        return None
    
    def detect_construct(self,data):
        # img = data[140:,:]

        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.construct_template,cv2.TM_CCOEFF_NORMED)
        min_val,self.max_val_c,min_loc,max_loc = cv2.minMaxLoc(res)
        rospy.loginfo("CONSTRUCTION VAL IS %.3f",self.max_val_c)
        if self.max_val_c >= 0.45 and 0.85< self.front_distance< 0.93:
            # rospy.loginfo("CONSTRUCTION VAL IS %.3f",self.max_val_c)
            return True
        # if self.max_val_c > 0.4: return True

        return False
               
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
        
    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        crop_img = image[90:,:]

        if self.front_distance is not None:
            rospy.loginfo("FRONT DISTANCE: %.2f m", self.front_distance)

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
                self.move(0,0)
                rospy.sleep(0.08)
                self.c_mode = 1
                self.hide_yellow = 1
    
                self.sign_count = 0
                rospy.loginfo('@@@ TURN LEFT !!!! @@@')

            elif self.sign == 'right':
                self.mode = 'turn_right'
                self.move(0,0)
                rospy.sleep(0.08)
                self.c_mode = 1
                self.hide_white = 1
                self.sign_count = 0
                rospy.loginfo('@@@ TURN RIGHT !!!! @@@')    

            self.prev_white_x = None
            self.prev_yellow_x = None
        # rospy.loginfo("C MODE IS %s    ",self.c_mode)


################################################################################################################
   
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
################################################################################################################           
                

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

        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask_all = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
        yellow_mask_all = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

        for start_y,end_y in roi_list:
            white_mask = white_mask_all[start_y:end_y,:].copy()
            yellow_mask = yellow_mask_all[start_y:end_y,:].copy()
            if self.hide_yellow == 1:
                yellow_mask[:, 3*width//5:] = 0
                # yellow_mask[3*height//5:, :] = 0
                yellow_mask[:2*height//5, :] = 0 
                # yellow_mask[:3*height//5, :] = 0 

                white_mask[:, width*4//6:] = 0
                        
            elif  self.hide_white == 1:
                white_mask[:, :width//2] = 0

  

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
            angular = -float(error_x) / 5
            wheel_distance = 0.2

            angular = np.clip(angular, -25.0, 25.0)

            self.v_l = linear - angular * wheel_distance * 3
            self.v_r = linear + angular * wheel_distance * 3

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


        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
 
        self.publish_velocity()
        rospy.loginfo("X_Y : %3f    X_W: %3f",self.cx_y,self.cx_w)
        cv2.imshow('output', display_image)

        if selected_white_mask is not None:
            cv2.imshow('white mask', selected_white_mask)

        if selected_yellow_mask is not None:
            cv2.imshow('yellow mask', selected_yellow_mask)

        cv2.imshow('crop image', crop_img)
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

        return max(candidates, key=lambda value: value[1])

    def publish_velocity(self):
       self.cmd_pub.publish(self.msg)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()
