import rospy
import cv2
import numpy as np
import math
import subprocess
import actionlib
import os
import yaml
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import UInt8

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.bridge = CvBridge()
        self.msg = Twist()

        BASE_DIR=os.path.dirname(os.path.abspath(__file__))

        self.lturn_template=cv2.imread(os.path.join(BASE_DIR,'images','lturn.jpg'),cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(7,7))

        self.rturn_template=cv2.imread(os.path.join(BASE_DIR,'images','rturn.jpg'),cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(7,7))

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
        self.ready_finish = None
        # self.nav_retry=0
        self.ready_gate = None
        self.stop = 0
        self.after_tunnel = None
        self.after_turn = None
        self.straight_count = 0
        self.straight = None
        self.parking_yellow_count = 0
        self.parking_yellow_enter_area = 1200
        self.parking_center_count = 0
        self.map_process=None
        self.tf_process=None
        self.amcl_process=None
        self.move_base_process=None
        self.cy_y = None
        self.nav_step=0
        self.nav_time=None
        self.yellow_area = None
        self.initialpose_pub=rospy.Publisher('/initialpose',PoseWithCovarianceStamped,queue_size=1,latch=True)
        self.move_base_client=actionlib.SimpleActionClient('move_base',MoveBaseAction)
        self.tunnel_in_time=None
        self.nav_started=False
        self.exit_x=None
        self.exit_y=None
        
        self.left_distance = float('inf')
        self.front_distance = float('inf')
        self.right_distance = float('inf')

        self.cmd_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub = rospy.Subscriber('/camera/image',Image,self.img_callback,queue_size=1,buff_size=2**24)
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

            if abs(err) < 0.02:
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
            self.cy_y = int(M_y["m01"]/M_y["m00"])
            rospy.loginfo("CY_Y is %d",self.cy_y)

            if self.cy_y < 39:
                self.move_yaw_hold(0.2,3.135)
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

            if cy_w < 41:
                self.move_yaw_hold(0.2,-0.02)
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

    def move_yaw_hold(self,linear,target_yaw):
        if self.yaw is None:
            self.move(linear,0)
            return

        yaw_err = target_yaw-self.yaw
        yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))

        angular = np.clip(yaw_err*3.0,-0.4,0.4)

        self.move(linear,angular)

        rospy.loginfo("MOVE YAW HOLD: YAW %.3f TARGET %.3f ERR %.3f ANG %.3f",self.yaw,target_yaw,yaw_err,angular)

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
            if self.front_distance > 0.26:
                self.move_yaw_hold(0.2,1.545)
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

        if self.front_distance <= 0.28:
            self.move(0.1,0)

            if self.front_distance <= 0.25:
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
        target_x = self.cx_w-95
        error_x = target_x-center_x

        linear = 8 #6
        angular = np.clip(-float(error_x)/3.5,-15.0,15.0)
        # angular = np.clip(-float(error_x)/2.5,-20.0,20.0)
        wheel_distance = 0.2

        self.v_l = linear-angular*wheel_distance*2
        self.v_r = linear+angular*wheel_distance*2

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()
        
    def detect_sign(self,data):
        self.detect_lturn(data)
        self.detect_rturn(data)

        rospy.loginfo("Lturn score: %.3f Rturn score: %.3f",self.max_val_l,self.max_val_r)

        if 0.1 < self.front_distance <= 0.47:
            if self.max_val_l > self.max_val_r+0.05 and self.max_val_l > 0.53 or self.max_val_l>0.65:#65
                return 'left'

            if self.max_val_r > self.max_val_l+0.05 and self.max_val_r > 0.53 or self.max_val_r>0.65 :
                return 'right'

        return None

    def detect_parking_yellow(self,data):
        hsv_frame = cv2.cvtColor(data,cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

        yellow_mask = cv2.erode(yellow_mask,None,iterations=1)
        yellow_mask = cv2.dilate(yellow_mask,None,iterations=2)

        contours,_ = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        yellow_objects = []
        max_yellow_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 40:
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            yellow_objects.append((cx,cy,area))
            max_yellow_area = max(max_yellow_area,area)

        yellow_objects.sort(key=lambda x:x[2],reverse=True)

        rospy.loginfo("PARKING CHECK YELLOW COUNT: %d AREA: %.1f",len(yellow_objects),max_yellow_area)

        if len(yellow_objects) >= 2 and max_yellow_area >= self.parking_yellow_enter_area:
            return True

        return False

    def go_parking(self,data):
        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f",self.left_distance,self.front_distance,self.right_distance)

        if self.parking_signal == 0:
            if self.detect_parking_yellow(data):
                self.parking_yellow_count += 1
            else:
                self.parking_yellow_count = 0

            rospy.loginfo("PARKING YELLOW COUNT: %d",self.parking_yellow_count)

            if self.parking_yellow_count >= 18:
                self.parking_signal = 1
                self.parking_yellow_count = 0
                self.parking_center_count = 0
                self.prev_yellow_x = None
                rospy.loginfo("@@@ YELLOW 2 DETECTED -> TRACK YELLOW CENTER @@@")

        if self.yaw_hold == 1 and self.alpha == 0:
            if 0.85 < self.left_distance < 1.1 and self.right_distance < 0.25 and 1.3 < self.front_distance < 1.6:
                self.alpha = 1
                self.move(0,0)
                rospy.loginfo("@@@ LEFT TB DETECTED @@@ PARKING SEQUENCE START @@@")
                rospy.sleep(0.04)
                return

            if 0.6  <= self.right_distance <= 0.95 and 1.2 <= self.front_distance <= 1.65 and self.left_distance < 0.28:
                self.alpha = 2
                self.move(0,0)
                rospy.loginfo("@@@ RIGHT TB DETECTED @@@ PARKING SEQUENCE START @@@")
                rospy.sleep(0.04)
                return

            if self.yaw is not None:
                yaw_err = -1.56-self.yaw
                yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                angular = np.clip(yaw_err*8.0,-0.61,0.61)

                self.move(0.15,angular)
                rospy.loginfo("YAW HOLD: %.3f TARGET: -1.560 ERR: %.3f",self.yaw,yaw_err)

            return

        if self.alpha == 1:
            if self.left_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.02)

                if self.set_yaw(2):
                    self.move(0.15,0)
                    rospy.sleep(1.7)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.6)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 3

        if self.alpha == 2:
            if self.right_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.04)

                if self.set_yaw(3):
                    self.move(0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.9)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 4

        if self.alpha == 3 or self.alpha == 4:
            if self.set_yaw(1):
                self.finish_parking()
                return

        height,width = data.shape[:2]
        data[:height//6,:] = 0

        roi_list = [
            (int(height*0.55),height),
            (int(height*0.45),int(height*0.72)),
            (int(height*0.35),int(height*0.55))
        ]

        yellow_objects = []
        white_objects = []
        max_yellow_area = 0

        for start_y,end_y in roi_list:
            crop = data[start_y:end_y,:]
            hsv_frame = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)

            yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))
            white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))

            if self.hide_white == 2:
                white_mask[:,:3*width//5] = 0

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

            for contour in yellow_contours:
                area = cv2.contourArea(contour)

                if area < 40:
                    continue

                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                temp_yellow_objects.append((cx,cy,area))
                max_yellow_area = max(max_yellow_area,area)

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

            temp_yellow_objects.sort(key=lambda x:x[0])
            temp_white_objects.sort(key=lambda x:x[2],reverse=True)

            if self.parking_signal == 0:
                if temp_white_objects:
                    white_objects = temp_white_objects
                    break

            elif self.parking_signal == 1:
                if len(temp_yellow_objects) >= 2:
                    yellow_objects = temp_yellow_objects
                    break

        target_x = None
        center_x = width/2.0

        if self.parking_signal == 0:
            if len(white_objects) >= 1:
                cx_w,cy_w,_ = white_objects[0]
                target_x = cx_w-80

        elif self.parking_signal == 1:
            if len(yellow_objects) >= 2:
                left_yellow = yellow_objects[0]
                right_yellow = yellow_objects[-1]

                cx_left = left_yellow[0]
                cx_right = right_yellow[0]

                yellow_center = (cx_left+cx_right)/2.0
                parking_target_x = center_x-20
                err_x = yellow_center-parking_target_x       

                # yellow_center = (cx_left+cx_right)/2.0
                # err_x = yellow_center-center_x

                rospy.loginfo(
                    "YELLOW 2 LEFT: %d RIGHT: %d CENTER: %.1f ERR: %.1f AREA: %.1f",
                    cx_left,cx_right,yellow_center,err_x,max_yellow_area
                )

                if abs(err_x) < 15:
                    self.parking_center_count += 1
                else:
                    self.parking_center_count = 0

                rospy.loginfo("PARKING CENTER COUNT: %d",self.parking_center_count)

                if self.parking_center_count >= 23:
                    self.yaw_hold = 1
                    self.parking_center_count = 0
                    rospy.loginfo("@@@ YELLOW CENTER OK -> YAW HOLD START @@@")
                    return

                if self.yaw is not None:
                    yaw_err = -1.56-self.yaw
                    yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))

                    yaw_angular = np.clip(yaw_err*1.0,-0.15,0.15)
                    center_angular = np.clip(-err_x*0.01,-0.6,0.6)

                    angular = center_angular+yaw_angular
                    angular = np.clip(angular,-0.65,0.65)

                    self.move(0.10,angular)

                    rospy.loginfo(
                        "YELLOW ALIGN YAW: %.3f YAW_ERR: %.3f CENTER_ERR: %.1f ANG: %.3f",
                        self.yaw,yaw_err,err_x,angular
                    )

                    return

            else:
                self.parking_center_count = 0

                if self.yaw is not None:
                    yaw_err = -1.56-self.yaw
                    yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                    angular = np.clip(yaw_err*1.0,-0.4,0.4)
                    self.move(0.10,angular)
                else:
                    self.move(0.10,0)

                return

        if target_x is None:
            self.move(0.1,0)
            return

        err_x = target_x-center_x

        linear = 6
        angular = np.clip(-float(err_x)/1.0,-35.0,35.0)
        wheel_distance = 0.2

        self.v_l = linear-angular*wheel_distance
        self.v_r = linear+angular*wheel_distance

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()

        cv2.imshow("go parking",data)
        cv2.imshow("WHITE",white_mask)
        cv2.waitKey(3)
  
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

        self.after_parking = 2

        self.hide_white = 0
        self.hide_yellow = 0

        self.prev_white_x = None
        self.prev_yellow_x = None

        self.ready_gate = 1

        rospy.loginfo("@@@ PARKING END -> TRACK YELLOW 2 @@@")

    def gate(self):
        rospy.loginfo("left: %3f   front: %3f   right: %3f", self.left_distance,self.front_distance,self.right_distance)

        if self.stop == 0:
            if 1.7< self.front_distance < 2.28 and 5<self.right_distance :
                rospy.loginfo("GATE CLOSED -> STOP")
                self.move(0,0)
                rospy.sleep(11)
                self.stop = 1
                return False

            return None

        if self.stop == 1:
            if self.front_distance < 2.2:
                rospy.loginfo("GATE OPEN -> GO")
                self.stop = 0
                self.mode = 'tunnel'
                return True

            self.move(0,0)
            return False

    def detect_lturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_l,_,_ = cv2.minMaxLoc(res)

    def detect_rturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_r,_,_ = cv2.minMaxLoc(res)

    def tunnel_in(self):
        if self.yaw is None:return

        rate=rospy.Rate(20)

        while not rospy.is_shutdown():
            err=-1.57-self.yaw
            err=math.atan2(math.sin(err),math.cos(err))

            if abs(err)<0.03:
                self.move(0,0)
                break

            angular=np.clip(err*3.0,-0.4,0.4)
            self.move(0,angular)
            rate.sleep()

        start=rospy.get_time()

        while not rospy.is_shutdown() and rospy.get_time()-start<3:
            rospy.loginfo("!@!@!@!@!")
            self.move_yaw_hold(0.2,0)
            rate.sleep()

        self.move(0,0)
        self.mode='tunnel'
  
    def find_exit(self):
        BASE_DIR=os.path.dirname(os.path.abspath(__file__))

        pgm_path=os.path.join(BASE_DIR,'turtlebot3_simulations','maze','tunnel.pgm')
        yaml_path=os.path.join(BASE_DIR,'turtlebot3_simulations','maze','tunnel.yaml')
        with open(yaml_path,'r') as f:
            info=yaml.safe_load(f)
        res=info['resolution']
        origin_x=info['origin'][0]
        origin_y=info['origin'][1]
        img=cv2.imread(pgm_path,cv2.IMREAD_GRAYSCALE)

        if img is None:
            rospy.logerr("TUNNEL MAP LOAD FAILED")
            return False
        occupied=np.zeros_like(img)
        occupied[img<50]=255
        contours,_=cv2.findContours(occupied,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        points=[]

        for contour in contours:
            if cv2.arcLength(contour,False)<20:
                continue
            approx=cv2.approxPolyDP(contour,3,False)
            for p in approx:
                x=int(p[0][0])
                y=int(p[0][1])
                duplicated=False
                for qx,qy in points:
                    if math.hypot(x-qx,y-qy)<5:
                        duplicated=True
                        break
                if not duplicated:
                    points.append((x,y))
        openings=[]
        for i in range(len(points)):
            for j in range(i+1,len(points)):
                x1,y1=points[i]
                x2,y2=points[j]
                width=math.hypot(x2-x1,y2-y1)*res
                if 0.3<width<0.6:
                    cx=(x1+x2)/2.0
                    cy=(y1+y2)/2.0
                    wx=origin_x+cx*res
                    wy=origin_y+(img.shape[0]-cy)*res
                    openings.append((wx,wy))

        if len(openings)<2:
            rospy.logerr("EXIT DETECTION FAILED")
            return False
        
        start_x=-1.6761
        start_y=-0.674
        openings.sort(key=lambda p:math.hypot(p[0]-start_x,p[1]-start_y))

        entrance=openings[0]
        exit_point=openings[-1]
        self.exit_x=exit_point[0]
        self.exit_y=exit_point[1]
        rospy.loginfo("ENTRANCE: %.3f %.3f",entrance[0],entrance[1])
        rospy.loginfo("EXIT: %.3f %.3f",self.exit_x,self.exit_y)
        return True

    def start_tunnel_navigation(self):
        rospy.loginfo("@@@ FIND EXIT START @@@")
        if not self.find_exit():
            rospy.logerr("EXIT DETECTION FAILED")
            return False
        rospy.loginfo("@@@ EXIT FOUND: %.3f %.3f @@@",self.exit_x,self.exit_y)
        self.map_process=subprocess.Popen(['rosrun','map_server','map_server','/home/sj/catkin_ws/src/turtlebot3_simulations/maze/tunnel.yaml'])
        self.tf_process=subprocess.Popen(['rosrun','tf2_ros','static_transform_publisher','0','0','0','0','0','0','base_footprint','base_scan'])
        self.amcl_process=subprocess.Popen(['rosrun','amcl','amcl','_base_frame_id:=base_footprint','_odom_frame_id:=odom','_global_frame_id:=map','_transform_tolerance:=0.5'])
        env=os.environ.copy()
        env['TURTLEBOT3_MODEL']='burger'
        self.move_base_process=subprocess.Popen(['roslaunch','turtlebot3_navigation','move_base.launch'],env=env)
        self.nav_time=rospy.Time.now()
        self.nav_step=1
        return True

    def tunnel(self):
        rospy.loginfo("NAV STEP: %d",self.nav_step)
        if self.nav_step==1:
            rospy.loginfo("WAIT INITIALPOSE")
            if (rospy.Time.now()-self.nav_time).to_sec()<3.0:
                return
            self.set_initial_pose()
            rospy.loginfo("INITIAL POSE SENT")
            self.nav_time=rospy.Time.now()
            self.nav_step=2
            return
        
        if self.nav_step==2:
            rospy.loginfo("WAIT MOVE BASE")
            if (rospy.Time.now()-self.nav_time).to_sec()<2.0:
                return
            if not self.move_base_client.wait_for_server(rospy.Duration(0.1)):
                rospy.loginfo("MOVE BASE NOT READY")
                return
            self.send_tunnel_goal()
            rospy.loginfo("TUNNEL GOAL SENT")
            self.nav_step=3
            return
        
        if self.nav_step==3:
            rospy.loginfo("MOVE BASE RUNNING")
            if self.yellow_area is not None and 200<=self.yellow_area<=250:
                self.move_base_client.cancel_goal()
                self.move(0,0)

                if self.set_yaw(3):
                    self.mode='lane'
                    self.after_tunnel=1
                    self.hide_white=4
                    self.prev_yellow_x=None
                    self.nav_started=False
                    self.nav_step=0
                    self.ready_finish=1
                    rospy.loginfo("@@@ TUNNEL END -> YAW 0 -> TRACK YELLOW @@@")
                return
            return

    def set_initial_pose(self):
        msg=PoseWithCovarianceStamped()
        msg.header.stamp=rospy.Time.now()
        msg.header.frame_id='map'
        msg.pose.pose.position.x=-1.6761
        msg.pose.pose.position.y=-0.674
        msg.pose.pose.position.z=0.0
        msg.pose.pose.orientation.x=0.0
        msg.pose.pose.orientation.y=0.0
        msg.pose.pose.orientation.z=-0.673
        msg.pose.pose.orientation.w=0.739
        msg.pose.covariance[0]=0.25
        msg.pose.covariance[7]=0.25
        msg.pose.covariance[35]=0.068
        self.initialpose_pub.publish(msg)

    def send_tunnel_goal(self):
        goal=MoveBaseGoal()
        goal.target_pose.header.frame_id='map'
        goal.target_pose.header.stamp=rospy.Time.now()
        goal.target_pose.pose.position.x=-0.352
        goal.target_pose.pose.position.y=-1.7
        goal.target_pose.pose.position.z=0.0
        goal.target_pose.pose.orientation.x=0.0
        goal.target_pose.pose.orientation.y=0.0
        goal.target_pose.pose.orientation.z=-0.015
        goal.target_pose.pose.orientation.w=1.0
        self.move_base_client.send_goal(goal)

    def img_callback(self,data):
        image = self.bridge.imgmsg_to_cv2(data,'bgr8')

        rospy.loginfo(self.mode)
        height,width = image.shape[:2]

        tunnel_roi = image[int(height*0.35):,:]
        hsv = cv2.cvtColor(tunnel_roi,cv2.COLOR_BGR2HSV)

        tunnel_yellow_mask = cv2.inRange( hsv, np.array([20,100,100]), np.array([50,255,255]) )

        self.yellow_area = cv2.countNonZero(tunnel_yellow_mask)
        rospy.loginfo("left: %3f   front: %3f   right: %3f", self.left_distance,self.front_distance,self.right_distance)
        rospy.loginfo(self.yellow_area)
        if self.mode == 'stop':
            if self.detect_light(image) != 'green':
                return

        if self.mode == 'lane' and self.gain == 1 and self.after_parking == 0:
            self.sign = self.detect_sign(image)

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


        if self.ready_gate == 1 and self.mode == 'gate':
            end = self.gate()

            if end == False:
                return

            if end == True:
                self.ready_gate = 0
            
        if self.c_mode == 1:
            if self.mode != 'construct':
                self.straight = self.is_straight(image)

                if self.straight:
                    self.mode = 'construct'
                    self.hide_white = 0
                    self.hide_yellow = 0
                    rospy.loginfo("@@@ CONSTRUCTION START @@@")

            if self.mode == 'construct':
                self.avoid_obstacle(image)
                return

        if self.park_num == 1 and self.c_mode == 2:
            self.go_parking(image)
            return


        if self.mode=='tunnel' and self.nav_started==False:
            rospy.loginfo("TUNNEL WAIT LEFT: %.2f FRONT: %.2f RIGHT: %.2f YELLOW: %d",self.left_distance,self.front_distance,self.right_distance,self.yellow_area)

            if (0.1<=self.left_distance<=0.15 and 0.2<=self.front_distance<=0.45 and 0.55<=self.right_distance<=0.65 and self.yellow_area<=0):

                rospy.loginfo("@@@ TUNNEL ENTRANCE DETECTED @@@")

                self.mode='tunnel_in'
                self.tunnel_in_time=rospy.Time.now()
                return


        if self.mode=='tunnel_in':
            self.move_yaw_hold(0.2,-1.3)
            if (rospy.Time.now()-self.tunnel_in_time).to_sec()>=2.0:
                self.move(0,0)
                if not self.start_tunnel_navigation():
                    rospy.logerr("TUNNEL NAVIGATION START FAILED")
                    return
                self.nav_started=True
                self.mode='tunnel'
                rospy.loginfo("@@@ TUNNEL NAVIGATION START @@@")
            return


        if self.mode=='tunnel' and self.nav_started==True:
            self.tunnel()
            return

        rospy.loginfo("IMERE")

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

            elif self.hide_yellow == 1: # TURN LEFT 반말고 왼쪽 노란거 1/4 날리기
                yellow_mask[:,3*width//4:] = 0
                white_mask[:,: ] = 0

            elif self.hide_white == 1: # TURN RIGHT 흰거 왼쪽 날리기
                white_mask[:,:width//2] = 0
                yellow_mask[:,: ] = 0

            elif self.hide_white == 2:
                # white_mask[:,:width//2] = 0
                # white_mask[:,:5*width//8] = 0
                white_mask[:,:3*width//4] = 0
                yellow_mask[:,:width//4] = 0

            elif self.hide_yellow == 2:
                white_mask[:,width//2:] = 0
                yellow_mask[:,width//2:] = 0

            elif self.hide_white == 3: # 오른쪽 흰색 날리기
                white_mask[:,width//2:] = 0

            elif self.hide_white == 4: # 흰색 나가리
                white_mask[:,:] = 0
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

        if self.after_parking == 2:
            if self.yaw is not None and 3.08 < self.yaw < 3.18:
                self.after_parking = 3
                self.mode = 'gate'
                self.hide_yellow = 0
                self.hide_white = 0
                self.c_mode = 0

        if self.after_parking == 3:
            self.hide_yellow = 0
            self.hide_white = 0

        if self.after_parking == 2:
            if yellow_line is not None:
                self.cx_y = yellow_line[0]
                self.prev_yellow_x = self.cx_y

                target_x = self.cx_y+90
                self.lost_count = 0
                rospy.loginfo("AFTER LEFT SIGN -> TRACK LEFT YELLOW ONLY")


        if self.after_tunnel==1:
            self.hide_white=4

            if yellow_line is not None:
                self.cx_y=yellow_line[0]
                self.prev_yellow_x=self.cx_y
                target_x=self.cx_y+self.lane_width/2.0 - 35
                self.lost_count=0
                rospy.loginfo("AFTER TUNNEL -> TRACK YELLOW")

        else: # 회전 후, 주차 전 
            if white_line is not None:
                self.cx_w = white_line[0] 
                self.prev_white_x = self.cx_w

            if yellow_line is not None:
                self.cx_y = yellow_line[0]
                self.prev_yellow_x = self.cx_y

                
            if self.mode == 'turn_left' and self.straight != True:
                if yellow_line is not None:
                    target_x = self.cx_y+85
                    self.lost_count = 0
                    rospy.loginfo("TURN LEFT -> LEFT YELLOW ONLY")

            elif self.mode == 'turn_right' and self.straight != True:
                if white_line is not None:
                    target_x = self.cx_w-90
                    self.lost_count = 0


            elif white_line is not None and yellow_line is not None:
                measured_lane_width = abs(self.cx_w-self.cx_y)

                if 100 < measured_lane_width < width:
                    self.lane_width = 0.9*self.lane_width+0.1*measured_lane_width

                target_x = (self.cx_w+self.cx_y)/2.0
                self.lost_count = 0

            elif white_line is not None: # 흰선만 보이면
                target_x = self.cx_w-self.lane_width/2.0
                self.lost_count = 0

            elif yellow_line is not None:
                if self.after_tunnel == None or self.after_tunnel == 0:
                    target_x = self.cx_y+self.lane_width/2.0
                elif self.after_tunnel == 1:
                    target_x = self.cx_y+self.lane_width/2.0 - 25
                self.lost_count = 0

        if target_x is not None:
            error_x = target_x-center_x

            
            
            wheel_distance = 0.2

            if self.after_parking <= 2:
                linear =6.5
                angular = np.clip(-float(error_x)/3.5,-15.0,15.0)
            if self.after_parking ==3:
                rospy.loginfo(self.after_parking)
                linear = 6
                angular = np.clip(-float(error_x)/3.5,-15.0, 15.0)

            self.v_l = linear-angular*wheel_distance*2
            self.v_r = linear+angular*wheel_distance*2

            self.last_v_l = self.v_l
            self.last_v_r = self.v_r

        elif self.lost_count < 8:
            self.lost_count += 1
            self.v_l = self.last_v_l*0.9
            self.v_r = self.last_v_r*0.6

        else:
            self.v_l = 0.0
            self.v_r = 0.0

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
        self.publish_velocity()

        # if selected_white_mask is not None:
        #     white_view = cv2.cvtColor(selected_white_mask,cv2.COLOR_GRAY2BGR)
        # else:
        #     white_view = np.zeros_like(display_image)

        # if selected_yellow_mask is not None:
        #     yellow_view = cv2.cvtColor(selected_yellow_mask,cv2.COLOR_GRAY2BGR)
        # else:
        #     yellow_view = np.zeros_like(display_image)
        # rospy.loginfo("HIDE is %f",self.hide_white)
        # view_w = 320
        # view_h = 240

        # # output_view = cv2.resize(display_image,(view_w,view_h))
        # # crop_view = cv2.resize(crop_img,(view_w,view_h))
        # # white_view = cv2.resize(white_view,(view_w,view_h))
        # # yellow_view = cv2.resize(yellow_view,(view_w,view_h))

        # # top = cv2.hconcat([output_view,crop_view])
        # # bottom = cv2.hconcat([yellow_view,white_view])
        # # total_view = cv2.vconcat([top,bottom])

        # # cv2.imshow('ALL VIEW',total_view)
        # # cv2.waitKey(3)

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
    
    def is_straight(self,image):
        height,width = image.shape[:2]

        roi = image[int(height*0.35):,:]
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv,np.array([0,0,200]),np.array([179,50,255]))
        yellow_mask = cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))

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
            white_mask[:,:7*width//8] = 0
            yellow_mask[:,:width//4] = 0

        elif self.hide_white == 3:
            white_mask[:,width//2:] = 0

        mask = cv2.bitwise_or(white_mask,yellow_mask)

        mask = cv2.erode(mask,None,iterations=1)
        mask = cv2.dilate(mask,None,iterations=2)

        edges = cv2.Canny(mask,50,150)

        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=40,maxLineGap=20)

        if lines is None:
            rospy.loginfo("STRAIGHT CHECK: NO LINE")
            return False

        angles = []

        for line in lines:
            x1,y1,x2,y2 = line[0]

            dx = x2-x1
            dy = y2-y1

            if abs(dy) < 20:
                continue

            angle = math.degrees(math.atan2(dx,dy))

            if abs(angle) < 60:
                angles.append(angle)

        if len(angles) < 1.5:
            rospy.loginfo("STRAIGHT CHECK: LINE COUNT %d",len(angles))
            return False

        angle_std = np.std(angles)
        mean_angle = np.mean(angles)

        rospy.loginfo("STRAIGHT CHECK MEAN: %.2f STD: %.2f LINES: %d",mean_angle,angle_std,len(angles))

        if angle_std < 5.0:
                        
            self.straight_count += 1
        else:
            self.straight_count = 0

        if self.straight_count >= 40: return True

        rospy.loginfo("STRAIGHT COUNT: %d",self.straight_count)

        return False
     

    def publish_velocity(self):
        self.cmd_pub.publish(self.msg)


if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()
