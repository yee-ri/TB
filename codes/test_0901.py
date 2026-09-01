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


    # def avoid_obstacle(self,image):
    #     if self.step == 1:
    #         if self.dis_y(image):
    #             self.move(0,0)
    #             rospy.sleep(0.02)
    #             self.step = 2
    #         return

    #     if self.step == 2:
    #         if self.set_yaw(1):
    #             self.move(0,0)
    #             rospy.sleep(0.04)
    #             self.step = 3
    #         return

    #     if self.step == 3:
    #         if self.front_distance > 0.26:
    #             self.move_yaw_hold(0.2,1.545)
    #         else:
    #             self.move(0,0)
    #             rospy.sleep(0.02)
    #             self.step = 4
    #         return

    #     if self.step == 4:
    #         if self.set_yaw(3):
    #             self.move(0,0)
    #             rospy.sleep(0.02)
    #             self.step = 5
    #         return

    #     if self.step == 5:
    #         if self.dis_w(image):
    #             self.move(0,0)
    #             rospy.sleep(0.02)
    #             self.step = 6
    #         return

    #     if self.step == 6:
    #         if self.set_yaw(1):
    #             self.move(0,0)
    #             rospy.sleep(0.02)
    #             self.step = 7
    #             self.gain = 0
    #             self.mode = 'lane'
    #             self.c_mode = 2
    #             self.hide_white = 2
    #             self.park_num = 1
    #         return

    #     crop_image = image[186:,:]
    #     height,width = crop_image.shape[:2]
    #     hsv_frame = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)

    #     white_mask = cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
    #     yellow_mask = cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

    #     white_line = self.find_line(white_mask,self.prev_white_x)

    #     M_w = cv2.moments(white_mask)
    #     M_y = cv2.moments(yellow_mask)

    #     self.cx_w = int(M_w["m10"]/M_w["m00"]) if M_w["m00"] > 0 else 0
    #     self.cx_y = int(M_y["m10"]/M_y["m00"]) if M_y["m00"] > 0 else 0

    #     rospy.loginfo("DIST: %.2f YELLOW: %d WHITE: %d",self.front_distance,self.cx_y,self.cx_w)

    #     if self.front_distance <= 0.28:
    #         self.move(0.1,0)

    #         if self.front_distance <= 0.25:
    #             self.move(0,0)
    #             rospy.sleep(0.2)

    #             if self.cx_y < 20 and self.cx_w > 170 and self.step == 0:
    #                 self.move(0,0)
    #                 rospy.sleep(0.02)

    #                 if self.set_yaw(2):
    #                     self.move(0,0)
    #                     rospy.sleep(0.02)
    #                     self.step = 1

    #                 return

    #         return

    #     if white_line is None:
    #         self.move(0,0)
    #         return

    #     self.cx_w = white_line[0]
    #     self.prev_white_x = self.cx_w

    #     center_x = width/2.0
    #     target_x = self.cx_w-95
    #     error_x = target_x-center_x

    #     linear = 8 #6
    #     angular = np.clip(-float(error_x)/3.5,-15.0,15.0)
    #     # angular = np.clip(-float(error_x)/2.5,-20.0,20.0)
    #     wheel_distance = 0.2

    #     self.v_l = linear-angular*wheel_distance*2
    #     self.v_r = linear+angular*wheel_distance*2

    #     self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/2.0
    #     self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation
    #     self.publish_velocity()
        
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

        rospy.loginfo(self.mode)
        height,width = image.shape[:2]

        # tunnel_roi = image[int(height*0.35):,:]
        # hsv = cv2.cvtColor(tunnel_roi,cv2.COLOR_BGR2HSV)

        # tunnel_yellow_mask = cv2.inRange( hsv, np.array([20,100,100]), np.array([50,255,255]) )

        # self.yellow_area = cv2.countNonZero(tunnel_yellow_mask)
        rospy.loginfo("left: %3f   front: %3f   right: %3f", self.left_distance,self.front_distance,self.right_distance)
        # rospy.loginfo(self.yellow_area)
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


        # if self.ready_gate == 1 and self.mode == 'gate':
        #     end = self.gate()

        #     if end == False:
        #         return

        #     if end == True:
        #         self.ready_gate = 0
            
        # if self.c_mode == 1:
        #     if self.mode != 'construct':
        #         self.straight = self.is_straight(image)

        #         if self.straight:
        #             self.mode = 'construct'
        #             self.hide_white = 0
        #             self.hide_yellow = 0
        #             rospy.loginfo("@@@ CONSTRUCTION START @@@")

        #     if self.mode == 'construct':
        #         self.avoid_obstacle(image)
        #         return

        # if self.park_num == 1 and self.c_mode == 2:
        #     self.go_parking(image)
        #     return


        # if self.mode=='tunnel' and self.nav_started==False:
        #     rospy.loginfo("TUNNEL WAIT LEFT: %.2f FRONT: %.2f RIGHT: %.2f YELLOW: %d",self.left_distance,self.front_distance,self.right_distance,self.yellow_area)

        #     if (0.1<=self.left_distance<=0.15 and 0.2<=self.front_distance<=0.45 and 0.55<=self.right_distance<=0.65 and self.yellow_area<=0):

        #         rospy.loginfo("@@@ TUNNEL ENTRANCE DETECTED @@@")

        #         self.mode='tunnel_in'
        #         self.tunnel_in_time=rospy.Time.now()
        #         return


        # if self.mode=='tunnel_in':
        #     self.move_yaw_hold(0.2,-1.3)
        #     if (rospy.Time.now()-self.tunnel_in_time).to_sec()>=2.0:
        #         self.move(0,0)
        #         if not self.start_tunnel_navigation():
        #             rospy.logerr("TUNNEL NAVIGATION START FAILED")
        #             return
        #         self.nav_started=True
        #         self.mode='tunnel'
        #         rospy.loginfo("@@@ TUNNEL NAVIGATION START @@@")
        #     return


        # if self.mode=='tunnel' and self.nav_started==True:
        #     self.tunnel()
        #     return

        # rospy.loginfo("IMERE")

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

        # if self.after_parking == 1:
        #     if len(yellow_lines) >= 2:
        #         target_x = (yellow_lines[0][0]+yellow_lines[1][0])/2.0
        #         self.lost_count = 0
        #         rospy.loginfo("AFTER PARKING -> TRACK YELLOW 2")

        # if self.after_parking == 2:
        #     if self.yaw is not None and 3.08 < self.yaw < 3.18:
        #         self.after_parking = 3
        #         self.mode = 'gate'
        #         self.hide_yellow = 0
        #         self.hide_white = 0
        #         self.c_mode = 0

        # if self.after_parking == 3:
        #     self.hide_yellow = 0
        #     self.hide_white = 0

        # if self.after_parking == 2:
        #     if yellow_line is not None:
        #         self.cx_y = yellow_line[0]
        #         self.prev_yellow_x = self.cx_y

        #         target_x = self.cx_y+90
        #         self.lost_count = 0
        #         rospy.loginfo("AFTER LEFT SIGN -> TRACK LEFT YELLOW ONLY")


        # if self.after_tunnel==1:
        #     self.hide_white=4

        #     if yellow_line is not None:
        #         self.cx_y=yellow_line[0]
        #         self.prev_yellow_x=self.cx_y
        #         target_x=self.cx_y+self.lane_width/2.0 - 35
        #         self.lost_count=0
        #         rospy.loginfo("AFTER TUNNEL -> TRACK YELLOW")

        # else: # 회전 후, 주차 전 
        #     if white_line is not None:
        #         self.cx_w = white_line[0] 
        #         self.prev_white_x = self.cx_w

        #     if yellow_line is not None:
        #         self.cx_y = yellow_line[0]
        #         self.prev_yellow_x = self.cx_y

                
        #     if self.mode == 'turn_left' and self.straight != True:
        #         if yellow_line is not None:
        #             target_x = self.cx_y+85
        #             self.lost_count = 0
        #             rospy.loginfo("TURN LEFT -> LEFT YELLOW ONLY")

        #     elif self.mode == 'turn_right' and self.straight != True:
        #         if white_line is not None:
        #             target_x = self.cx_w-90
        #             self.lost_count = 0


        #     elif white_line is not None and yellow_line is not None:
        #         measured_lane_width = abs(self.cx_w-self.cx_y)

        #         if 100 < measured_lane_width < width:
        #             self.lane_width = 0.9*self.lane_width+0.1*measured_lane_width

        #         target_x = (self.cx_w+self.cx_y)/2.0
        #         self.lost_count = 0

        #     elif white_line is not None: # 흰선만 보이면
        #         target_x = self.cx_w-self.lane_width/2.0
        #         self.lost_count = 0

        #     elif yellow_line is not None:
        #         if self.after_tunnel == None or self.after_tunnel == 0:
        #             target_x = self.cx_y+self.lane_width/2.0
        #         elif self.after_tunnel == 1:
        #             target_x = self.cx_y+self.lane_width/2.0 - 25
        #         self.lost_count = 0

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
