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
        self.lturn_template = cv2.resize(self.lturn_template,(10,10))
        self.rturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/rturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.rturn_template = cv2.resize(self.rturn_template,(10,10))
        self.construct_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/construct.png',cv2.IMREAD_GRAYSCALE)
        self.construct_template = cv2.resize(self.construct_template,(60,60)) # 대충 0.4는 넘음
        self.parking_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/parking.png',cv2.IMREAD_GRAYSCALE)
        self.parking_template = cv2.resize(self.parking_template,(20,20))
        self.hide = 0
        self.park_num =0 
        self.v_l = 0
        self.v_r = 0
        self.gain = 1
        self.wheel_radius = 0.033
        self.wheel_separation = 0.16
        self.lane_width = 285.0
        self.step = 0
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
        self.parking_signal =0
        self.scan_active = False
        self.alpha = 0
        self.site = 0
        self.keep_w = 0
        self.yaw_hold = 0
        self.yaw = None
        self.left_distance = float('inf')
        self.front_distance = float('inf')
        self.right_distance = float('inf')
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image = rospy.Subscriber('/camera/image',Image,self.img_callback)
        self.scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom = rospy.Subscriber('/odom',Odometry,self.odom_callback)

    # def detect_light(self,data):
    #     hsv_frame = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

    #     green_mask = cv2.inRange(hsv_frame, np.array([35, 50, 50]), np.array([85, 255, 255]))
    #     green_mask = cv2.erode(green_mask, None, iterations=1)
    #     green_mask = cv2.dilate(green_mask, None, iterations=2)
    #     green_pixel = cv2.countNonZero(green_mask)
    #     # rospy.loginfo ("%.3f",green_pixel)

    #     if green_pixel >= 200:
    #         rospy.loginfo("DETECTED GREEN LIGHT")
    #         self.light = 'green'
    #         self.mode = 'lane'
    #         return 'green'

    def scan_callback(self,data):
        left_ranges = []
        front_ranges = []
        right_ranges = []

        for i in range(len(data.ranges)):
            distance = data.ranges[i]

            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = data.angle_min + i*data.angle_increment
            angle_deg = np.degrees(angle)

            if angle_deg > 180:
                angle_deg -= 360

            if 15 <= angle_deg <= 60:
                right_ranges.append(distance)

            elif -15 <= angle_deg <= 15:
                front_ranges.append(distance)

            elif -60 <= angle_deg <= -15:
                left_ranges.append(distance)

        self.left_distance = min(left_ranges) if len(left_ranges) > 0 else float('inf')
        self.front_distance = min(front_ranges) if len(front_ranges) > 0 else float('inf')
        self.right_distance = min(right_ranges) if len(right_ranges) > 0 else float('inf')

        rospy.loginfo( "LEFT: %.2f  FRONT: %.2f  RIGHT: %.2f",self.left_distance,self.front_distance, self.right_distance)

        
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
            target_yaw = 1.545

        elif num == 2:
            target_yaw = 3.135

        elif num == 3:
            target_yaw = -0.02

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
                rospy.sleep(0.03)
                return 1

            v_yaw = err * 8
            v_yaw = np.clip(v_yaw,-1.5,1.5)

            self.move(0,v_yaw)



    def move(self,linear,angular):
        self.msg.linear.x=linear
        self.msg.linear.y=0
        self.msg.linear.z=0
        self.msg.angular.x=0
        self.msg.angular.y=0
        self.msg.angular.z=angular
        self.publish_velocity()


    def detect_parking(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.parking_template,cv2.TM_CCOEFF_NORMED)
        min_val,self.max_val_p,min_loc,max_loc = cv2.minMaxLoc(res)
        rospy.loginfo("Parking VAL IS %.3f",self.max_val_p)
        if 0.325 <= self.max_val_p  and 0.5 < self.front_distance < 0.6:
            rospy.loginfo("DETECT PARKING SIGN!!!!!")
            return True
        return False
        
    def go_parking(self,data):
        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f",self.left_distance,self.front_distance,self.right_distance)

        if self.parking_signal == 0:
            parking_sign = self.detect_parking(data)

            if parking_sign:
                self.parking_signal = 1
                self.keep_w = 1
                self.site = 0

        if self.yaw_hold == 1 and self.alpha == 0:
            if 1.0 < self.left_distance < 1.2 and self.right_distance < 0.9 and 1.6 < self.front_distance < 1.7:
                self.alpha = 1
                self.move(0,0)
                rospy.loginfo("@@@ LEFT PARKING SEQUENCE START @@@")
                rospy.sleep(3)
                return

            elif 0.9 < self.right_distance < 1.0 and 1.6 < self.front_distance < 1.7 and 0.2 < self.left_distance < 0.25:
                self.alpha = 2
                self.move(0,0)
                rospy.loginfo("@@@ RIGHT PARKING SEQUENCE START @@@")
                rospy.sleep(3)
                return

            if self.yaw is not None:
                target_yaw = -1.56
                yaw_err = target_yaw-self.yaw
                yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                angular = yaw_err*3.0
                angular = np.clip(angular,-0.5,0.5)
                self.move(0.15,angular)
                rospy.loginfo("YAW HOLD: %.3f TARGET: -1.560 ERR: %.3f",self.yaw,yaw_err)

            return

        if self.alpha == 1:
            if self.left_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.02)

                yaw_dir = 2
                num = self.set_yaw(yaw_dir)
                rospy.loginfo("@@@@@ NUM IS %f @@@@@",num)

                if num:
                    self.move(0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 3

        if self.alpha == 3:
            yaw_dir = 1
            num = self.set_yaw(yaw_dir)
            self.move(0,0)
            rospy.sleep(0.04)

            if num:
                self.sign = self.detect_sign(data)

                if self.sign == 'left':
                    self.mode = 'turn_left'
                    self.c_mode = 0
                    self.park_num = 0
                    self.hide_yellow = 1
                    self.parking_signal = 0
                    self.site = 0
                    self.keep_w = 0
                    self.yaw_hold = 0
                    self.alpha = 0
                    rospy.loginfo('@@@ TURN LEFT !!!! @@@')
                    return

        if self.alpha == 2:
            if self.right_distance <= 1.1:
                self.move(0,0)
                rospy.sleep(0.04)

                yaw_dir = 3
                num = self.set_yaw(yaw_dir)
                rospy.loginfo("@@@@@ NUM IS %f @@@@@",num)

                if num:
                    self.move(0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.move(-0.15,0)
                    rospy.sleep(1.8)
                    self.move(0,0)
                    rospy.sleep(0.08)
                    self.alpha = 4

        if self.alpha == 4:
            yaw_dir = 1
            num = self.set_yaw(yaw_dir)
            self.move(0,0)
            rospy.sleep(2)

            if num:
                self.sign = self.detect_sign(data)

                if self.sign == 'left':
                    self.mode = 'turn_left'
                    self.c_mode = 0
                    self.park_num = 0
                    self.hide_yellow = 1
                    self.parking_signal = 0
                    self.site = 0
                    self.keep_w = 0
                    self.yaw_hold = 0
                    self.alpha = 0
                    rospy.loginfo('@@@ TURN LEFT !!!! @@@')
                    return

        height,width = data.shape[:2]

        roi_list = [
            (int(height*0.55),height),
            (int(height*0.45),int(height*0.72)),
            (int(height*0.35),int(height*0.55))
        ]

        selected_yellow_mask = None
        selected_white_mask = None
        yellow_objects = []
        white_objects = []
        roi_start = 0

        for start_y,end_y in roi_list:
            crop = data[start_y:end_y,:]
            hsvFrame = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)

            yellow_mask = cv2.inRange(hsvFrame,np.array([20,100,100]),np.array([50,255,255]))
            white_mask = cv2.inRange(hsvFrame,np.array([0,0,200]),np.array([179,50,255]))

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

                if area > max_yellow_area:
                    max_yellow_area = area

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
                        target_yaw = -1.56
                        yaw_err = target_yaw-self.yaw
                        yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))
                        angular = yaw_err*3.0
                        angular = np.clip(angular,-0.5,0.5)
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

            temp_yellow_objects = sorted(temp_yellow_objects,key=lambda x:x[2],reverse=True)
            temp_white_objects = sorted(temp_white_objects,key=lambda x:x[2],reverse=True)

            if len(temp_yellow_objects) > 0 or len(temp_white_objects) > 0:
                yellow_objects = temp_yellow_objects
                white_objects = temp_white_objects
                selected_yellow_mask = yellow_mask
                selected_white_mask = white_mask
                roi_start = start_y
                break

        target_x = None

        if self.parking_signal == 0:
            if len(white_objects) >= 1:
                cx_w,cy_w,_ = white_objects[0]
                target_x = cx_w-80
                rospy.loginfo("BEFORE PARKING -> TRACK WHITE")

        elif self.parking_signal == 1:
            if len(yellow_objects) >= 2:
                cx1,cy1,_ = yellow_objects[0]
                cx2,cy2,_ = yellow_objects[1]
                target_x = (cx1+cx2)/2.0+20
                rospy.loginfo("TRACK YELLOW 2")

            elif len(yellow_objects) == 1:
                cx_y,cy_y,_ = yellow_objects[0]
                target_x = cx_y+100
                rospy.loginfo("TRACK YELLOW 1")

        if target_x is None:
            self.move(0.1,0)

            if selected_yellow_mask is not None:
                cv2.imshow("yellow mask",selected_yellow_mask)

            if selected_white_mask is not None:
                cv2.imshow("white mask",selected_white_mask)

            cv2.imshow("parking",data)
            cv2.waitKey(3)
            return

        center_x = width/2.0
        err_x = target_x-center_x

        linear = 5
        angular = -float(err_x)/4.0
        wheel_distance = 0.2

        angular = np.clip(angular,-15.0,15.0)

        self.v_l = linear-angular*wheel_distance
        self.v_r = linear+angular*wheel_distance

        self.msg.linear.x = self.wheel_radius*(self.v_r+self.v_l)/4.0
        self.msg.angular.z = self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation

        self.publish_velocity()

        if selected_yellow_mask is not None:
            cv2.imshow("yellow mask",selected_yellow_mask)

        if selected_white_mask is not None:
            cv2.imshow("white mask",selected_white_mask)

        cv2.imshow("parking",data)
        cv2.waitKey(3)

    def detect_lturn(self,data):
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        min_val,self.max_val_l,min_loc,max_loc = cv2.minMaxLoc(res)

        return False
        
    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.crop_img = image[50:,:]

        # if self.front_distance is not None:
        #     rospy.loginfo("FRONT DISTANCE: %.2f m", self.front_distance)

        # if self.mode == 'stop':
        #     self.light = self.detect_light(image)

        #     if self.light == 'green':
        #         self.mode = 'lane'
        #     else:
        #         return

        # if self.mode == 'lane' and self.gain == 1:
        #     self.sign = self.detect_sign(image)
        #     self.detect_construct(image)
        #     if self.sign == 'left':
        #         self.mode = 'turn_left'
        #         self.c_mode = 1
        #         self.hide_yellow = 1
        #         rospy.loginfo('@@@ TURN LEFT !!!! @@@')

        #     elif self.sign == 'right':
        #         self.mode = 'turn_right'
        #         self.c_mode = 1
        #         self.hide_white = 1
        #         rospy.loginfo('@@@ TURN RIGHT !!!! @@@')    

        #     self.prev_white_x = None
        #     self.prev_yellow_x = None

        # if self.c_mode == 1:

        #     if self.mode != 'construct':
        #         self.sign_c = self.detect_construct(image)

        #         if self.sign_c:
        #             self.mode = 'construct'
        #             self.hide_white = 0
        #             self.hide_yellow = 0
        #             rospy.loginfo("@@@ CONSTRUCTION SIGN DETECTED @@@")

        #     if self.mode == 'construct':
        #         self.avoid_obstacle(image)
        #         return
        self.park_num = 1 
        self.c_mode = 2

            
#####################################################################
        if self.park_num == 1 and self.c_mode == 2:
            self.go_parking(image)
            return
#####################################################################




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

        for start_y, end_y in roi_list:
            crop_img = image[start_y:end_y, :]
            hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            white_mask = cv2.inRange(hsv_frame, np.array([0, 0, 200]), np.array([179, 50, 255]))
            yellow_mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([50, 255, 255]))

            if  self.hide_yellow == 1:   ### 노란선 왼쪽 날리기
                yellow_mask[:, width//2:] = 0
                rospy.loginfo("Hide YELLOW ")
                
            elif  self.hide_white == 1:  ### 흰선 오른쪽 날리기
                white_mask[:, :width//2] = 0
                rospy.loginfo("Hide WHITE ")

            elif  self.hide_white == 2:   ### 흰선 오른쪽 날리기 + 노란선 오른쪽 1/4 날리기
                white_mask[:, :width//2] = 0
                yellow_mask[:, :1*width//4] = 0
                rospy.loginfo("Hide WHITE & YELLOW")

            elif  self.hide_white == 3:   ### 흰선 왼쪽 날리기
                white_mask[:,width//2 :] = 0
                rospy.loginfo("Hide WHITE !!")

            elif self.hide == 1:
                white_mask[: ,:] = 0
                yellow_mask[: 2*height//3,:] = 0
                rospy.loginfo("Hide WHITE & YELLOW HALF")



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
            angular = -float(error_x) / 4
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
        rospy.loginfo("IM HERE")

        # if selected_white_mask is not None:
        #     white_view = cv2.cvtColor(selected_white_mask, cv2.COLOR_GRAY2BGR)
        # else:
        #     white_view = np.zeros_like(display_image)

        # if selected_yellow_mask is not None:
        #     yellow_view = cv2.cvtColor(selected_yellow_mask, cv2.COLOR_GRAY2BGR)
        # else:
        #     yellow_view = np.zeros_like(display_image)

        # crop_view = crop_img.copy()

        # view_w = 320
        # view_h = 240

        # output_view = cv2.resize(display_image, (view_w, view_h))
        # crop_view = cv2.resize(crop_view, (view_w, view_h))
        # white_view = cv2.resize(white_view, (view_w, view_h))
        # yellow_view = cv2.resize(yellow_view, (view_w, view_h))


        # top = cv2.hconcat([output_view, crop_view])
        # bottom = cv2.hconcat([yellow_view,white_view])

        # total_view = cv2.vconcat([top, bottom])

        # cv2.imshow('ALL VIEW', total_view)
        # cv2.waitKey(3)

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
       self.cmd_pub.publish(self.msg)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()

