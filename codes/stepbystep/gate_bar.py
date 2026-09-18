#!/usr/bin/env python
import rospy
import cv2
import math
import numpy as np
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image,PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.step_sub=rospy.Subscriber('/step',UInt8,self.step_callback) #rostopic pub사용

        self.cmd_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw',Image,self.img_callback,queue_size=1,buff_size=2**24)
        self.scan_sub=rospy.Subscriber('/livox/lidar',PointCloud2,self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom',Odometry,self.odom_callback)
        self.bridge=CvBridge()

        rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)
  
        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn1.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))
        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn1.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

############## 구간별 테스트 설정 ##############
        self.step= 7 #(원래 0으로 세팅)
        self.yaw = 0.0
        self.initial_yaw=None

############## 라이다 관련 변수 ##############
        self.left_distance=float('inf')
        self.front_distance=float('inf')
        self.right_distance=float('inf')

############## 회전 관련 변수 ##############
        self.turn_detect=None #회전표지판 감지
        self.turn_direction=None # 회전명령 @@@@@@
        self.turn_end = None

############## 라인 트랙킹 관련 변수 ##############
        self.prev_error=0.0
        self.prev_target_x=None
        self.prev_v_l=0.0
        self.prev_v_r=0.0

        self.wheel_radius=0.0475
        self.wheel_separation=0.19 #0.148이었던것

############## tf setting 관련 변수 ##############
        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer)
        self.robot_half_width=0.08# 0.085
        self.safety_margin=0.03

############## 장애물 관련 변수 ##############
        self.obstacle_direction=None
        self.obstacle_pass_count=0
        self.obstacle_clear_count=0
        self.obstacle_points=[]
        self.white_area = None

############## 직선 구간 관련 변수 ##############
        self.straight_count = 0
        self.straight = None

############## 주차 관련 변수 ##############
        self.parking_ready = None
        self.parking_end = None
        self.yellow_area = None
        self.set = None
        self.parking_count = 0
        self.signal = 0
        self.sequence = 0
        self.parking_side=None      

############## 게이트바 관련 변수 ##############
        self.gate_state = 0
        self.gate_open_count = 0
        self.gate_close_count=0
        self.red_area = 0
        self.depth = 0

    def step_callback(self,data): # 그냥 rostopic pub으로 구간별 테스트하려고 만든거
        self.step=data.data
        self.prev_error=0.0
        self.prev_target_x=None
        self.prev_v_l=0.0
        self.prev_v_r=0.0

        if self.step<=2:
            
            pass

        if self.step==3:
            self.obstacle_direction=None
            self.obstacle_clear_count=0

        rospy.loginfo("@@@@@@ STEP CHANGE -> %d @@@@@@",self.step)

    def odom_callback(self,data):
        q=data.pose.pose.orientation
        _,_,raw_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])

        if self.initial_yaw is None:
            self.initial_yaw=raw_yaw

        self.yaw=raw_yaw-self.initial_yaw
        self.yaw=math.atan2(math.sin(self.yaw),math.cos(self.yaw))

    def scan_callback(self,data): # 라이다 
        try:
            transform=self.tf_buffer.lookup_transform('base_footprint',data.header.frame_id,rospy.Time(0),rospy.Duration(0.1))
            cloud=tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(data,transform)
        except Exception as e:
            rospy.logwarn("TF ERROR: %s",e)
            return

        self.obstacle_points=[]

        left_ranges=[]
        front_ranges=[]
        right_ranges=[]

        safe_width=self.robot_half_width+self.safety_margin
        side_limit=0.55

        for point in point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True):
            x,y,z=point

            if -0.15<x<0.8 and abs(y)<side_limit:
                self.obstacle_points.append((x,y))

            if x<=0.1:
                continue

            if z>0.30:
                continue

            if abs(y)<safe_width:
                front_ranges.append(x)
            elif safe_width<=y<side_limit:
                left_ranges.append(np.sqrt(x*x+y*y))
            elif -side_limit<y<=-safe_width:
                right_ranges.append(np.sqrt(x*x+y*y))

        self.left_distance=min(left_ranges) if left_ranges else float('inf')
        self.front_distance=min(front_ranges) if front_ranges else float('inf')
        self.right_distance=min(right_ranges) if right_ranges else float('inf')

        # rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f DIR: %s",self.left_distance,self.front_distance,self.right_distance,str(self.obstacle_direction))
        # rospy.loginfo("Front : %.2f",self.front_distance)
    def stop(self): # 초록불 감지 전 정지상태
        self.cmd_pub.publish(Twist())

    def traffic_light(self,data): # 초록불 감지 함수
        hsv_frame=cv2.cvtColor(data,cv2.COLOR_BGR2HSV)

        green_mask=cv2.inRange(hsv_frame,np.array([45,100,80]),np.array([85,255,255]))
        green_mask=cv2.erode(green_mask,None,iterations=1)
        green_mask=cv2.dilate(green_mask,None,iterations=2)

        if cv2.countNonZero(green_mask)>=200:
            rospy.loginfo("@@@@@@ GREEN LIGHT @@@@@@")
            self.stop()
            rospy.sleep(0.5)
            self.step=1
            rospy.loginfo("@@@@@@ STEP 1 @@@@@@")
            return True

        self.stop()
        return False

    def detect_sign(self,image): # 좌/우회전 감지, self.turn_detect값 반환
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        yellow_mask=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))
        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179, 55, 255]))
        # white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        blue_mask=cv2.inRange(hsv,np.array([90,80,50]),np.array([130,255,255]))

        mask=cv2.bitwise_or(yellow_mask,white_mask)
        mask=cv2.bitwise_or(mask,blue_mask)

        filtered=cv2.bitwise_and(image,image,mask=mask)
        gray=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)

        left_result=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,max_val_l,_,_=cv2.minMaxLoc(left_result)

        right_result=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,max_val_r,_,_=cv2.minMaxLoc(right_result)

        rospy.loginfo("LEFT: %.1f%% RIGHT: %.1f%%",max_val_l*100,max_val_r*100)

        if (max_val_l>max_val_r+0.05 and max_val_l>0.3) or max_val_l>0.315:
            return 'left'

        if (max_val_r>max_val_l+0.05 and max_val_r>0.3) or max_val_r>0.315:
            return 'right'

        return None

    def check_turn(self,image): # self.turn_detect들어오면 회전명령 수행
        sign=self.detect_sign(image)

        if self.turn_detect is None :
            if sign=='left' and self.front_distance<=0.55:
                self.turn_detect='left'
                rospy.loginfo("@@@@@@ LEFT SIGN DETECTED @@@@@@")

            elif sign=='right' and self.front_distance<=0.55:
                self.turn_detect='right'
                rospy.loginfo("@@@@@@ RIGHT SIGN DETECTED @@@@@@")


        if self.turn_detect is not None and self.front_distance<0.33:
            self.turn_direction=self.turn_detect
            self.turn_detect=None
            self.prev_error=0.0
            self.prev_target_x=None

            if self.turn_direction=='left':
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left',1,0.03,0.4)
                

            elif self.turn_direction=='right':
                rospy.loginfo("@@@@@@ TURN RIGHT @@@@@@")
                self.turn_time('right',1,0.03,0.4)
                
            if self.step ==1:
                self.step=2
            if self.step ==6:
                self.step = 7
            rospy.loginfo("@@@@@@ TURN END  @@@@@@")
            self.turn_end = 1
            
            return True

        return False

    def turn_time(self,direction,duration,linear_speed,angular_speed): # 회전인식->회전명령 되면 회전 수행하는 함수
        msg=Twist()
        msg.linear.x=linear_speed

        if direction=='left':
            msg.angular.z=angular_speed

        elif direction=='right':
            msg.angular.z=-angular_speed

        start_time=rospy.Time.now()
        rate=rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed=(rospy.Time.now()-start_time).to_sec()

            if elapsed>=duration:
                break

            self.cmd_pub.publish(msg)
            rate.sleep()

        self.stop()

    def is_straight(self,image,iter): # 직선구간인지 판별
        height,width = image.shape[:2]

        roi = image[int(height*0.35):,:]
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

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
            # rospy.loginfo("STRAIGHT CHECK: LINE COUNT %d",len(angles))
            return False

        angle_std = np.std(angles)
        mean_angle = np.mean(angles)

        # rospy.loginfo("STRAIGHT CHECK MEAN: %.2f STD: %.2f LINES: %d",mean_angle,angle_std,len(angles))

        if angle_std < 5.0:
                        
            self.straight_count += 1
        else:
            self.straight_count = 0

        if self.straight_count >= iter : return True

        rospy.loginfo("STRAIGHT COUNT: %d",self.straight_count)

        return False

    def check_obstacle_start(self): # 정면에 장애물 검출되면 장애물 구간 시작으로 인식
        if self.front_distance<=0.32:
            self.step=3
            self.turn_direction=None
            self.obstacle_direction=None
            self.obstacle_clear_count=0
            rospy.loginfo("@@@@@@ OBSTACLE START -> STEP 3 @@@@@@")
            return True

        return False

    def obstacle_move(self,image): # 대충 장애물 피하기
        height,width=image.shape[:2]
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,130]),np.array([179,80,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

        bottom=4*height//5
        left_yellow_b=yellow_mask[bottom:,:width//2]
        right_white_b=white_mask[bottom:,width//2:]

        yellow=yellow_mask[height//2:,:]
        white=white_mask[height//2:,:width//2]

        yellow_area_b=cv2.countNonZero(left_yellow_b)
        white_area_b=cv2.countNonZero(right_white_b)

        yellow_area=cv2.countNonZero(yellow)
        self.white_area=cv2.countNonZero(white)
        rospy.loginfo(self.white_area)

        #########################################
        # yellow_full=np.zeros((height,width),dtype=np.uint8)
        # white_full=np.zeros((height,width),dtype=np.uint8)
        # left_yellow_b_full=np.zeros((height,width),dtype=np.uint8)
        # right_white_b_full=np.zeros((height,width),dtype=np.uint8)

        # yellow_full[height//2:,:]=yellow
        # white_full[height//2:,width//2:]=white
        # left_yellow_b_full[bottom:,:width//2]=left_yellow_b
        # right_white_b_full[bottom:,width//2:]=right_white_b

        # cv2.putText(yellow_full,'yellow',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
        # cv2.putText(white_full,'white',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
        # cv2.putText(left_yellow_b_full,'left_yellow_b',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
        # cv2.putText(right_white_b_full,'right_white_b',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

        # top=cv2.hconcat([yellow_full,white_full])
        # bottom_view=cv2.hconcat([left_yellow_b_full,right_white_b_full])
        # mask_view=cv2.vconcat([top,bottom_view])
        # mask_view=cv2.resize(mask_view,None,fx=0.4,fy=0.4)

        # cv2.imshow('obstacle_masks',mask_view)
        # cv2.imshow('image',image)
        # cv2.waitKey(1)
        #########################################        

        # rospy.loginfo("Y_b: %3f W_b: %3f",yellow_area_b,white_area_b)
        rospy.loginfo("Y: %3f W: %3f",yellow_area,self.white_area)

        line_area=2000
        safe_width=self.robot_half_width+self.safety_margin

        near_points=[]
        front_points=[]

        for x,y in self.obstacle_points:
            if -0.05<x<0.4 and abs(y)<safe_width+0.06: #side_width + 0.06
                near_points.append((x,y))

            if 0<x<0.35 and abs(y)<safe_width:
                front_points.append((x,y))

        # rospy.loginfo("DIR:%s LEFT:%.2f FRONT:%.2f RIGHT:%.2f",str(self.obstacle_direction),self.left_distance,self.front_distance,self.right_distance)

        #################################################################################
        # if white_area>8100:
        #     if 0<= white_area_b:
        #         rospy.loginfo("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #         self.obstacle_clear_count +=1
        #     else :self.obstacle_clear_count = 0

        # if self.obstacle_clear_count >=1:
        #     self.step = 5
        #     return

        if self.obstacle_direction is None:
            if self.right_distance<0.25 and self.front_distance<=0.25 and self.left_distance>1.0 and white_area_b>4000:
                self.obstacle_direction='left'
                rospy.loginfo("RIGHT+FRONT BLOCKED -> TURN LEFT")
                self.move(0.01,0.35)
                # return False

            #대충반대
            elif self.left_distance<0.23 and self.front_distance<=0.23 and self.right_distance>1.0 and yellow_area_b>3000:
                self.obstacle_direction='right'
                rospy.loginfo("LEFT+FRONT BLOCKED -> TURN RIGHT")
                self.move(0.01,-0.35)
                # return False

            elif self.left_distance>self.right_distance:
                self.obstacle_direction = 'left'
                self.move(0.01,0.35)
                rospy.loginfo("RIGHT+FRONT BLOCKED -> TURN LEFT22")
            elif self.right_distance>self.left_distance:
                self.obstacle_direction='right'
                rospy.loginfo("LEFT+FRONT BLOCKED -> TURN RIGHT22")
                self.move(0.01,-0.35)

                #####################################################################
        if self.right_distance<0.25 and self.left_distance<=0.25 :
            if yellow_area>6000:
                self.obstacle_direction='left'
                rospy.loginfo("slow left")
                self.move(0.01,0.05)
                
            if self.white_area>6000:
                self.obstacle_direction='right'
                rospy.loginfo("slow right")
                self.move(0.01,-0.05)
            #####################################################

        if self.obstacle_direction=='left': # 오른쪽 장애물 있어서 왼쪽으로 이동하자

            if self.front_distance<0.22 :
                rospy.loginfo("front obstalce too close -> more fast turn left")
                self.move(0.01,0.3)

            elif front_points: #안전영역 안에 뭔가 계속 감지되면, 걍 계속 장애물 회피하는 겨
                rospy.loginfo("keep going turn left")
                self.move(0.015,0.2)


            elif yellow_area>12000: #self.front_distance>0.40 and # 왼쪽으로 가다가 노란선 넘 많이 보이면
                ##근데 뭔가 front point나 right distance 제한도 넣어야할듯
                self.obstacle_direction='right'
                rospy.loginfo("Yellow too close!!!! turn RIGHT")
                self.move(0.02,-0.15)

            else:
                if self.front_distance<=0.4 and white_area_b<yellow_area_b<13000: # 대충 값 보고 바꿔야함 linearea랑 100000 둘다
                    rospy.loginfo("obstacle detected !!! turn right start !!!")
                    self.obstacle_direction='right'
                    self.move(0.018,-0.20)
                else: # 일단 넣어봄,,
                    rospy.loginfo("IDK (left)")
                    self.move(0.06,0.05)



        elif self.obstacle_direction=='right': # 왼쪽에장애물 있어서 오른쪽으로 가기 또는 노란선 넘 가까워서 오르ㅜㄴ쪽회전

            if 4000<yellow_area<8000: #self.front_distance>0.40 and # 오른쪽으로 가다가 흰선 넘 많이 보이면
                self.obstacle_direction='right'
                rospy.loginfo("yellow too close!!!! turn small right")
                self.move(0.02,-0.025)
            elif self.left_distance>1 and self.front_distance>1 and self.right_distance<0.2:

                self.obstacle_direction='right'
                rospy.loginfo("yellow and obstacle too close!!!! turn small right22")
                self.move(0.02,-0.01)

            elif self.white_area>7000 and self.front_distance>=0.25 : #self.front_distance>0.40 and # 오른쪽으로 가다가 흰선 넘 많이 보이면
                self.obstacle_direction='left'
                rospy.loginfo("White too close!!!! turn left")
                self.move(0.02,0.1)


            elif front_points: #안전영역 안에 뭔가 계속 감지되면, 걍 계속 장애물 회피하는 겨
                rospy.loginfo("keep going turn right")
                self.move(0.015,-0.13)
                

            elif  self.right_distance<=0.18:
                rospy.loginfo("right obstacle too close!!!! turn left")
                self.obstacle_direction = 'left'
                self.move(0.02,0.01) 
            

            else:
                if self.front_distance<=0.4 and line_area<white_area_b<100000: # 대충 값 보고 바꿔야함 linearea랑 100000 둘다
                    rospy.loginfo("obstacle detected !!! turn left start !!!")
                    self.obstacle_direction='left'
                    self.move(0.018,0.20)
                else: # 일단 넣어봄,,
                    rospy.loginfo("IDK (right)")
                    self.move(0.06,-0.18)

    def move(self,linear,angular): # 장애물 피할때 선속도/각속도 설정하는 부분
        msg=Twist()
        msg.linear.x=linear
        msg.angular.z=angular
        self.cmd_pub.publish(msg)

    def check_obstacle_end(self): # 장애물 구간 끝났는지 확인
        if self.front_distance>0.9 and self.left_distance>0.5 and self.right_distance>0.5 and self.white_area>=200:
            self.obstacle_clear_count+=1
            rospy.loginfo("OBSTACLE CLEAR COUNT: %d",self.obstacle_clear_count)
        else:
            self.obstacle_clear_count=0

        if self.obstacle_clear_count>=10:
            self.step=4
            self.obstacle_clear_count=0
            self.obstacle_direction=None
            rospy.loginfo("@@@@@@ OBSTACLE END -> STEP 5 @@@@@@")
            return True

        return False

    def lane_tracking(self,image): #기본적인 라인 트래킹 수행
        crop_img=image[300:,:]
        height,width=crop_img.shape[:2]

        hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179, 55, 255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
        # red_mask = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([10, 255, 160]))        # mask=cv2.bitwise_or(red_mask)
   
        # rospy.loginfo("TURN DETECT(SIGN) is %s",str(self.turn_detect))

        if self.turn_direction=='left':
            yellow_mask[:,width//2:]=0

        elif self.turn_direction=='right':
            white_mask[:,:width//2]=0

        elif self.turn_direction=='parking':
            yellow_mask[:height//2,:]=0
            yellow_mask[:,width//2:]=0
            white_mask[:,:]=0

        elif self.step == 2 or self.step == 3:
            white_mask[:,:width//2]=0
        else:pass

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

        # self.red_area = cv2.countNonZero(red_mask)
        self.yellow_area=cv2.countNonZero(yellow_mask)
        white_area=cv2.countNonZero(white_mask)

        

        center_x=width/2.0
        target_x=None

        if M_w['m00']>0 and M_y['m00']>0:
            cx_w=int(M_w['m10']/M_w['m00'])
            cx_y=int(M_y['m10']/M_y['m00'])

            h=white_mask.shape[0]

            white_upper=white_mask[:h//2,:]
            white_lower=white_mask[h//2:,:]
            yellow_upper=yellow_mask[:h//2,:]
            yellow_lower=yellow_mask[h//2:,:]

            Mw_u=cv2.moments(white_upper)
            Mw_l=cv2.moments(white_lower)
            My_u=cv2.moments(yellow_upper)
            My_l=cv2.moments(yellow_lower)

            target_x=(cx_w+cx_y)/2.0

            if Mw_u['m00']>0 and Mw_l['m00']>0 and My_u['m00']>0 and My_l['m00']>0:
                cx_w_u=int(Mw_u['m10']/Mw_u['m00'])
                cx_w_l=int(Mw_l['m10']/Mw_l['m00'])
                cx_y_u=int(My_u['m10']/My_u['m00'])
                cx_y_l=int(My_l['m10']/My_l['m00'])

                white_curve=abs(cx_w_u-cx_w_l)
                yellow_curve=abs(cx_y_u-cx_y_l)

        elif M_w['m00']>0:
            cx_w=int(M_w['m10']/M_w['m00'])

            h=white_mask.shape[0]
            upper_mask=white_mask[:h//2,:]
            lower_mask=white_mask[h//2:,:]

            M_upper=cv2.moments(upper_mask)
            M_lower=cv2.moments(lower_mask)

            offset=300

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)

                if curve>190:
                    offset+=min((curve-190)*0.82,100)

            target_x=cx_w-offset

        elif M_y['m00']>0:
            cx_y=int(M_y['m10']/M_y['m00'])

            h=yellow_mask.shape[0]
            upper_mask=yellow_mask[:h//2,:]
            lower_mask=yellow_mask[h//2:,:]

            M_upper=cv2.moments(upper_mask)
            M_lower=cv2.moments(lower_mask)

            offset=310

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)

                if curve>170:
                    offset+=min((curve-170)*0.8,100)

            target_x=cx_y+offset

        if target_x is not None:
            if self.prev_target_x is None:
                self.prev_target_x=target_x
            else:
                target_x=0.25*target_x+0.75*self.prev_target_x
                self.prev_target_x=target_x

            err_x=target_x-center_x
            diff_x=err_x-self.prev_error

            Kp=0.15
            Kd=0.008
            linear=14.0
            max_angular=38.0

            angular=-(Kp*err_x+Kd*diff_x)
            angular=np.clip(angular,-max_angular,max_angular)

            self.prev_error=err_x

            wheel_distance=0.148

            v_l=linear-angular*wheel_distance*0.5
            v_r=linear+angular*wheel_distance*0.5

            self.prev_v_l=v_l
            self.prev_v_r=v_r

            self.publish_velocity(v_l,v_r)

        else:
            if self.turn_direction=='parking':
                self.move(0.06,0)
                return

            self.publish_velocity(self.prev_v_l,self.prev_v_r)

        mask_view=cv2.vconcat([white_mask,yellow_mask])
        cv2.imshow('MASK VIEW',mask_view)
        cv2.waitKey(1)

    def parking(self,image): #노란2선 직선인식 후, 주차 시퀀스->뒤돌아서 다시 빠져나오기 수행까지
        rospy.loginfo("IMHERE")

        if self.signal==0:

            if self.set ==2 : # 오른쪽에 뭐있음 -->정지-> 왼쪽으로회전(0)--> 출구쪽 바라봄(1.57)
                self.signal=1
                self.parking_side='right'

            elif self.set ==3:    
                self.signal=1
                self.parking_side='left'

            else:
                return

            rospy.loginfo("PARKING DETECTED: %s",self.parking_side)

            rospy.loginfo("@@@@@@@@@@@@@@@@@@@@@")

            # self.move_time(0.04,0,1)
            self.move(0,0)
            rospy.sleep(0.5)

        if self.parking_side=='right':
            self.move(0,0)
            rospy.loginfo("IM HERE")

            if self.set_yaw(3): # yaw 3 is 0.0
                self.move_time(0.04,0,2.5)
                rospy.sleep(0.04)
                self.move_time(-0.04,0,2.1)
                rospy.sleep(0.04)
                self.sequence=2

        elif self.parking_side=='left':
            self.move(0,0)
            rospy.loginfo("IM HERE22")

            if self.set_yaw(2): # yaw 2 is 3.14
                self.move_time(0.04,0,2.5)
                rospy.sleep(0.04)
                self.move_time(-0.04,0,2.1)
                rospy.sleep(0.04)
                self.sequence=2

        if self.parking_end==None and self.sequence==2:
            if self.set_yaw(1):
                self.parking_end=1
                self.step=6
                return

    def set_yaw(self,num):
        if self.yaw is None:
            return

        if num == 1:
            target_yaw = 1.565
        elif num == 2:
            target_yaw = -3.095
        elif num == 3:
            target_yaw = -0.05
        else:
            return

        while not rospy.is_shutdown():
            err = target_yaw-self.yaw
            err = math.atan2(math.sin(err),math.cos(err))

            rospy.loginfo("NUM: %d YAW: %.3f TARGET: %.3f ERR: %.3f",num,self.yaw,target_yaw,err)

            if abs(err) < 0.01 :
                self.move(0,0)
                rospy.sleep(0.03)
                return 1

            v_yaw = np.clip(err*8,-0.25,0.25)
            self.move(0,v_yaw)

    def move_time(self,linear,angular,duration):
        start=rospy.Time.now()
        rate=rospy.Rate(20)

        while not rospy.is_shutdown():
            if (rospy.Time.now()-start).to_sec()>=duration:
                break

            self.move(linear,angular)
            rate.sleep()

        self.move(0,0)

    def depth_callback(self,data):
        dep=self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')

        height,width=dep.shape[:2]

        roi=dep[:int(height*2/3),:int(width*3/4)]
        # roi=depth[y1:y2,x1:x2]

        valid=roi[(roi>0)&np.isfinite(roi)]

        if len(valid)==0:
            rospy.loginfo("DEPTH: NO DATA")
            return

        self.depth=np.percentile(valid,5)

        if data.encoding=='16UC1':
            self.depth/=1000.0

        rospy.loginfo("CLOSEST FRONT DEPTH: %.3f m",self.depth)


    def img_callback(self,data): # 메인코드
        image=self.bridge.imgmsg_to_cv2(data,'bgr8')

        if image is None:
            return

        # rospy.loginfo("@@@@@@ STEP: %d @@@@@@",self.step)

        if self.step==0:  # stop status
            self.traffic_light(image)
            return

        if self.step==1: # detect sign ~ end turn
            if self.check_turn(image):
                return
        
        elif self.step == 2: # straight find --> turn detect reset for obstacle !!~!~!~
            self.turn_detect='left'
            self.straight = self.is_straight(image,70)
            if self.straight:
                self.turn_detect = None
                self.step = 3
                self.straight_count = 0
                self.straight =None                

        elif self.step==3: # obstacle detect ~ end obstacle
            if self.check_obstacle_start():
                self.obstacle_move(image)
                return
            if self.check_obstacle_end():
                return

        elif self.step==4: # 장애물 후 straight 발견 --> 주차구간 시작 직전임
            self.straight = self.is_straight(image,80)
            if self.straight:
                self.step = 5
                self.straight_count = 0
                self.straight =None

        elif self.step==5:
            self.turn_end = 0
            self.turn_direction='parking'

            crop_img=image[300:,:]
            height,width=crop_img.shape[:2]
            hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
            yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
            yellow_mask[:height//2,:]=0
            yellow_mask[:,width//2:]=0
            self.yellow_area=cv2.countNonZero(yellow_mask)

            if self.yellow_area <=300 and self.set == None:
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left',0.8,0.03,0.045)
                self.set = 1

            if self.parking_end:
                self.stop()
                return

            if self.set ==1 and (0.15<=self.right_distance<=0.28): #오른ㅉ고에 뭐있음
                self.set = 2
            if self.set ==1 and 0.15<=self.left_distance<=0.28: #왼쪽에뭐있음
                self.set = 3

            if (self.set==2 and 0.3<=self.right_distance )or (self.set==3 and self.left_distance>=0.3):
                self.parking(image)
                return

        elif self.step ==6: # parking 끝 ~ 왼쪽회전까ㅣㅈ
            
            crop_img=image[300:,:]
            height,width=crop_img.shape[:2]
            hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
            yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
            yellow_mask[:height//2,:]=0
            yellow_mask[:,width//2:]=0
            self.yellow_area=cv2.countNonZero(yellow_mask)

            if self.yellow_area <=100 and self.front_distance <=0.25:
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left',0.8,0.03,0.45)
                self.step =7
                return
                
        elif self.step == 7: # 지그재그 전, 흰/노란선 다 보이기 시작 ~ gatebar
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([10, 255, 160]))        # mask=cv2.bitwise_or(red_mask)
            self.red_area=cv2.countNonZero(red_mask)   
            rospy.loginfo("Red is: %3f",self.red_area)
            self.turn_direction=None

            if self.gate_state == 0:
                if self.depth <= 0.25 and self.red_area>3000:
                    self.gate_close_count += 1
                    rospy.loginfo("GATE CLOSE COUNT: %d / 5",self.gate_close_count)
                else:
                    self.gate_close_count=0

                if self.gate_close_count >= 5:
                    self.gate_state=1
                    self.gate_close_count=0
                    self.gate_open_count=0
                    rospy.loginfo("@@@ GATE DETECTED -> STOP @@@")
                    self.stop()
                    return

            elif self.gate_state == 1:
                self.stop()

                if  self.red_area<=4000:
                    self.gate_open_count += 1
                    rospy.loginfo("GATE OPEN COUNT: %d / 10",self.gate_open_count)
                else:
                    self.gate_open_count=0

                if self.gate_open_count >= 10:
                    rospy.loginfo("@@@ GATE OPEN -> STEP 8 @@@")
                    self.gate_state=2
                    self.gate_open_count=0
                    self.step=8

                return            

            elif self.step ==8: pass


        self.lane_tracking(image)

    def publish_velocity(self,v_l,v_r): # 속도 퍼블리시 
        msg=Twist()

        msg.linear.x=(self.wheel_radius*(v_r+v_l)/2.0)*0.1
        msg.angular.z=(self.wheel_radius*(v_r-v_l)/self.wheel_separation)*0.2

        self.cmd_pub.publish(msg)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()

