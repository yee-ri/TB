#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image,PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from std_msgs.msg import UInt8
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.step_sub=rospy.Subscriber('/step',UInt8,self.step_callback)

        self.cmd_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw',Image,self.img_callback,queue_size=1,buff_size=2**24)
        self.scan_sub=rospy.Subscriber('/livox/lidar',PointCloud2,self.scan_callback)
        self.bridge=CvBridge()

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn1.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn1.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

        self.step=2
        # 0 :신호등 기달(정지상ㅌ)
        # 1 :초록불감지하고 주행시작
        # 2 :회전인식하고 회전함수까지 돌림끝
        # 3 :장애물 구간 시작전부터 끝
        # 4 :장애물 끝나고 다시 그대로가ㄱ기

        self.left_distance=float('inf')
        self.front_distance=float('inf')
        self.right_distance=float('inf')

        self.turn_detect=None #회전표지판 감지
        self.turn_direction=None # 회전명령

        self.prev_error=0.0
        self.prev_target_x=None
        self.prev_v_l=0.0
        self.prev_v_r=0.0

        self.wheel_radius=0.0475
        self.wheel_separation=0.19 #0.148이었던것




        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer)

        self.robot_half_width=0.08# 0.085
        self.safety_margin=0.03

        self.obstacle_direction=None
        self.obstacle_pass_count=0
        self.obstacle_clear_count=0
        self.obstacle_points=[]

    def step_callback(self,data):
        self.step=data.data
        self.prev_error=0.0
        self.prev_target_x=None
        self.prev_v_l=0.0
        self.prev_v_r=0.0

        if self.step<2:
            self.turn_detect=None
            self.turn_direction=None

        if self.step==3:
            self.obstacle_direction=None
            self.obstacle_clear_count=0

        rospy.loginfo("@@@@@@ STEP CHANGE -> %d @@@@@@",self.step)

    def traffic_light(self,data):
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

    def scan_callback(self,data):
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
        side_limit=0.25

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

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f DIR: %s",self.left_distance,self.front_distance,self.right_distance,str(self.obstacle_direction))
    
    def detect_sign(self,image):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        yellow_mask=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))
        white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
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

    def check_turn(self,image):
        sign=self.detect_sign(image)

        if self.turn_detect is None:
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

            self.step=2
            rospy.loginfo("@@@@@@ TURN END -> STEP 2 @@@@@@")
            return True

        return False

    def check_obstacle_start(self):
        if self.front_distance<=0.32:
            self.step=3
            self.turn_direction=None
            self.obstacle_direction=None
            self.obstacle_clear_count=0
            rospy.loginfo("@@@@@@ OBSTACLE START -> STEP 3 @@@@@@")
            return True

        return False

 #################################################################################

    def obstacle_move(self,image):
        height,width=image.shape[:2]
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,130]),np.array([179,80,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

        bottom=4*height//5
        left_yellow_b=yellow_mask[bottom:,:width//2]
        right_white_b=white_mask[bottom:,width//2:]

        yellow=yellow_mask[height//2:,:]
        white=white_mask[height//2:,width//2:]

        yellow_area_b=cv2.countNonZero(left_yellow_b)
        white_area_b=cv2.countNonZero(right_white_b)

        yellow_area=cv2.countNonZero(yellow)
        white_area=cv2.countNonZero(white)

        ##########################################
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
        # cv2.waitKey(1)
        ##########################################        

        rospy.loginfo("Y_b: %3f W_b: %3f",yellow_area_b,white_area_b)
        rospy.loginfo("Y: %3f W: %3f",yellow_area,white_area)

        line_area=2000
        safe_width=self.robot_half_width+self.safety_margin

        near_points=[]
        front_points=[]

        for x,y in self.obstacle_points:
            if -0.05<x<0.4 and abs(y)<safe_width+0.06: #side_width + 0.06
                near_points.append((x,y))

            if 0<x<0.35 and abs(y)<safe_width:
                front_points.append((x,y))

        rospy.loginfo("DIR:%s LEFT:%.2f FRONT:%.2f RIGHT:%.2f",str(self.obstacle_direction),self.left_distance,self.front_distance,self.right_distance)

        #################################################################################
        if white_area>8100:
            if 0<= white_area_b:
                rospy.loginfo("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                self.obstacle_clear_count +=1
            else :self.obstacle_clear_count = 0

        if self.obstacle_clear_count >=3:
            self.step = 4

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
                
            if white_area>6000:
                self.obstacle_direction='right'
                rospy.loginfo("slow right")
                self.move(0.01,-0.05)
#####################################################

        if self.obstacle_direction=='left': # 오른쪽 장애물 있어서 왼쪽으로 이동하자

            if self.front_distance<0.25 :
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

            elif white_area>7000 and self.front_distance>=0.25 : #self.front_distance>0.40 and # 오른쪽으로 가다가 흰선 넘 많이 보이면
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

    def lane_tracking(self,image):
        crop_img=image[300:,:]
        height,width=crop_img.shape[:2]

        hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

        if self.turn_direction=='left':
            yellow_mask[:,width//2:]=0

        elif self.turn_direction=='right':
            white_mask[:,:width//2]=0

        else: 
            white_mask[:,:width//4]=0

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

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
            self.publish_velocity(self.prev_v_l,self.prev_v_r)

        mask_view=cv2.vconcat([white_mask,yellow_mask])
        cv2.imshow('MASK VIEW',mask_view)
        cv2.waitKey(1)

    def turn_time(self,direction,duration,linear_speed,angular_speed):
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

    def move(self,linear,angular):
        msg=Twist()
        msg.linear.x=linear
        msg.angular.z=angular
        self.cmd_pub.publish(msg)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def publish_velocity(self,v_l,v_r):
        msg=Twist()

        msg.linear.x=(self.wheel_radius*(v_r+v_l)/2.0)*0.1
        msg.angular.z=(self.wheel_radius*(v_r-v_l)/self.wheel_separation)*0.2

        self.cmd_pub.publish(msg)

    def img_callback(self,data):
        image=self.bridge.imgmsg_to_cv2(data,'bgr8')

        if image is None:
            return

        rospy.loginfo("@@@@@@ STEP: %d @@@@@@",self.step)

        if self.step==0:
            self.traffic_light(image)
            return

        if self.step==1:
            if self.check_turn(image):
                return

        elif self.step==2:
            if self.check_obstacle_start():
                self.obstacle_move(image)
                return

        elif self.step==3:
            self.obstacle_move(image)
            return

        self.lane_tracking(image)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()
