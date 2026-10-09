#!/usr/bin/env python3
import rospy
import cv2
import math
import numpy as np
from pathlib import Path
from threading import RLock
from tf.transformations import euler_from_quaternion,quaternion_matrix
from sensor_msgs.msg import Image,PointCloud2,LaserScan
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
from tunnel.mission import TunnelMission, load_config

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)
        self.command_lock=RLock()
        self.shutting_down=False
        self.tunnel_generation=0
        self.tunnel_active=False
        config_path=rospy.get_param('~tunnel_config',str(Path(__file__).with_name('tunnel.yaml')))
        self.tunnel_config=load_config(config_path,rospy.get_param('~tunnel_profile','normal'))
        self.lidar_type=rospy.get_param('~lidar_type','pointcloud2')
        if self.lidar_type not in ('pointcloud2','laserscan'):
            raise ValueError('~lidar_type must be pointcloud2 or laserscan')
        self.lidar_topic=rospy.get_param('~lidar_topic',
                                       '/livox/lidar' if self.lidar_type=='pointcloud2' else '/scan_mid360_raw')
        self.tunnel=TunnelMission(self.tunnel_config,clock=lambda:rospy.Time.now().to_sec())
        self.cmd_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.bridge=CvBridge()
        image_dir=Path(__file__).resolve().parents[2]/'images'
        self.lturn_template=cv2.imread(str(image_dir/'lturn1.png'),cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))
        self.rturn_template=cv2.imread(str(image_dir/'rturn1.png'),cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

        

############## 구간별 테스트 설정 ##############
        # self.step=int(rospy.get_param('~start_step',0))
        self.step = 1
        self.yaw = 0.0
        self.initial_yaw=None

############## 라이다 관련 변수 ##############
        self.left_distance=float('inf')
        self.front_distance=float('inf')
        self.right_distance=float('inf')

############## 회전 관련 변수 ##############
        self.right_count=0
        self.left_count=0
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
        self.white_area = 0 
        self.obstacle_move_count=0
        self.obstacle_started=False

############## 직선 구간 관련 변수 ##############
        self.straight_count = 0
        self.straight_history=[]
        self.straight = None

############## 주차 관련 변수 ##############
        self.parking_ready = None
        self.parking_end = None
        self.yellow_area = 0
        self.set = None
        self.parking_count = 0
        self.signal = 0
        self.sequence = 0
        self.parking_side=None      
        self.yellow_zero_count=0

############## 게이트바 관련 변수 ##############
        self.gate_state = 0
        self.gate_open_count = 0
        self.gate_close_count=0
        self.red_area = 0
        self.depth = 0
        self.speed = 0

############## 터널 관련 변수 ##############        
        self.maze_count = 0
        self.odom_x=0.0
        self.odom_y=0.0
        self.odom_received=False
        self.odom_frame=None
        self.initial_x=None
        self.initial_y=None
        self.initial_yaw=None

        # ROS callbacks can run immediately: subscribe only after initialization.
        self.step_sub=rospy.Subscriber('/step',UInt8,self.step_callback)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw',Image,self.img_callback,queue_size=1,buff_size=2**24)
        scan_type,scan_callback=(PointCloud2,self.scan_callback) if self.lidar_type=='pointcloud2' else (LaserScan,self.laser_callback)
        self.scan_sub=rospy.Subscriber(self.lidar_topic,scan_type,scan_callback,queue_size=1)
        self.odom_sub=rospy.Subscriber('/odom',Odometry,self.odom_callback,queue_size=1)
        self.depth_sub=rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)
        self.tunnel_timer=rospy.Timer(rospy.Duration(self.tunnel_config['control']['period']),self.tunnel_tick)
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        with self.command_lock:
            if self.shutting_down:
                return
            self.shutting_down=True
            self.tunnel_generation+=1
            self.tunnel_timer.shutdown()
            self.cmd_pub.publish(Twist())
        self.tunnel.close()

    def step_callback(self,data): # 그냥 rostopic pub으로 구간별 테스트하려고 만든거
        with self.command_lock:
            was_tunnel=self.step in (9,10)
            self.step=data.data
            self.tunnel_generation+=1
            self.tunnel_active=False
            self.tunnel.close()
            self.prev_error=0.0
            self.prev_target_x=None
            self.prev_v_l=0.0
            self.prev_v_r=0.0
            if was_tunnel or self.step in (9,10):
                self.move(0.0,0.0,tunnel=self.step in (9,10))
            if self.step==3:
                self.obstacle_direction=None
                self.obstacle_clear_count=0

        rospy.loginfo("@@@@@@ STEP CHANGE -> %d @@@@@@",self.step)

    def odom_callback(self,data):
        x=data.pose.pose.position.x
        y=data.pose.pose.position.y
        q=data.pose.pose.orientation
        _,_,raw_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])

        if self.initial_yaw is None or self.initial_x is None or self.initial_y is None:
            self.initial_yaw=raw_yaw
            self.initial_x=x
            self.initial_y=y

        self.odom_x,self.odom_y,self.yaw=self.local_pose(x,y,raw_yaw)
        self.odom_frame=data.header.frame_id
        self.odom_received=True
        self.tunnel.update_odometry((self.odom_x,self.odom_y,self.yaw),
                                    data.twist.twist.linear.x,data.twist.twist.angular.z,
                                    data.header.stamp.to_sec())

    def local_pose(self,x,y,yaw):
        dx,dy=x-self.initial_x,y-self.initial_y
        c,s=math.cos(self.initial_yaw),math.sin(self.initial_yaw)
        angle=yaw-self.initial_yaw
        return c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(angle),math.cos(angle))

    def tunnel_tick(self,_event):
        with self.command_lock:
            if self.shutting_down or self.step not in (9,10):
                return
            if not self.tunnel_active:
                self.tunnel.reset()
                self.tunnel_active=True
            generation=self.tunnel_generation
        linear,angular,finished=self.tunnel.step(rospy.Time.now().to_sec())
        with self.command_lock:
            if generation!=self.tunnel_generation or self.step not in (9,10):
                return
            self.move(linear,angular,tunnel=True)
            if finished:
                self.step=11
                self.tunnel_active=False
                self.prev_error=0.0
                self.prev_target_x=None
                self.prev_v_l=self.prev_v_r=0.0
            else:
                self.step=9 if self.tunnel.state=='ENTRY' else 10
            rospy.loginfo_throttle(1.0,"TUNNEL: %s %s",self.tunnel.state,self.tunnel.reason)

    def lane_detect(self,image):
        crop=image[300:,:]
        if crop.size==0:
            return False
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        white=cv2.inRange(hsv,np.array([0,0,210]),np.array([179,55,255]))
        yellow=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
        minimum=self.tunnel_config['sensors']['minimum_lane_pixels']
        return cv2.countNonZero(white)>minimum and cv2.countNonZero(yellow)>minimum

    def lidar_transform(self,header):
        try:
            if self.step in (9,10):
                if header.stamp.to_sec()<=0.0:
                    return None
                return self.tf_buffer.lookup_transform(self.tunnel_config['sensors']['base_frame'],
                                                       header.frame_id,header.stamp,rospy.Duration(0.1))
            return self.tf_buffer.lookup_transform('base_footprint',header.frame_id,
                                                   rospy.Time(0),rospy.Duration(0.1))
        except Exception as e:
            rospy.logwarn("TF ERROR: %s",e)
            return None

    def scan_callback(self,data):
        transform=self.lidar_transform(data.header)
        if transform is None:
            return
        try:
            cloud=tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(data,transform)
        except Exception as e:
            rospy.logwarn("PointCloud transform failed: %s",e)
            return
        self.lidar_points(point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True),transform,data.header)

    def laser_callback(self,data):
        # Preserve real LaserScan no-return rays; missing cloud bins stay NaN.
        if not (math.isfinite(data.angle_min) and math.isfinite(data.angle_increment)
                and data.angle_increment!=0.0 and math.isfinite(data.range_min)
                and math.isfinite(data.range_max) and 0.0<=data.range_min<data.range_max):
            return
        ranges=np.asarray(data.ranges,dtype=float)
        valid=np.flatnonzero(np.isfinite(ranges))
        valid=valid[(ranges[valid]>=data.range_min)&(ranges[valid]<=data.range_max)]
        if not len(valid) and not np.any(np.isposinf(ranges)):
            return
        transform=self.lidar_transform(data.header)
        if transform is None:
            return
        if self.step in (9,10):
            poses=self.tunnel_lidar_pose(transform,data.header)
            if poses is not None:
                sensor_pose,scan_pose=poses
                self.tunnel.update_scan(ranges,data.angle_min,data.angle_increment,
                                        data.range_min,data.range_max,sensor_pose,scan_pose,
                                        data.header.stamp.to_sec())
            return
        if not len(valid):
            return
        angles=data.angle_min+valid*data.angle_increment
        r=ranges[valid]
        points=np.column_stack((r*np.cos(angles),r*np.sin(angles),np.zeros(len(r))))
        p,q=transform.transform.translation,transform.transform.rotation
        rotation=quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
        points=points.dot(rotation.T)+np.array([p.x,p.y,p.z])
        self.lidar_points(points,transform,data.header)

    def lidar_points(self,points,transform,header):
        settings=self.tunnel_config['sensors']

        self.obstacle_points=[]
        left_ranges=[]
        front_ranges=[]
        right_ranges=[]

        tunnel_mode=self.step in (9,10)
        tunnel_points=[]
        safe_width=self.robot_half_width+self.safety_margin
        side_limit=0.55

        for point in points:
            x,y,z=point

            if tunnel_mode:
                if (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
                        and settings['z_min']<z<settings['z_max']
                        and math.hypot(x,y)<=settings['range_max']):
                    tunnel_points.append((x,y))
                continue
            if -0.15<x<0.8 and abs(y)<side_limit:self.obstacle_points.append((x,y))
            if x<=0.15 or z>0.30:continue
            if abs(y)<safe_width:front_ranges.append(x)
            elif safe_width<=y<side_limit:left_ranges.append(math.hypot(x,y))
            elif -side_limit<y<=-safe_width:right_ranges.append(math.hypot(x,y))

        if tunnel_mode:
            poses=self.tunnel_lidar_pose(transform,header)
            if poses is not None:
                sensor_pose,scan_pose=poses
                self.tunnel.update_cloud(tunnel_points,sensor_pose,scan_pose,header.stamp.to_sec())
            return
        self.left_distance=np.percentile(left_ranges,10) if left_ranges else float('inf')
        self.front_distance=np.percentile(front_ranges,10) if front_ranges else float('inf')
        self.right_distance=np.percentile(right_ranges,10) if right_ranges else float('inf')

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f DIR: %s",self.left_distance,self.front_distance,self.right_distance,str(self.obstacle_direction))

    def tunnel_lidar_pose(self,transform,header):
        if not self.odom_received or not self.odom_frame:
            return None
        try:
            odom_tf=self.tf_buffer.lookup_transform(self.odom_frame,self.tunnel_config['sensors']['base_frame'],
                                                   header.stamp,rospy.Duration(0.1))
        except Exception as e:
            rospy.logwarn_throttle(1.0,"Tunnel acquisition TF unavailable: %s",e)
            return None
        p,q=odom_tf.transform.translation,odom_tf.transform.rotation
        yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        scan_pose=self.local_pose(p.x,p.y,yaw)
        p,q=transform.transform.translation,transform.transform.rotation
        sensor_pose=(p.x,p.y,euler_from_quaternion([q.x,q.y,q.z,q.w])[2])
        return sensor_pose,scan_pose

    def stop(self): # 초록불 감지 전 정지상태
        self.move(0.0,0.0)

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

        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179, 55, 255]))
        # white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        # blue_mask=cv2.inRange(hsv,np.array([90,80,50]),np.array([130,255,255]))

        mask=cv2.bitwise_or(yellow_mask,white_mask)
        # mask=cv2.bitwise_or(mask,blue_mask)

        filtered=cv2.bitwise_and(image,image,mask=mask)
        gray=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)

        left_result=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,max_val_l,_,_=cv2.minMaxLoc(left_result)

        right_result=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,max_val_r,_,_=cv2.minMaxLoc(right_result)

        rospy.loginfo("LEFT: %.1f%% RIGHT: %.1f%%",max_val_l*100,max_val_r*100)

        if 1.2<=self.front_distance<=2.2:
            if max_val_r>=max_val_l and max_val_r>0.29:
                self.right_count+=1
            elif max_val_l>=max_val_r and max_val_l>0.29:
                self.left_count+=1

            if self.right_count>=5:
                self.right_count=0
                self.left_count=0
                return 'right'

            if self.left_count>=4:
                self.right_count=0
                self.left_count=0
                return 'left'

        if self.step == 6 and (max_val_l >=0.25 or max_val_r>=0.25):
            self.left_count+=1

            if self.left_count>=4:
                self.right_count=0
                self.left_count=0
                return 'left'

        return None

    def check_turn(self,image): # self.turn_detect들어오면 회전명령 수행
        sign=self.detect_sign(image)

        if self.turn_detect is None:
            if sign=='left':
                self.turn_detect='left'
                rospy.loginfo("@@@@@@ LEFT SIGN DETECTED @@@@@@")

            elif sign=='right':
                self.turn_detect='right'
                rospy.loginfo("@@@@@@ RIGHT SIGN DETECTED @@@@@@")


        if self.turn_detect == 'right' and self.front_distance<0.8 and self.step<=2: #0.35:
            self.turn_direction=self.turn_detect
            self.turn_detect=None
            self.prev_error=0.0
            self.prev_target_x=None

        if self.turn_detect =='left' and self.front_distance<1.35 and self.step<=2: #0.35:
            self.turn_direction=self.turn_detect
            self.turn_detect=None
            self.prev_error=0.0
            self.prev_target_x=None

            if self.turn_direction=='left' and self.step <=2:
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left',1,0.03,0.35)
                

            elif self.turn_direction=='right' and self.step <=2:
                rospy.loginfo("@@@@@@ TURN RIGHT @@@@@@")
                self.turn_time('straight',0.7,0.07,0)
                self.turn_time('right',1,0.03,0.35)
                
            if self.step ==1:
                self.step=2

            rospy.loginfo("@@@@@@ TURN END  @@@@@@")
            self.turn_end = 1
            
            return True

        if self.step ==6 and self.turn_detect == 'left' and self.front_distance <=0.3:
            return 'left'



        return False

    def turn_time(self,direction,duration,linear_speed,angular_speed):
        msg=Twist()
        msg.linear.x=linear_speed

        if direction=='left':
            msg.angular.z=angular_speed
        elif direction=='right':
            msg.angular.z=-angular_speed
        elif direction=='straight':
            msg.angular.z=0

        start_time=rospy.Time.now()
        rate=rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed=(rospy.Time.now()-start_time).to_sec()

            if elapsed>=duration:
                break

            self.move(msg.linear.x,msg.angular.z)
            rate.sleep()

        self.stop()

    def check_obstacle_start(self):
        if self.front_distance<=0.32:
            self.obstacle_pass_count+=1
        else:
            self.obstacle_pass_count=0

        if self.obstacle_pass_count==4:
            self.turn_direction=None
            self.obstacle_direction=None
            self.obstacle_clear_count=0
            rospy.loginfo("@@@@@@ NEW OBSTACLE DETECTED @@@@@@")

        if self.obstacle_pass_count>=4:
            self.obstacle_pass_count=4
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
        rospy.loginfo("white is %.3f",white_area_b)
        rospy.loginfo("YEllow is %.3f",yellow_area_b)

        #########################################    
        # rospy.loginfo("Y: %3f W: %3f",yellow_area,self.white_area)

        line_area=2000
        safe_width=self.robot_half_width+self.safety_margin

        near_points=[]
        front_points=[]
        rospy.loginfo(self.obstacle_direction)

        for x,y in self.obstacle_points:
            if -0.05<x<0.4 and abs(y)<safe_width+0.06: #side_width + 0.06
                near_points.append((x,y))

            if 0<x<0.35 and abs(y)<safe_width:
                front_points.append((x,y))


        if self.obstacle_direction is None:
            if self.right_distance<0.25 and self.front_distance<=0.25 and self.left_distance>1.0 and white_area_b>4000:
                self.obstacle_direction='left'
                rospy.loginfo("RIGHT+FRONT BLOCKED -> TURN LEFT")
                self.move(0.015,0.35)
                # return False

            #대충반대
            elif self.left_distance<0.23 and self.front_distance<=0.23 and self.right_distance>1.0 and yellow_area_b>3000:
                self.obstacle_direction='right'
                rospy.loginfo("LEFT+FRONT BLOCKED -> TURN RIGHT")
                self.move(0.015,-0.35)
                # return False

            elif self.left_distance>self.right_distance:
                self.obstacle_direction = 'left'
                self.move(0.015,0.35)
                rospy.loginfo("RIGHT+FRONT BLOCKED -> TURN LEFT22")
            elif self.right_distance>self.left_distance:
                self.obstacle_direction='right'
                rospy.loginfo("LEFT+FRONT BLOCKED -> TURN RIGHT22")
                self.move(0.015,-0.35)

                #####################################################################
        if self.right_distance<0.25 and self.left_distance<=0.25 :
            if yellow_area>6000:
                self.obstacle_direction='left'
                rospy.loginfo("slow left")
                self.move(0.015,0.05)
                
            if self.white_area>6000:
                self.obstacle_direction='right'
                rospy.loginfo("slow right")
                self.move(0.015,-0.05)
            #####################################################

        if self.obstacle_direction=='left': # 오른쪽 장애물 있어서 왼쪽으로 이동하자

            if yellow_area_b>3500 and white_area_b<500:
                self.obstacle_direction='right'
                rospy.loginfo("YELLOW BOTTOM LARGE -> TURN RIGHT")
                self.move(0.015,-0.30)

            elif self.front_distance<0.22 :
                rospy.loginfo("front obstalce too close -> more fast turn left")
                self.move(0.015,0.3)

            elif front_points: #안전영역 안에 뭔가 계속 감지되면, 걍 계속 장애물 회피하는 겨
                rospy.loginfo("keep going turn left")
                self.move(0.02,0.2)

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
                self.move(0.025,-0.025)
            elif self.left_distance>1 and self.front_distance>1 and self.right_distance<0.2:

                self.obstacle_direction='right'
                rospy.loginfo("yellow and obstacle too close!!!! turn small right22")
                self.move(0.025,-0.01)

            # elif self.white_area>7000 and self.front_distance>=0.25 : #self.front_distance>0.40 and # 오른쪽으로 가다가 흰선 넘 많이 보이면
            elif white_area_b>3000 and self.front_distance>=0.25:
                self.obstacle_direction='left'
                rospy.loginfo("White too close!!!! turn left")
                self.move(0.025,0.1)


            elif front_points: #안전영역 안에 뭔가 계속 감지되면, 걍 계속 장애물 회피하는 겨
                rospy.loginfo("keep going turn right")
                self.move(0.02,-0.15)
                

            elif  self.right_distance<=0.18:
                rospy.loginfo("right obstacle too close!!!! turn left")
                self.obstacle_direction = 'left'
                self.move(0.02,0.015) 
            

            else:
                if self.front_distance<=0.4 and line_area<white_area_b<100000: # 대충 값 보고 바꿔야함 linearea랑 100000 둘다
                    rospy.loginfo("obstacle detected !!! turn left start !!!")
                    self.obstacle_direction='left'
                    self.move(0.018,0.20)
                else: # 일단 넣어봄,,
                    rospy.loginfo("IDK (right)")
                    self.move(0.06,-0.18)

    def move(self,linear,angular,tunnel=False): # 기존 노드가 유일한 명령 발행자
        msg=Twist()
        msg.linear.x=linear
        msg.angular.z=angular
        with self.command_lock:
            if self.shutting_down or (self.step in (9,10))!=tunnel:
                return
            self.cmd_pub.publish(msg)

    def check_obstacle_end(self): # 장애물 구간 끝났는지 확인
        if self.front_distance>0.9 and self.left_distance>0.5 and self.right_distance>0.5 :#and self.white_area>=200:
            self.obstacle_clear_count+=1
            rospy.loginfo("OBSTACLE CLEAR COUNT: %d",self.obstacle_clear_count)
        else:
            self.obstacle_clear_count=0

        if self.obstacle_clear_count>=40:
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
        if self.turn_direction=='left':
            yellow_mask[:,width//2:]=0
            yellow_mask[:1*height//2,:] = 0
            white_mask[:1*height//4,:] = 0

        elif self.turn_direction=='right':
            white_mask[:,:width//2]=0

        elif self.turn_direction=='parking':
            yellow_mask[:height//2,:]=0
            yellow_mask[:,width//2:]=0
            white_mask[:,:]=0

        elif self.turn_direction == 'straight':
            yellow_mask[:height//4,:] = 0
            white_mask[:3*height//4,:] = 0
            # white_mask[:,:width//2]=0
            white_mask[:,:width//2]=0

        elif self.turn_direction == 'left_yellow':
            yellow_mask[:,width//2:]=0

        else: 
            white_mask[:,:width//2]=0

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

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

            offset=360

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)

                if curve>170:
                    offset+=min((curve-170)*0.82,100)

            target_x=cx_w-offset

        elif M_y['m00']>0:
            cx_y=int(M_y['m10']/M_y['m00'])

            h=yellow_mask.shape[0]
            upper_mask=yellow_mask[:h//2,:]
            lower_mask=yellow_mask[h//2:,:]

            M_upper=cv2.moments(upper_mask)
            M_lower=cv2.moments(lower_mask)

            offset=330

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)

                if curve>160: # 작을수록 직선에 가깝고 클수록 곡선으로 봄
                    offset+=min((curve-160)*0.8,100)

            target_x=cx_y+offset

        if target_x is not None:
            if self.prev_target_x is None:
                self.prev_target_x=target_x
            else:
                if self.step==2:
                    target_x=0.8*target_x+0.2*self.prev_target_x
                else:
                    target_x=0.25*target_x+0.75*self.prev_target_x

                self.prev_target_x=target_x

            err_x=target_x-center_x
            diff_x=err_x-self.prev_error

            if self.step==2:
                Kp=0.25
                Kd=0.012
                linear=17.0
                max_angular=85.0

            elif self.speed==1:
                Kp=0.15
                Kd=0.008
                linear=13.0
                max_angular=45.0

                # Kp 증가: 차선 중심에서 벗어났을 때 더 강하게 회전함.
                # Kd 증가: 오차가 빠르게 변할 때 반응을 더 크게 만듦. 급격한 흔들림이 생길 수도 있음.
            else:
                # Kp=0.15
                # Kd=0.008
                # linear=15.0
                # max_angular=50.0
                Kp=0.2
                Kd=0.008
                linear=20.0
                max_angular=75.0

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

            # if 0.15<=self.right_distance<=0.3: # 오른쪽에 뭐있음 -->정지-> 왼쪽으로회전(0)--> 출구쪽 바라봄(1.57)
            #     self.signal=1
            #     self.parking_side='right'

            # elif 0.15<=self.left_distance<=0.3: # 왼쪽에 뭐있음 -->정지-> 오른쪽으로회전(0)--> 출구쪽 바라봄(1.57)
            elif self.set ==3:    
                self.signal=1
                self.parking_side='left'

            else:
                return

            rospy.loginfo("PARKING DETECTED: %s",self.parking_side)

            rospy.loginfo("@@@@@@@@@@@@@@@@@@@@@")

            # self.move_time(0.04,0,1)
            self.move(0,0)
            rospy.sleep(0.2)

        if self.parking_side=='right':
            self.move(0,0)
            rospy.loginfo("IM HERE")

            if self.set_yaw(3): # yaw 3 is 0.0
                #linear,angular,duration)
                self.move_time(0.08,0,1.4)
                rospy.sleep(0.04)
                self.move_time(-0.08,0,1.4)
                rospy.sleep(0.01)
                
                
                self.sequence=2

        elif self.parking_side=='left':
            self.move(0,0)
            rospy.loginfo("IM HERE22")

            if self.set_yaw(2): # yaw 2 is 3.14
                self.move_time(0.08,0,1.4)
                rospy.sleep(0.01)
                self.move_time(-0.08,0,1.4)
                rospy.sleep(0.01)
                
                self.sequence=2

        # if self.parking_end==None and self.sequence==2:
        #     if self.set_yaw(1):
        #         self.move_time(0.04,0,2.5)
        #         self.parking_end=1
        #         self.step=6
        #         return
        if self.parking_end==None and self.sequence==2:
            if self.set_yaw(1):
                self.move_time_yaw_hold(0.04,1.565,2.5)
                self.parking_end=1
                self.step=6
                return

    def set_yaw(self,num):
        if self.yaw is None:return

        if num==1:target_yaw=1.567
        elif num==2:target_yaw=-3.14
        elif num==3:target_yaw=0.0
        else:return

        rate=rospy.Rate(30)

        while not rospy.is_shutdown():
            err=target_yaw-self.yaw
            err=math.atan2(math.sin(err),math.cos(err))
            rospy.loginfo("SET YAW: current=%.3f target=%.3f err=%.3f",self.yaw,target_yaw,err)

            if abs(err)<=0.015:
                self.move(0,0)
                return 1

            if abs(err)>0.1:
                v_yaw=np.clip(err*1.2,-0.25,0.25)
            else:
                v_yaw=np.clip(err*0.5,-0.08,0.08)

            if 0<v_yaw<0.045:v_yaw=0.045
            elif -0.045<v_yaw<0:v_yaw=-0.045

            self.move(0,v_yaw)
            rate.sleep()

    def move_time_yaw_hold(self,linear,target_yaw,duration):
        start=rospy.Time.now()
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():
            if (rospy.Time.now()-start).to_sec()>=duration:break
            err=target_yaw-self.yaw
            err=math.atan2(math.sin(err),math.cos(err))
            angular=np.clip(err*1.5,-0.15,0.15)
            self.move(linear,angular)
            rate.sleep()
        self.move(0,0)

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

        # rospy.loginfo("CLOSEST FRONT DEPTH: %.3f m",self.depth)

    def is_straight(self,image,total_count):
        self.straight_count+=1
        rospy.loginfo("STRAIGHT COUNT: %d / %d",self.straight_count,total_count)

        if self.straight_count>=total_count:
            self.straight_count=0
            return True

        return False

    def img_callback(self,data): # 메인코드
        image=self.bridge.imgmsg_to_cv2(data,'bgr8')

        if image is None:
            return

        if self.step in (9,10):
            self.tunnel.update_lane(self.lane_detect(image),data.header.stamp.to_sec())
            return

        rospy.loginfo("@@@@@@ STEP: %d @@@@@@",self.step)

        if self.step==0:  # stop status
            self.traffic_light(image)
            return

        if self.step==1: # detect sign ~ end turn
            if self.check_turn(image):
                return
        
        elif self.step == 2: # straight find --> turn detect reset for obstacle !!~!~!~
            # self.turn_detect='left'
            self.straight = self.is_straight(image,250) #6
            if self.straight:
                self.turn_detect = 'straight'#None
                self.turn_direction = 'straight'
                self.step = 3
                # self.straight_count = 0
                self.straight =None                


        elif self.step==3:
            if self.check_obstacle_start():
                # self.turn_detect=None
                self.obstacle_move(image)
                self.obstacle_move_count+=1
                return

            if self.obstacle_move_count>=20:
                if self.check_obstacle_end():
                    self.obstacle_move_count=0
                    self.obstacle_pass_count=0
                    self.turn_detect=None
                    return


        elif self.step==4: # 장애물 후 straight 발견 --> 주차구간 시작 직전임
            self.straight = self.is_straight(image,100) #25
            if self.straight:
                self.step = 5
                # self.straight_count = 0
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
                # self.turn_time('left',5.2,0.03,0.085)
                self.turn_time('straight',1.3,0.06,0)
                self.turn_time('left',1.2,0.02,0.32) #duration,linear_speed,angular_speed)
                # self.turn_time('left',1,0.03,0.35)
                self.set = 1

            if self.parking_end:
                # self.turn_direction=None
                self.stop()
                return

            # if self.set ==1 and (0.15<=self.right_distance<=0.21): #오른ㅉ고에 뭐있음
            if self.set ==1 and (0.15<=self.right_distance<=0.31):
                self.set = 2
            # if self.set ==1 and 0.15<=self.left_distance<=0.21: #왼쪽에뭐있음
            if self.set ==1 and 0.15<=self.left_distance<=0.31:
                self.set = 3

            if (self.set==2 and 0.6<=self.right_distance )or (self.set==3 and self.left_distance>=0.6):
                self.parking(image)
                return

        elif self.step ==6: # parking 끝 ~ 왼쪽회전까ㅣㅈ
            
            crop_img=image[270:,:]
            height,width=crop_img.shape[:2]
            hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
            yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
            yellow_mask[:1*height//2,:]=0
            yellow_mask[:,1*width//3:]=0
            self.yellow_area=cv2.countNonZero(yellow_mask)

            rospy.loginfo("Yellow is: %d",self.yellow_area)
            left = self.check_turn(image)

            if self.yellow_area<=0:
                self.yellow_zero_count+=1
            else:
                self.yellow_zero_count=0

            if self.yellow_zero_count>=5:
                self.yellow_zero_count=0
                self.turn_time('straight',0.8,0.06,0)
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left',0.8,0.03,0.45)
                self.step=7
                return
                
        elif self.step == 7: # 지그재그 전, 흰/노란선 다 보이기 시작 ~ gatebar

            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            self.turn_direction = 'left_yellow'
            red_mask=cv2.inRange(hsv,np.array([0,180,107]),np.array([5,255,194]))
            self.red_area=cv2.countNonZero(red_mask)   
            rospy.loginfo("Red is: %3f",self.red_area)
            self.turn_direction=None

            if self.gate_state == 0:
                if 500<self.red_area<12000:
                    self.speed = 1
                    
                if (self.depth <= 0.75 and self.red_area>7000) or self.red_area >=12000:
                    self.gate_close_count += 1
                    rospy.loginfo("GATE CLOSE COUNT: %d / 5",self.gate_close_count)
                else:
                    # self.gate_close_count=0
                    pass

                if self.gate_close_count >= 5:
                    self.gate_state=1
                    self.gate_close_count=0
                    self.gate_open_count=0
                    rospy.loginfo("@@@ GATE DETECTED -> STOP @@@")
                    self.stop()
                    return

            elif self.gate_state == 1:
                self.stop()

                if  self.red_area<=2200:
                    self.gate_open_count += 1
                    rospy.loginfo("GATE OPEN COUNT: %d / 10",self.gate_open_count)
                else:
                    self.gate_open_count=0

                if self.gate_open_count >= 20:
                    rospy.loginfo("@@@ GATE OPEN -> STEP 8 @@@")
                    rospy.sleep(1.0)
                    self.gate_state=2
                    self.gate_open_count=0
                    self.turn_direction = None
                    self.step=8
                    self.speed = 0

                return  

        elif self.step==8:
            crop_img=image[300:,:]
            height,width=crop_img.shape[:2]
            hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)
            white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179,55,255]))
            yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
            yellow_area=cv2.countNonZero(yellow_mask)
            white_area=cv2.countNonZero(white_mask)

            rospy.loginfo("Y: %.3f W: %.3f",yellow_area,white_area)

            if yellow_area<=30 and white_area<=30:
                self.maze_count+=1
            else:
                self.maze_count=0

            if self.maze_count>=10:
                self.step=8.5
                return

        elif self.step==8.5:
            self.move_time_yaw_hold(0.06,-1.565,2.5)
            self.move(0,0)

            with self.command_lock:
                self.step=9
                self.tunnel_generation+=1
                self.tunnel_active=False
            return


        self.lane_tracking(image)

    def publish_velocity(self,v_l,v_r): # 속도 퍼블리시 
        msg=Twist()

        msg.linear.x=(self.wheel_radius*(v_r+v_l)/2.0)*0.1
        msg.angular.z=(self.wheel_radius*(v_r-v_l)/self.wheel_separation)*0.2

        with self.command_lock:
            if not self.shutting_down and self.step not in (9,10):
                self.cmd_pub.publish(msg)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()
