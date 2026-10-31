#!/usr/bin/env python3

import rospy
import cv2
import math
import os
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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tunnel.mission import TunnelMission, load_config
from tunnel.odometry import read_linear_scale, read_angular_command_scale, scale_pose
from tunnel.cloud import tunnel_cloud_xy
from tunnel.motor import arm_motor_torque

class _ColorLUTAddition: #LUT 관련 기능 모아 놓은 클래스
    def __init__(self, path):
        self.load_lut(path) #LUT 읽기

    #파일을 읽고 검사하기
    def load_lut(self, path):
        path = Path(path)
        if path.stat().st_size != 256 ** 3:
            raise ValueError('LUT must contain exactly 16777216 bytes: ' + str(path))
        table = np.fromfile(str(path), dtype=np.uint8)
        if table.size != 256 ** 3:
            raise ValueError('LUT read was incomplete')
        values, counts = np.unique(table, return_counts=True)
        if not set(values.tolist()).issubset({0, 1, 2, 4, 8, 16}):
            raise ValueError('Unexpected LUT class values')
        self.table = table
        self.counts = dict(zip(values.tolist(), counts.tolist()))

    #영상의 RGB를 색상 번호로 변환하기
    def rgb_to_label(self, image, bgr=False):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a uint8 H x W x 3 image')
        pixels = image.astype(np.uint32)
        if bgr:
            r, g, b = pixels[:, :, 2], pixels[:, :, 1], pixels[:, :, 0]
        else:
            r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        index = (r << 16) | (g << 8) | b
        return self.table[index]

    #색상 분류 영상 축소하기
    @staticmethod
    def downsample_label(label):
        if label.dtype != np.uint8 or label.ndim != 2:
            raise ValueError('Expected a uint8 H x W label image')
        h, w = label.shape
        if h % 2 or w % 2:
            raise ValueError('OR downsampling requires even width and height')
        return (label[0::2, 0::2] | label[0::2, 1::2]
                | label[1::2, 0::2] | label[1::2, 1::2])

    #분류 결과를 화면 표시용 색으로 바꾸기
    @staticmethod
    def label_to_rgb(label):
        # Same display palette as lua_label_to_rgb, including mixed-label black.
        if label.dtype != np.uint8 or label.ndim != 2:
            raise ValueError('Expected a uint8 H x W label image')
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = [26, 26, 26]
        palette[1] = [255, 255, 255]
        palette[2] = [255, 160, 0]
        palette[4] = [0, 0, 255]
        palette[8] = [255, 0, 0]
        palette[16] = [0, 255, 0]
        return palette[label]

    #원하는 색만 선택하기
    def mask(self, bgr, label):
        # LUT labels: White=1, Yellow=2, Blue=4, Red=8, Green=16.
        return (self.rgb_to_label(bgr, bgr=True) == label).astype(np.uint8) * 255


class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)
        # An absent rosout option makes rospy call the master on every log.
        # Set its unchanged default before callbacks can publish drive commands.
        if not rospy.has_param('/rosout_disable_topics_generation'):
            rospy.set_param('/rosout_disable_topics_generation',False)
        rospy.get_param_cached('/rosout_disable_topics_generation',False)
        self.command_lock=RLock()
        self.shutting_down=False
        self.enable_motor_torque=rospy.get_param('~enable_motor_torque',False)
        if type(self.enable_motor_torque) is not bool:
            raise ValueError('~enable_motor_torque must be a boolean')
        self.show_lane_debug=rospy.get_param('~show_lane_debug',True)
        if type(self.show_lane_debug) is not bool:
            raise ValueError('~show_lane_debug must be a boolean')
        self.drive_ready=not self.enable_motor_torque
        self.arming_cancelled=False
        self.tunnel_timer=None
        self.tunnel_generation=0
        self.tunnel_active=False
        config_path=rospy.get_param('~tunnel_config',os.environ.get(
            'TB_TUNNEL_CONFIG',str(Path(__file__).with_name('tunnel.real.yaml'))))
        self.tunnel_config=load_config(config_path,rospy.get_param('~tunnel_profile','normal'))
        self.tunnel_linear_scale=read_linear_scale(self.tunnel_config)
        self.tunnel_angular_command_scale=read_angular_command_scale(self.tunnel_config)
        rospy.loginfo("Tunnel linear scale: %.6f (distance/speed x scale, linear command / scale); angular command / %.6f",
                      self.tunnel_linear_scale,self.tunnel_angular_command_scale)
        self.lidar_type=rospy.get_param('~lidar_type','pointcloud2')
        if self.lidar_type not in ('pointcloud2','laserscan'):
            raise ValueError('~lidar_type must be pointcloud2 or laserscan')
        self.lidar_topic=rospy.get_param('~lidar_topic',
                                       '/livox/lidar' if self.lidar_type=='pointcloud2' else '/scan_mid360_raw')
        self.tunnel=TunnelMission(self.tunnel_config,clock=lambda:rospy.Time.now().to_sec())
        self.cmd_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.bridge=CvBridge()

        # LUT 초기화: 템플릿도 LUT로 처리하므로 템플릿 로딩 전에 실행합니다.
        self._show_lut_debug = rospy.get_param('~show_lut_debug', False)
        if type(self._show_lut_debug) is not bool:
            raise ValueError('~show_lut_debug must be a boolean')
        lut_path = Path(rospy.get_param(
            '~color_lut_path', str(Path(__file__).resolve().with_name('current.lut'))
        )).expanduser()
        if not lut_path.is_file():
            raise FileNotFoundError('LUT color recognition requires: ' + str(lut_path))
        self._color_lut = _ColorLUTAddition(lut_path)
        for label, name in ((1, 'White'), (2, 'Yellow'), (4, 'Blue')):
            if not self._color_lut.counts.get(label, 0):
                raise ValueError('LUT requires registered ' + name + ' colors')
        rospy.loginfo('LUT color recognition: File=%s counts=%s',
                      lut_path, self._color_lut.counts)
        image_dir=Path(__file__).resolve().parents[2]/'images'
        self.lturn_template = self.load_sign_template(str(image_dir/'left.png'))
        self.rturn_template = self.load_sign_template(str(image_dir/'right.png'))
        self.start_time = None
        

############## 구간별 테스트 설정 ##############
        self.step=int(rospy.get_param('~start_step',0))
        self.standalone_gate_start=self.step==8
        # Full-course odometry uses its original heading. A fresh step8 run
        # instead captures the approach heading when the lane ends.
        self.gate_approach_yaw=-1.565
        self.yaw = 0.0
        self.initial_yaw=None

############## 라이다 관련 변수 ##############
        self.left_distance=float('inf')
        self.front_distance=float('inf')
        self.right_distance=float('inf')
        self.lidar_update_count = 0
        self.obstacle_last_lidar_count = 0
############## 회전 관련 변수 ##############
        self.right_count=0
        self.left_count=0
        self.turn_detect=None #회전표지판 감지
        self.turn_direction=None # 회전명령 @@@@@@
        self.max_val_l=0
        self.max_val_r=0
        self.sign_threshold = 0.70  
        self.sign_left_count = 0
        self.sign_right_count = 0

############## 라인 트랙킹 관련 변수 ##############
        self.prev_error=0.0
        self.prev_target_x=None
        self.prev_v_l=0.0
        self.prev_v_r=0.0
        self.wheel_radius=0.04775
        self.wheel_separation=0.148 #0.148이었던것

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

############## 직선 구간 관련 변수 ##############
        self.straight_count = 0
        self.straight = None

############## 주차 관련 변수 ##############
        self.parking_end = None
        self.yellow_area = 0
        self.set = None
        self.signal = 0
        self.sequence = 0
        self.parking_side=None      
        self.yellow_zero_count=0


        self.obstacle_stage = None
        self.obstacle_stage_count = 0
        self.obstacle_turn_start = None

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
        scan_options={'queue_size':1}
        if self.lidar_type=='pointcloud2':
            # A Mid-360 cloud is ~522 kB. Read multiple complete messages so
            # queue_size=1 can discard old clouds after a processing stall.
            scan_options['buff_size']=4*1024*1024
        self.scan_sub=rospy.Subscriber(self.lidar_topic,scan_type,scan_callback,**scan_options)
        self.odom_sub=rospy.Subscriber('/odom',Odometry,self.odom_callback,queue_size=1)
        self.depth_sub=rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)
        rospy.on_shutdown(self.shutdown)
        if self.enable_motor_torque:
            if self.step not in (0,8,9,10):
                raise ValueError('Physical torque startup requires start step 0, 8, 9 or 10')
            with arm_motor_torque(self.cmd_pub,Twist,
                                  lambda:self.shutting_down or self.arming_cancelled,
                                  self.shutdown,self.command_lock):
                try:
                    with self.command_lock:
                        if self.shutting_down or self.arming_cancelled:
                            raise RuntimeError('Motor startup cancelled')
                        self.drive_ready=True
                        self.tunnel_timer=rospy.Timer(rospy.Duration(self.tunnel_config['control']['period']),self.tunnel_tick)
                except BaseException:
                    self.shutdown()
                    raise
        else:
            self.tunnel_timer=rospy.Timer(rospy.Duration(self.tunnel_config['control']['period']),self.tunnel_tick)

    def shutdown(self):
        with self.command_lock:
            if self.shutting_down:
                return
            self.shutting_down=True
            self.drive_ready=False
            self.tunnel_generation+=1
            if self.tunnel_timer is not None:
                self.tunnel_timer.shutdown()
            self.cmd_pub.publish(Twist())
        self.tunnel.close()

    def step_callback(self,data): # 그냥 rostopic pub으로 구간별 테스트하려고 만든거
        with self.command_lock:
            if not self.drive_ready:
                self.arming_cancelled=True
            was_tunnel=self.step in (9,10)
            self.step=data.data
            self.tunnel_generation+=1
            self.tunnel_active=False
            self.tunnel.close()
            self.prev_error=0.0
            self.prev_target_x=None
            self.prev_v_l=0.0
            self.prev_v_r=0.0
            self.speed=0
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
        tunnel_pose=scale_pose((self.odom_x,self.odom_y,self.yaw),self.tunnel_linear_scale)
        self.tunnel.update_odometry(tunnel_pose,
                                    data.twist.twist.linear.x*self.tunnel_linear_scale,data.twist.twist.angular.z,
                                    data.header.stamp.to_sec())

    def local_pose(self,x,y,yaw):
        dx,dy=x-self.initial_x,y-self.initial_y
        c,s=math.cos(self.initial_yaw),math.sin(self.initial_yaw)
        angle=yaw-self.initial_yaw
        return c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(angle),math.cos(angle))

    def tunnel_tick(self,_event):
        with self.command_lock:
            if self.shutting_down or not self.drive_ready or self.step not in (9,10):
                return
            if not self.tunnel_active:
                self.tunnel.reset()
                self.tunnel_active=True
            generation=self.tunnel_generation
        linear,angular,finished=self.tunnel.step(rospy.Time.now().to_sec())
        with self.command_lock:
            if generation!=self.tunnel_generation or self.step not in (9,10):
                return
            self.move(linear/self.tunnel_linear_scale,
                      angular/self.tunnel_angular_command_scale,tunnel=True)
            if finished:
                self.step=11
                self.tunnel_active=False
                self.prev_error=0.0
                self.prev_target_x=None
                self.prev_v_l=self.prev_v_r=0.0
            else:
                self.step=9 if self.tunnel.state in ('LOCALIZING','ENTRY') else 10
            status=(self.tunnel.state,self.tunnel.reason)
            if status!=getattr(self,'tunnel_logged_status',None):
                rospy.loginfo("TUNNEL: %s %s",*status)
                self.tunnel_logged_status=status
            else:
                rospy.loginfo_throttle(1.0,"TUNNEL: %s %s",*status)

    def lane_detect(self,image):
        crop=image[300:,:]
        if crop.size==0:
            return False
        white = self._color_lut.mask(crop, 1)
        yellow = self._color_lut.mask(crop, 2)
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
            if self.step in (9,10):
                p,q=transform.transform.translation,transform.transform.rotation
                # Use the same KDL rotation coefficients as do_transform_cloud,
                # including its treatment of non-normalized quaternions.
                kdl_rotation=tf2_sensor_msgs.tf2_sensor_msgs.transform_to_kdl(transform).M
                rotation=np.array([[kdl_rotation[i,j] for j in range(3)] for i in range(3)])
                wall_band=(self.tunnel_config.get('localization',{}).get('height_band')
                           if self.tunnel.anchor is None else None)
                points=tunnel_cloud_xy(data,rotation,np.array([p.x,p.y,p.z]),
                                       self.tunnel_config['sensors'],point_cloud2.read_points,
                                       motion_poses=self.tunnel_motion_poses,wall_height_band=wall_band)
                wall_points=None
                if wall_band is not None and points is not None:
                    points,wall_points=points
                if points is not None and self.step in (9,10):
                    self.obstacle_points=[]
                    pose_timeout=0. if self.tunnel_config['sensors'].get('livox_motion_compensation',False) else .1
                    poses=self.tunnel_lidar_pose(transform,data.header,timeout=pose_timeout)
                    if poses is not None:
                        sensor_pose,scan_pose=poses
                        self.tunnel.update_cloud(points,sensor_pose,scan_pose,data.header.stamp.to_sec(),
                                                 localization_points=wall_points)
                    return
            cloud=tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(data,transform)
        except Exception as e:
            rospy.logwarn("PointCloud transform failed: %s",e)
            return
        self.lidar_points(point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True),transform,data.header)

    def tunnel_motion_poses(self,stamps):
        if not self.odom_received or not self.odom_frame:
            raise ValueError('Tunnel motion compensation needs odometry')
        odom_frame=self.odom_frame
        base_frame=self.tunnel_config['sensors']['base_frame']
        poses=np.empty((len(stamps),3),dtype=float)

        order=[len(stamps)-1]+list(range(len(stamps)-1))
        for index in order:
            timeout=.1 if index==len(stamps)-1 else 0.
            odom_tf=self.tf_buffer.lookup_transform(odom_frame,base_frame,
                         rospy.Time.from_sec(float(stamps[index])),rospy.Duration(timeout))
            p,q=odom_tf.transform.translation,odom_tf.transform.rotation
            yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
            poses[index]=scale_pose(self.local_pose(p.x,p.y,yaw),self.tunnel_linear_scale)
        if self.odom_frame!=odom_frame:
            raise ValueError('Odometry frame changed during cloud acquisition')
        return poses

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
        # self.left_distance=np.percentile(left_ranges,10) if left_ranges else float('inf')
        # self.front_distance=np.percentile(front_ranges,10) if front_ranges else float('inf')
        # self.right_distance=np.percentile(right_ranges,10) if right_ranges else float('inf')
        self.left_distance = np.percentile(left_ranges, 10) if left_ranges else float('inf')
        self.front_distance = min(front_ranges) if front_ranges else float('inf')
        self.right_distance = np.percentile(right_ranges, 10) if right_ranges else float('inf')
        self.lidar_update_count += 1

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f DIR: %s",self.left_distance,self.front_distance,self.right_distance,str(self.obstacle_direction))

    def tunnel_lidar_pose(self,transform,header,timeout=.1):
        if not self.odom_received or not self.odom_frame:
            return None
        try:
            odom_tf=self.tf_buffer.lookup_transform(self.odom_frame,self.tunnel_config['sensors']['base_frame'],
                                                   header.stamp,rospy.Duration(timeout))
        except Exception as e:
            rospy.logwarn_throttle(1.0,"Tunnel acquisition TF unavailable: %s",e)
            return None
        p,q=odom_tf.transform.translation,odom_tf.transform.rotation
        yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        scan_pose=scale_pose(self.local_pose(p.x,p.y,yaw),self.tunnel_linear_scale)
        p,q=transform.transform.translation,transform.transform.rotation
        sensor_pose=(p.x,p.y,euler_from_quaternion([q.x,q.y,q.z,q.w])[2])
        return sensor_pose,scan_pose

    def stop(self): # 초록불 감지 전 정지상태
        self.move(0.0,0.0)

    def traffic_light(self,data): # 초록불 감지 함수

        # green_mask = self._color_lut.mask(data, 16)
        # green_mask=cv2.erode(green_mask,None,iterations=1)
        # green_mask=cv2.dilate(green_mask,None,iterations=2)
        hsv_frame = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_frame, np.array([45, 100, 80]), np.array([85, 255, 255]))
        if cv2.countNonZero(green_mask)>=200:
            rospy.loginfo("@@@@@@ GREEN LIGHT @@@@@@")
            # self.stop()
            rospy.sleep(0.5)
            self.start_time = rospy.Time.now().to_sec()
            self.prev_error = 0.0
            self.prev_target_x = None
            self.step = 1
            # self.step=1
            rospy.loginfo("@@@@@@ STEP 1 @@@@@@")
            return True

        self.stop()
        return False

    def sign_blue_mask(self, image):
        mask = self._color_lut.mask(image, 4)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def sign_arrow_mask(self, image):
        image = cv2.resize(image, (100, 100))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array([0, 0, 80]),np.array([179, 100, 255]))

        inner = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(inner, (50, 50), 40, 255, -1)
        return cv2.bitwise_and(mask, inner)

    def load_sign_template(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError('Cannot read sign template: ' + str(path))
        contours, _ = cv2.findContours(self.sign_blue_mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('No LUT Blue region in sign template: ' + str(path))
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        template = self.sign_arrow_mask(image[y:y+h, x:x+w])
        if cv2.countNonZero(template) in (0, template.size):
            raise ValueError('No usable white arrow in sign template: ' + str(path))
        return template

    def detect_sign(self, image):
        contours, _ = cv2.findContours(self.sign_blue_mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.max_val_l = 0.0
        self.max_val_r = 0.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 400 or not 0.65 <= w / float(h) <= 1.5:
                continue
            if cv2.contourArea(contour) / float(w * h) < 0.45:
                continue

            arrow = self.sign_arrow_mask(image[y:y+h, x:x+w])
            if not 0.03 <= cv2.countNonZero(arrow) / 10000.0 <= 0.40:
                continue

            left = float(cv2.matchTemplate(arrow, self.lturn_template, cv2.TM_CCOEFF_NORMED)[0, 0])
            right = float(cv2.matchTemplate(arrow, self.rturn_template, cv2.TM_CCOEFF_NORMED)[0, 0])
            if not np.isfinite(left) or not np.isfinite(right):
                continue

            if max(left, right) > max(self.max_val_l, self.max_val_r):
                self.max_val_l, self.max_val_r = left, right

        rospy.loginfo("LEFT: %.2f RIGHT: %.2f", self.max_val_l, self.max_val_r)
        detected = max(self.max_val_l, self.max_val_r) >= self.sign_threshold


        near = 0.2 <= self.front_distance <= 0.75

        if detected and self.max_val_l > self.max_val_r:
            self.sign_left_count += 1
            self.sign_right_count = 0
            self.left_count = self.left_count + 1 if near else 0
            self.right_count = 0

        elif detected and self.max_val_r > self.max_val_l:
            self.sign_right_count += 1
            self.sign_left_count = 0
            self.right_count = self.right_count + 1 if near else 0
            self.left_count = 0

        else:
            self.left_count = self.right_count = 0
            self.sign_left_count = self.sign_right_count = 0

        direction = None
        if self.left_count >= 3 or self.sign_left_count >= 3:
            direction = 'left'
        elif self.right_count >= 3 or self.sign_right_count >= 3:
            direction = 'right'

        if direction is not None:
            self.left_count = self.right_count = 0
            self.sign_left_count = self.sign_right_count = 0

        return direction
   
    def check_turn(self, image):
        sign = self.detect_sign(image)

        if self.turn_detect is None and sign in ('left', 'right'):
            self.turn_detect = sign
            rospy.loginfo("@@@@@@ %s SIGN SAVED @@@@@@", sign.upper())

        if self.step == 1 and self.turn_detect in ('left', 'right') and math.isfinite(self.front_distance) and 0 < self.front_distance <= 0.65:
            self.turn_direction = self.turn_detect
            self.turn_detect = None
            self.prev_error = 0.0
            self.prev_target_x = None

            if self.turn_direction == 'left':
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left', 1, 0.03, 0.35)
            else:
                rospy.loginfo("@@@@@@ TURN RIGHT @@@@@@")
                self.turn_time('straight', 0.7, 0.07, 0)
                self.turn_time('right', 1, 0.03, 0.35)

            self.step = 2
            rospy.loginfo("@@@@@@ TURN END @@@@@@")
            return True

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
        if self.obstacle_last_lidar_count == self.lidar_update_count:
            return False

        self.obstacle_last_lidar_count = self.lidar_update_count

        if math.isfinite(self.front_distance) and self.front_distance <= 0.55:
            self.obstacle_pass_count += 1
        else:
            self.obstacle_pass_count = 0

        if self.obstacle_pass_count >= 7:
            self.obstacle_pass_count = 0
            self.turn_direction = None
            self.obstacle_direction = None
            self.obstacle_clear_count = 0
            rospy.loginfo("@@@@@@ NEW OBSTACLE DETECTED @@@@@@")
            return True

        return False

    def obstacle_move(self, image):
        height, width = image.shape[:2]
        yellow_mask = self._color_lut.mask(image, 2)
        yellow_area = cv2.countNonZero(yellow_mask[height // 2:, :width // 2])

        yellow_threshold = 200
        clear_distance = 0.50
        turn_speed = 0.03
        left_angular = 0.30
        right_angular = -0.30
        turn_timeout = 10.0
        now = rospy.Time.now().to_sec()

        rospy.loginfo("OBSTACLE STAGE: %s / FRONT: %.2f", self.obstacle_stage, self.front_distance)

        # 회전 제한시간 초과 시 정지
        if self.obstacle_stage in ('left', 'right'):
            if now - self.obstacle_turn_start >= turn_timeout:
                self.move(0, 0)
                rospy.logwarn_throttle(1.0, "OBSTACLE TURN TIMEOUT")
                return

        # 1. 첫 번째 장애물: 왼쪽으로 회피
        if self.obstacle_stage == 'left':
            self.speed = 2
            self.obstacle_lane_started = False
            self.obstacle_direction = 'left'

            rospy.loginfo_throttle(0.2, "LEFT TURN: FRONT=%.2f YELLOW=%d", self.front_distance, yellow_area)

            if yellow_area >= yellow_threshold and math.isfinite(self.front_distance) and self.front_distance > clear_distance:
                self.move(0, 0)
                self.obstacle_stage = 'lane'
                self.obstacle_stage_count = 0
                self.obstacle_direction = None
                self.turn_direction = 'straight'
                self.prev_error = 0.0
                self.prev_target_x = None
                rospy.loginfo("FIRST OBSTACLE CLEAR -> LANE TRACKING")
                return

            self.move(turn_speed, left_angular)
            return
        # 2. 차선 추종 중 정면 0.2m 이내면 오른쪽 회피
        if self.obstacle_stage == 'lane':
            if math.isfinite(self.front_distance) and self.front_distance <= 0.28:
                self.obstacle_stage = 'right'
                self.obstacle_stage_count = 0
                self.obstacle_turn_start = now
                self.obstacle_direction = 'right'
                self.obstacle_lane_started = False
                self.move(turn_speed, right_angular)
                rospy.loginfo("FRONT <= 0.2 -> TURN RIGHT")
                return

            self.turn_direction = 'straight'
            self.lane_tracking(image)
            return

        # 3. 오른쪽 회피 중 정면 1.5m 이상이면 회전 종료
        if self.obstacle_stage == 'right':
            if math.isfinite(self.front_distance) and self.front_distance >= 1.0:
                self.move(0, 0)
                self.obstacle_stage = 'done'
                self.obstacle_stage_count = 0
                self.obstacle_clear_count = 0

                self.prev_target_x = None
                self.prev_error = 0.0
                self.obstacle_direction = None
                self.turn_direction = 'obstacle'
                self.obstacle_lane_started = True
                rospy.loginfo("FRONT >= 1.5 -> LANE TRACKING")
                return

            self.obstacle_direction = 'right'
            self.move(turn_speed, right_angular)
            return

        # 4 마지막 장애물 회피하고 흰선 따라가가ㅣ

        if self.obstacle_stage == 'done':
            self.obstacle_direction = None

            # 다시 정면 장애물을 만나면 차선 추종보다 회피를 우선
            if math.isfinite(self.front_distance) and self.front_distance <= 0.2:
                self.obstacle_stage = 'right'
                self.obstacle_stage_count = 0
                self.obstacle_turn_start = now
                self.obstacle_clear_count = 0
                self.obstacle_direction = 'right'
                self.obstacle_lane_started = False
                self.move(turn_speed, right_angular)
                rospy.loginfo("FRONT <= 0.2 -> TURN RIGHT AGAIN")
                return

            if math.isfinite(self.front_distance) and self.front_distance >= 1.15:
                self.obstacle_lane_started = True

            if getattr(self, 'obstacle_lane_started', False):
                self.turn_direction = 'obstacle'
                self.lane_tracking(image)
                return

        
        self.move(0, 0)

    def move(self,linear,angular,tunnel=False): # 기존 노드가 유일한 명령 발행자
        msg=Twist()
        msg.linear.x=linear
        msg.angular.z=angular
        with self.command_lock:
            if self.shutting_down or (self.step in (9,10))!=tunnel:
                return
            if not self.drive_ready:
                msg=Twist()
            self.cmd_pub.publish(msg)

    def check_obstacle_end(self):
        if self.obstacle_stage != 'done': return False
        if self.obstacle_last_lidar_count == self.lidar_update_count: return False
        self.obstacle_last_lidar_count = self.lidar_update_count

        if math.isfinite(self.front_distance) and self.front_distance >= 1.0:
            self.obstacle_clear_count += 1
        else:
            self.obstacle_clear_count = 0

        rospy.loginfo("OBSTACLE CLEAR COUNT: %d / 8", self.obstacle_clear_count)

        if self.obstacle_clear_count >= 5:
            self.step = 4
            self.obstacle_clear_count = 0
            self.obstacle_direction = None
            rospy.loginfo("@@@@@@ OBSTACLE END -> STEP 4 @@@@@@")
            return True
        return False

    def detect_curve(self, white_mask, yellow_mask, threshold=95):
        minimum_pixels = 15
        single_threshold = 50
        curve_values = []
        visible_count = 0

        for mask in (white_mask, yellow_mask):
            h = mask.shape[0]
            upper = mask[:h//2, :]
            lower = mask[h//2:, :]
            total_pixels = cv2.countNonZero(mask)
            upper_pixels = cv2.countNonZero(upper)
            lower_pixels = cv2.countNonZero(lower)

            if total_pixels < minimum_pixels:
                continue

            visible_count += 1
            reasons = []
            if upper_pixels < minimum_pixels: reasons.append("위쪽 부족")
            if lower_pixels < minimum_pixels: reasons.append("아래쪽 부족")

            if reasons:
                continue

            mu = cv2.moments(upper)
            ml = cv2.moments(lower)
            cx_upper = mu['m10'] / mu['m00']
            cx_lower = ml['m10'] / ml['m00']
            value = abs(cx_upper - cx_lower)
            curve_values.append(value)

        if not curve_values:
            return True

        limit = single_threshold if visible_count == 1 else threshold
        curve = max(curve_values)
        is_curve = curve > limit
        return is_curve

    def lane_tracking(self,image): #기본적인 라인 트래킹 수행
        crop_img=image[300:,:]
        height,width=crop_img.shape[:2]


        white_mask = self._color_lut.mask(crop_img, 1)
        yellow_mask = self._color_lut.mask(crop_img, 2)
        if self.step==3:
            is_curve=self.detect_curve(white_mask,yellow_mask)

        if self.turn_direction=='left':
            yellow_mask[:,width//2:]=0
            yellow_mask[:1*height//3,:] = 0
            white_mask[:1*height//4,:] = 0

        elif self.turn_direction=='right':
            white_mask[:,:width//2]=0

        elif self.turn_direction=='parking': 
            yellow_mask[:1*height//2,:]=0
            yellow_mask[:,width//2:]=0
            white_mask[:,:]=0

        elif self.turn_direction=='parking2': 
            yellow_mask[:2*height//3,:]=0
            yellow_mask[:,width//2:]=0
            white_mask[:,:]=0
            rospy.loginfo("INHERE")

        elif self.turn_direction == 'straight':
            yellow_mask[:height//4,:] = 0
            white_mask[:3*height//4,:] = 0
            # white_mask[:,:width//2]=0
            white_mask[:,:width//2]=0
        elif self.turn_direction == 'obstacle':
            # yellow_mask[:height//2, :] = 0
            # white_mask[:height//2, :] = 0
            yellow_mask=yellow_mask
            white_mask=white_mask

        elif self.turn_direction=='left_yellow':
            yellow_mask[:,width//2:]=0

        else:
            white_mask[:,:width//2]=0

        if self.step != 3:
            is_curve = self.detect_curve(white_mask, yellow_mask)


        M_w = cv2.moments(white_mask)
        M_y = cv2.moments(yellow_mask)
        if self.step == 3: rospy.loginfo("FINAL MASK: W=%d Y=%d", cv2.countNonZero(white_mask), cv2.countNonZero(yellow_mask))
        center_x=width/2.0
        target_x=None

        if M_w['m00']>0 and M_y['m00']>0:
            cx_w=int(M_w['m10']/M_w['m00'])
            cx_y=int(M_y['m10']/M_y['m00'])
            target_x=(cx_w+cx_y)/2.0 

        elif M_w['m00']>0:
            cx_w=int(M_w['m10']/M_w['m00'])

            h=white_mask.shape[0]
            upper_mask=white_mask[:h//2,:]
            lower_mask=white_mask[h//2:,:]

            M_upper=cv2.moments(upper_mask)
            M_lower=cv2.moments(lower_mask)
            if self.step ==3 and self.speed ==2:
                offset = 400
                print("HEHEHEHEH")
            else:
                offset=350

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)
                if (self.step == 3 and self.speed ==2):
                    if curve>110: # 작을수록 직선에 가깝고 클수록 곡선으로 봄 #160
                        offset+=min((curve-110)*0.90,100) #0.8
                elif  self.step ==6 or self.step==5:
                    if curve>170: # 작을수록 직선에 가깝고 클수록 곡선으로 봄 #160
                        offset+=min((curve-170)*0.80,100) #0.8
                else:
                    if curve>170: #170
                        offset+=min((curve-170)*0.8,100) #0.82 -->3.2

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
                if (self.step == 3 and self.speed ==2) or self.step ==6:
                    if curve>160: # 작을수록 직선에 가깝고 클수록 곡선으로 봄 #160
                        offset+=min((curve-160)*0.82,100) #0.8
                else:                    
                    if curve>170: # 작을수록 직선에 가깝고 클수록 곡선으로 봄 #160
                        offset+=min((curve-170)*0.8,100) #0.8

            target_x=cx_y+offset

        # 마지막 회피 후: 가까운 흰선을 기준으로 왼쪽 목표 설정
        if self.step == 3 and self.obstacle_stage == 'done':
            h = white_mask.shape[0]
            near_white = white_mask[3*h//4:, :]
            near_m = cv2.moments(near_white)

            if near_m['m00'] > 0:
                near_cx = near_m['m10'] / near_m['m00']
                target_x = min( near_cx - 400,center_x - 60)

        if target_x is not None: 
            if self.prev_target_x is None:
                self.prev_target_x = target_x
            else:
                if self.step == 3 and self.obstacle_stage == 'done':
                    target_x = 0.8 * target_x + 0.2 * self.prev_target_x
                elif self.step == 2:
                    target_x = 0.8 * target_x + 0.2 * self.prev_target_x
                else:
                    target_x = 0.25 * target_x + 0.75 * self.prev_target_x

                self.prev_target_x = target_x

            err_x=target_x-center_x

            if self.step == 3: rospy.loginfo("LANE: W=%d Y=%d TARGET=%.1f CENTER=%.1f ERR=%.1f", cv2.countNonZero(white_mask), cv2.countNonZero(yellow_mask), target_x, center_x, err_x)
            diff_x=err_x-self.prev_error

            if self.speed==1: #gate bar 포착되면 속도 늦추는 거임
                Kp=0.15
                Kd=0.008
                linear=7.0
                max_angular=45.0


            elif (self.speed ==2 and self.step ==3)  : # 장애물 구간에서 피하는 속도
                Kp=0.18
                Kd=0.02
                linear=17.0
                max_angular=60.0
                rospy.loginfo("speed 2 active")
                # Kp 증가: 차선 중심에서 벗어났을 때 더 강하게 회전함.
                # Kd 증가: 오차가 빠르게 변할 때 반응을 더 크게 만듦. 급격한 흔들림이 생길 수도 있음.
            elif self.step ==6:
                Kp=0.15
                Kd=0.015
                linear=18.0
                max_angular=30.0
            elif self.step ==5 :
                Kp=0.11
                Kd=0.25
                linear=18.0
                max_angular=50.0
            else:
                if is_curve and self.step not in (2,4,5,7):
                    Kp=0.15
                    Kd=0.4
                    linear=21.0
                    max_angular=32.0
                elif self.step ==4:
                    Kp=0.14
                    Kd=0.15
                    linear=23.0
                    max_angular = 50
                elif self.step in (5,6,7) :
                    Kp=0.14
                    Kd=0.0015
                    linear=23.0
                    max_angular=50.0

                elif self.step ==2:
                    Kp=0.15 #0.15
                    Kd=0.008 #0.008
                    linear=20.0 #19
                    max_angular=50.0
                    rospy.loginfo("IMHERE")
                else:
                    Kp=0.2#0.23 -> 0.17 ->0.10
                    Kd=0.4#0.008 -> 0.0025 ->0.25
                    linear=32.0 #22.0 ->30
                    max_angular=85.0 #0.85 ->95 ->85

            if self.step in (1, 8) and self.start_time is not None:
                elapsed = rospy.Time.now().to_sec() - self.start_time
                ratio = np.clip(elapsed / 1.2, 0.0, 1.0)

                # 현재 구간의 원래 속도까지 2초 동안 가속
                target_linear = linear
                start_linear = min(10.0, target_linear)
                linear = start_linear + (target_linear - start_linear) * ratio

            p_term=Kp*err_x
            d_term=Kd*diff_x
            raw_angular=-(p_term+d_term)
            angular=np.clip(raw_angular,-max_angular,max_angular)
            self.prev_error=err_x

            wheel_distance=0.148

            v_l=linear-angular*wheel_distance*0.5
            v_r=linear+angular*wheel_distance*0.5

            self.prev_v_l=v_l
            self.prev_v_r=v_r

            self.publish_velocity(v_l,v_r)

        else:

            if self.turn_direction=='parking':
                err=-1.57-self.yaw
                err=math.atan2(math.sin(err),math.cos(err))
                angular=np.clip(err*1.5,-0.15,0.15)
                self.move(0.1,angular) #0.06
                return

            rospy.loginfo("LANE LOST -> SLOW STRAIGHT")

            self.move(0.06,0.0)

    def parking(self, image):
        if self.parking_end is not None:
            return

        if getattr(self, "parking_failed", False):
            self.move(0, 0)
            return

        if self.signal == 0:
            if self.set == 2:
                self.parking_side = 'right'
            elif self.set == 3:
                self.parking_side = 'left'
            else:
                return

            self.signal = 1
            rospy.loginfo("PARKING DETECTED: %s", self.parking_side)

        if self.sequence != 2:
            if self.yaw is None:
                self.move(0, 0)
                return

            base_yaw = self.yaw
            direction = 1.0 if self.parking_side == 'right' else -1.0
            entry_yaw = base_yaw + direction * math.radians(90)
            entry_yaw = math.atan2(math.sin(entry_yaw), math.cos(entry_yaw))

            if not self.parking_turn(entry_yaw, direction, 0.06, 0.35):
                self.parking_failed = True
                return

            if rospy.is_shutdown():
                self.move(0, 0)
                return
            self.move_time(0.06, 0, 0.5)
            if rospy.is_shutdown():
                self.move(0, 0)
                return
            self.move_time(-0.06, 0, 0.5)
            if rospy.is_shutdown():
                self.move(0, 0)
                return

            self.sequence = 2

        if self.sequence == 2:
            if not self.reverse_turn_to_exit():
                self.parking_failed = True
                return

            if rospy.is_shutdown():
                self.move(0, 0)
                return

            self.move(0, 0)
            self.parking_end = 1
            self.step = 6

    def parking_turn(self, target_yaw, direction, linear_speed, max_angular):
        tolerance = math.radians(2)
        timeout = 15.0
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(30)

        try:
            while not rospy.is_shutdown():
                if self.yaw is None:
                    rospy.logwarn("PARKING: YAW NOT AVAILABLE")
                    return False

                err = target_yaw - self.yaw
                err = math.atan2(math.sin(err), math.cos(err))

                if abs(err) <= tolerance:
                    return True

                if rospy.Time.now().to_sec() - start >= timeout:
                    rospy.logwarn("PARKING TURN TIMEOUT")
                    return False

                if direction * err < 0:
                    rospy.logwarn("PARKING TURN OVERSHOOT")
                    return False

                self.move(linear_speed, direction * max_angular)
                rate.sleep()

        finally:
            self.move(0, 0)

        return False

    def reverse_turn_to_exit(self):
        if self.parking_side == 'right':
            direction = 1.0
        elif self.parking_side == 'left':
            direction = -1.0
        else:
            self.move(0, 0)
            return False

        return self.parking_turn(1.565, direction, -0.06, 0.35)

    def parking_turn(self, target_yaw, direction, linear_speed, max_angular):
        tolerance = math.radians(2)
        timeout = 15.0
        gain = 1.2
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(30)

        try:
            while not rospy.is_shutdown():
                if self.yaw is None:
                    rospy.logwarn("PARKING: YAW NOT AVAILABLE")
                    return False

                err = target_yaw - self.yaw
                err = math.atan2(math.sin(err), math.cos(err))

                if abs(err) <= tolerance:
                    return True

                if rospy.Time.now().to_sec() - start >= timeout:
                    rospy.logwarn("PARKING TURN TIMEOUT")
                    return False

                if direction * err < 0:
                    rospy.logwarn("PARKING TURN OVERSHOOT")
                    return False

                angular = direction * min(max_angular, max(0.06, abs(err) * gain))
                linear = linear_speed * min(1.0, abs(err) / 0.3)
                self.move(linear, angular)
                rate.sleep()

        finally:
            self.move(0, 0)

        return False

    def reverse_turn_to_exit(self):
        if self.parking_side == 'right':
            direction = 1.0
        elif self.parking_side == 'left':
            direction = -1.0
        else:
            self.move(0, 0)
            return False

        return self.parking_turn(1.565, direction, -0.045, 0.35)

    def set_yaw(self,num):
        if self.yaw is None:return

        if num==1:target_yaw=1.57
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
                v_yaw=np.clip(err*1.4,-0.25,0.25) #1.2
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
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        if self._show_lut_debug:
            labels = self._color_lut.rgb_to_label(image, bgr=True)
            rgb_view = self._color_lut.label_to_rgb(labels)
            cv2.imshow('LUT LABEL', cv2.cvtColor(rgb_view, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        now = rospy.Time.now().to_sec()


        # Keep the start stage fixed until motor preparation is complete.
        if not self.drive_ready:
            self.move(0,0,tunnel=self.step in (9,10))
            return

        if self.step in (9,10):
            self.tunnel.update_lane(self.lane_detect(image),data.header.stamp.to_sec())
            return

        if self.standalone_gate_start and self.step in (8,8.5):
            if not self.drive_ready or not self.odom_received:
                self.move(0,0)
                return

        rospy.loginfo(" STEP: %d" ,self.step)

        if self.step==0:  # stop status
            self.traffic_light(image)
            return

        if self.step==1: # detect sign ~ end turn
            if self.check_turn(image):
                return
        
        elif self.step == 2: # straight find --> turn detect reset for obstacle !!~!~!~
            # self.turn_detect='left'
            self.straight = self.is_straight(image,170) #6
            if self.straight:
                self.turn_detect = 'straight'#None
                self.turn_direction = 'straight'
                self.step = 3
                # self.straight_count = 0
                self.straight =None          


        elif self.step == 3:

            # 첫 장애물 감지 시 회피 순서 시작
            if self.obstacle_stage is None:
                if not self.check_obstacle_start():
                    self.lane_tracking(image)
                    return

                self.obstacle_stage = 'left'
                self.turn_direction = 'obstacle'
                self.speed =0
                self.obstacle_stage_count = 0
                self.obstacle_turn_start = rospy.Time.now().to_sec()
                self.obstacle_clear_count = 0
                rospy.loginfo("FIRST OBSTACLE -> TURN LEFT")


            if self.obstacle_stage == 'done' and self.is_straight(image, 140):
                self.step = 4
                self.obstacle_stage = None
                self.obstacle_stage_count = 0
                self.obstacle_turn_start = None
                self.obstacle_pass_count = 0
                self.obstacle_clear_count = 0
                self.obstacle_direction = None
                self.turn_detect = None
                self.turn_direction = 'straight'
                self.prev_error = 0.0
                self.prev_target_x = None
                self.speed = 0
                self.move(0, 0)
                rospy.loginfo("@@@@@@ STRAIGHT AFTER OBSTACLE -> STEP 4 @@@@@@")
                return

            self.obstacle_move(image)
            return


        elif self.step==4: # 장애물 후 straight 발견 --> 주차구간 시작 직전임
            self.direction = 'straight'
            self.straight = self.is_straight(image,30) #25
            if self.straight:
                self.step = 5
                # self.straight_count = 0
                self.straight =None

        elif self.step==5:
            self.turn_direction='parking'

            crop_img=image[300:,:]
            height,width=crop_img.shape[:2]
            yellow_mask = self._color_lut.mask(crop_img, 2)
            self.yellow_area=cv2.countNonZero(yellow_mask)

            if self.yellow_area <=300 and self.set == None:
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('straight',1.5,0.06,0)
                self.turn_time('left',1.2,0.02,0.35) #duration,linear_speed,angular_speed)
                self.set = 1

            if self.parking_end:
                self.stop()
                return

            if self.set ==1 and (0.15<=round(self.right_distance,1)<=0.4):
                self.set = 2
            # if self.set ==1 and 0.15<=self.left_distance<=0.21: #왼쪽에뭐있음
            if self.set ==1 and 0.15<=round(self.left_distance,1)<=0.4:
                self.set = 3

            if self.set==2 or self.set==3:
                self.parking(image)
                return

        elif self.step == 6:
            self.turn_direction = 'parking2'

            crop_img = image #[230:, :]
            if crop_img.size == 0:
                self.move(0, 0)
                return

            height, width = crop_img.shape[:2]
            yellow_mask = self._color_lut.mask(crop_img, 2)

            yellow_mask[:height // 4, :] = 0
            self.yellow_area = cv2.countNonZero(yellow_mask)
            rospy.loginfo("Yellow is: %d", self.yellow_area)

            if self.yellow_area <= 200:
                self.yellow_zero_count += 1

                if self.yellow_zero_count < 5:
                    self.move(0.04, 0)
                    return

                self.move(0, 0)
                self.yellow_zero_count = 0
                self.turn_time('straight', 0.5, 0.1, 0)
                rospy.loginfo("@@@@@@ TURN LEFT @@@@@@")
                self.turn_time('left', 0.8, 0.04, 0.45)
                self.step = 7
                return

            self.yellow_zero_count = 0

            half = width // 2
            left_m = cv2.moments(yellow_mask[:, :half], binaryImage=True)
            right_m = cv2.moments(yellow_mask[:, half:], binaryImage=True)
            left_x = left_m['m10'] / left_m['m00'] if left_m['m00'] >= 10 else None
            right_x = half + right_m['m10'] / right_m['m00'] if right_m['m00'] >= 10 else None

            if left_x is not None and right_x is not None:
                lane_center = (left_x + right_x) / 2.0 + 15
                self.parking_lane_width = right_x - left_x
            elif left_x is not None and getattr(self, 'parking_lane_width', None) is not None:
                lane_center = left_x + self.parking_lane_width / 2.0
            elif right_x is not None and getattr(self, 'parking_lane_width', None) is not None:
                lane_center = right_x - self.parking_lane_width / 2.0
            else:
                self.move(0, 0)
                return

            # 중심이 화면 왼쪽이면 좌회전(+), 오른쪽이면 우회전(-)
            error = (width / 2.0 - lane_center) / (width / 2.0)
            angular = float(np.clip(error * 0.6, -0.25, 0.25))
            self.move(0.1, angular)
            return

        elif self.step == 7: # 지그재그 전, 흰/노란선 다 보이기 시작 ~ gatebar

            # self.turn_direction = 'left_yellow'
            # red_mask = self._color_lut.mask(image, 8)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange( hsv, np.array([0, 180, 107]), np.array([5, 255, 194]))
            self.red_area=cv2.countNonZero(red_mask)   
            rospy.loginfo("Red is: %3f",self.red_area)
            self.turn_direction=None

            if self.gate_state == 0:
                if 500<self.red_area<12000:
                    self.speed = 1
                    
                if (self.depth <= 0.75 and self.red_area>1300) or self.red_area >=1600:
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

                if  self.red_area<=1100:
                    self.gate_open_count += 1
                    rospy.loginfo("GATE OPEN COUNT: %d / 10",self.gate_open_count)
                else:
                    self.gate_open_count=0

                if self.gate_open_count >= 20:
                    rospy.loginfo("@@@ GATE OPEN -> STEP 8 @@@")

                    self.start_time = rospy.Time.now().to_sec()
                    self.prev_error = 0.0
                    self.prev_target_x = None

                    self.gate_state = 2
                    self.gate_open_count = 0
                    self.turn_direction = None
                    self.step = 8
                    self.speed = 0

                return  

        elif self.step==8:
            crop_img=image[300:,:]
            height,width=crop_img.shape[:2]
            white_mask = self._color_lut.mask(crop_img, 1)
            yellow_mask = self._color_lut.mask(crop_img, 2)
            yellow_area=cv2.countNonZero(yellow_mask)
            white_area=cv2.countNonZero(white_mask)

            rospy.loginfo("Y: %.3f W: %.3f",yellow_area,white_area)

            if yellow_area<=30 and white_area<=30:
                self.maze_count+=1
            else:
                self.maze_count=0

            if self.maze_count>=5:
                self.step=8.5
                return

        elif self.step==8.5:
            self.move_time_yaw_hold(0.06,self.gate_approach_yaw,1)
            self.move(0,0)

            with self.command_lock:
                self.step=9
                self.tunnel_generation+=1
                self.tunnel_active=False
            return


        self.lane_tracking(image)

    def publish_velocity(self,v_l,v_r): # 속도 퍼블리시 
        msg=Twist()
        if (self.step ==3 and self.speed ==2) or self.step ==6 :
            msg.linear.x=(self.wheel_radius*(v_r+v_l)/2.0)*0.1 #0.1
            msg.angular.z=(self.wheel_radius*(v_r-v_l)/self.wheel_separation)*0.156 #0.2
        else:
            msg.linear.x=(self.wheel_radius*(v_r+v_l)/2.0)*0.12 #0.13
            msg.angular.z=(self.wheel_radius*(v_r-v_l)/self.wheel_separation)*0.35 #0.28
        with self.command_lock:
            if not self.shutting_down and self.step not in (9,10):
                if not self.drive_ready:
                    msg=Twist()
                self.cmd_pub.publish(msg)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()
