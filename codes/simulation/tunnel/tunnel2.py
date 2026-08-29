import rospy
import cv2
import numpy as np
import math
import subprocess
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
        self.lturn_template = cv2.resize(self.lturn_template,(7,7))

        self.rturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/rturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.rturn_template = cv2.resize(self.rturn_template,(7,7))

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

        self.mode ='lane'
        self.c_mode = 0
        self.gain = 1
        self.step = 0
        self.alpha = 0

        self.sign = None
        self.sign_c = False
        self.light = None

        self.yaw = None
        self.yaw_hold = 0

        self.ready_gate = None
        self.stop = 0
        self.parking_center_count = 0
        self.after_turn = None
        self.straight_count = 0
        self.straight = None
        self.parking_yellow_count = 0
        self.parking_yellow_enter_area = 1200

        self.slam_process=None
        self.tf_process=None
        self.slam_started=False
        self.tunnel_in_time=None
        self.tunnel_start_time=None
        self.tunnel_turning=False
        self.tunnel_target_yaw=None
        self.tunnel_exit_ready=False
        self.tunnel_no_yellow_count=0
        self.tunnel_yellow_count=0
        self.left_distance=float('inf')
        self.front_distance=float('inf')
        self.right_distance=float('inf')
        self.left_open=float('inf')
        self.front_open=float('inf')
        self.right_open=float('inf')

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
    def odom_callback(self,data):
        q = data.pose.pose.orientation
        _,_,self.yaw = euler_from_quaternion([q.x,q.y,q.z,q.w])

    def scan_callback(self,data):
        left_ranges=[]
        front_ranges=[]
        right_ranges=[]
        for i in range(len(data.ranges)):
            distance=data.ranges[i]
            if np.isinf(distance) or np.isnan(distance) or distance<=0:
                continue
            angle=data.angle_min+i*data.angle_increment
            angle_deg=np.degrees(angle)
            if angle_deg>180:angle_deg-=360
            if 30<=angle_deg<=80:right_ranges.append(distance)
            elif -20<=angle_deg<=20:front_ranges.append(distance)
            elif -80<=angle_deg<=-30:left_ranges.append(distance)
        self.left_distance=min(left_ranges) if left_ranges else float('inf')
        self.front_distance=min(front_ranges) if front_ranges else float('inf')
        self.right_distance=min(right_ranges) if right_ranges else float('inf')
        self.left_open=float(np.percentile(left_ranges,70)) if left_ranges else float('inf')
        self.front_open=float(np.percentile(front_ranges,50)) if front_ranges else float('inf')
        self.right_open=float(np.percentile(right_ranges,70)) if right_ranges else float('inf')
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
    def move(self,linear,angular):
        self.msg.linear.x = linear
        self.msg.linear.y = 0
        self.msg.linear.z = 0
        self.msg.angular.x = 0
        self.msg.angular.y = 0
        self.msg.angular.z = angular
        self.publish_velocity()

    def get_yellow_area(self,image):
        height,width=image.shape[:2]
        roi=image[int(height*0.55):height,:]
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))
        mask=cv2.erode(mask,None,iterations=1)
        mask=cv2.dilate(mask,None,iterations=2)
        return cv2.countNonZero(mask)
    
    def start_slam(self):
        if self.slam_started:return
        self.move(0,0)
        self.tf_process=subprocess.Popen(['rosrun','tf2_ros','static_transform_publisher','0','0','0','0','0','0','base_footprint','base_scan'])
        self.slam_process=subprocess.Popen(['rosrun','gmapping','slam_gmapping','scan:=/scan','_base_frame:=base_footprint','_odom_frame:=odom','_map_frame:=map'])
        self.slam_started=True
        self.tunnel_start_time=rospy.Time.now()
        rospy.loginfo("SLAM START")

    def stop_slam(self):
        if self.slam_process is not None and self.slam_process.poll() is None:self.slam_process.terminate()
        if self.tf_process is not None and self.tf_process.poll() is None:self.tf_process.terminate()
        self.slam_process=None
        self.tf_process=None
        self.slam_started=False

    def start_tunnel_turn(self,delta):
        if self.yaw is None:return
        self.tunnel_target_yaw=self.yaw+delta
        self.tunnel_target_yaw=math.atan2(math.sin(self.tunnel_target_yaw),math.cos(self.tunnel_target_yaw))
        self.tunnel_turning=True

    def run_tunnel_turn(self):
        if not self.tunnel_turning or self.yaw is None:return False
        err=self.tunnel_target_yaw-self.yaw
        err=math.atan2(math.sin(err),math.cos(err))
        if abs(err)<0.05:
            self.move(0,0)
            self.tunnel_turning=False
            self.tunnel_target_yaw=None
            return False
        self.move(0,np.clip(err*2.5,-0.55,0.55))
        return True
    
    def tunnel(self,image):
        self.start_slam()
        yellow_area=self.get_yellow_area(image)
        if yellow_area<100:
            self.tunnel_no_yellow_count+=1
            if self.tunnel_no_yellow_count>=10:self.tunnel_exit_ready=True
        else:
            self.tunnel_no_yellow_count=0
        if self.tunnel_exit_ready and yellow_area>350:
            self.tunnel_yellow_count+=1
        else:
            self.tunnel_yellow_count=0
        if self.tunnel_yellow_count>=5:
            self.move(0,0)
            self.stop_slam()
            self.mode='lane'
            self.prev_white_x=None
            self.prev_yellow_x=None
            self.tunnel_turning=False
            self.tunnel_exit_ready=False
            self.tunnel_no_yellow_count=0
            self.tunnel_yellow_count=0
            rospy.loginfo("TUNNEL EXIT -> LANE")
            return
        rospy.loginfo("TUNNEL L %.2f F %.2f R %.2f LO %.2f FO %.2f RO %.2f Y %d",self.left_distance,self.front_distance,self.right_distance,self.left_open,self.front_open,self.right_open,yellow_area)

        if self.run_tunnel_turn():return

        if self.front_distance<0.28:
            rospy.loginfo("IH HERE11")
            if self.left_open>0.55 and self.left_open>=self.right_open:
                self.start_tunnel_turn(math.pi/2.0)
            elif self.right_open>0.55:
                self.start_tunnel_turn(-math.pi/2.0)
            else:
                self.start_tunnel_turn(math.pi)
            return

        if self.left_open>0.30:
            rospy.loginfo("IH HERE22")
            self.start_tunnel_turn(math.pi/2.0)
            return

        if self.front_open>0.3:
            rospy.loginfo("IH HERE33")
            angular=0.0
            if np.isfinite(self.left_distance) and np.isfinite(self.right_distance) and self.left_distance<0.8 and self.right_distance<0.8:
                angular=np.clip((self.left_distance-self.right_distance)*1.8,-0.35,0.35)
            elif self.left_distance<0.22:
                angular=-0.25
            elif self.right_distance<0.22:
                angular=0.25
            self.move(0.15,angular)
            return
        if self.right_open>0.35:
            rospy.loginfo("IH HERE44")
            self.start_tunnel_turn(-math.pi/2.0)
            return
        self.start_tunnel_turn(math.pi)

    def move_yaw_hold(self,linear,target_yaw):
        if self.yaw is None:
            self.move(linear,0)
            return

        yaw_err = target_yaw-self.yaw
        yaw_err = math.atan2(math.sin(yaw_err),math.cos(yaw_err))

        angular = np.clip(yaw_err*3.0,-0.4,0.4)

        self.move(linear,angular)

        rospy.loginfo("MOVE YAW HOLD: YAW %.3f TARGET %.3f ERR %.3f ANG %.3f",self.yaw,target_yaw,yaw_err,angular)

    def img_callback(self,data):
        image = self.bridge.imgmsg_to_cv2(data,'bgr8')
        rospy.loginfo(self.mode)

        rospy.loginfo("left: %3f   front: %3f   right: %3f", self.left_distance,self.front_distance,self.right_distance)

        if self.mode=='lane':
            yellow_area=self.get_yellow_area(image)
            if 0.1<=self.left_distance<=0.15 and 0.2<=self.front_distance<=0.45 and 0.55<=self.right_distance<=0.65 and yellow_area<100:
                self.mode='tunnel_in'
                self.tunnel_in_time=rospy.Time.now()
                return

        if self.mode=='tunnel_in':
            self.move_yaw_hold(0.2,-1.45)

            if (rospy.Time.now()-self.tunnel_in_time).to_sec()>=2.0:
                self.move(0,0)
                self.mode='tunnel'
                self.tunnel_exit_ready=False
                self.tunnel_no_yellow_count=0
                self.tunnel_yellow_count=0
            return
        if self.mode=='tunnel':
            self.tunnel(image)
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
                # if self.after_parking <=1:
                target_x = self.cx_y+self.lane_width/2.0
                # elif self.after_parking ==2:
                #     target_x = self.cx_y+self.lane_width/2.0 + 5
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
