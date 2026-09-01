#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.img_callback,queue_size=1)

        self.bridge = CvBridge()

        self.msg = Twist()
        self.v_l = 0.0
        self.v_r = 0.0

        self.cx_w=0
        self.cy_w=0
        self.cx_y=0
        self.cy_y=0

        self.wheel_radius = 0.0475
        self.wheel_separation = 0.148

####### related with TURN SIGN ######
        self.hide = 0
        self.hide_yellow = 0
        self.hide_white = 0
        self.template=cv2.imread('/home/sj/Desktop/TB/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.template = cv2.resize(self.template,(100,100))
        if self.template is None:
            rospy.logerr("TEMPLATE LOAD FAIL")
            rospy.signal_shutdown("template load fail")
            return
      
#####################################  

    def img_callback(self,data):

        image=cv2.imdecode(np.frombuffer(data.data,np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            return

        height,width=image.shape[:2]

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



        roi_list=[
            (int(height*0.82),int(height*0.94)),
            (int(height*0.68),int(height*0.80)),
            (int(height*0.54),int(height*0.66))
        ]

        selected_white_mask=None
        selected_yellow_mask=None
        white_line=None
        yellow_line=None
        roi_start=0
        roi_end=height

        for start_y,end_y in roi_list:
            crop_img=image[start_y:end_y,:]
            hsv_frame=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

            white_mask=cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
            yellow_mask=cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

            if self.hide_yellow == 1: # TURN LEFT 반말고 왼쪽 노란거 1/4 날리기
                yellow_mask[:,3*width//4:] = 0
                white_mask[:,: ] = 0

            elif self.hide_white == 1: # TURN RIGHT 흰거 왼쪽 날리기
                white_mask[:,:width//2] = 0
                yellow_mask[:,: ] = 0


            white_mask=cv2.erode(white_mask,None,iterations=1)
            white_mask=cv2.dilate(white_mask,None,iterations=2)
            yellow_mask=cv2.erode(yellow_mask,None,iterations=1)
            yellow_mask=cv2.dilate(yellow_mask,None,iterations=2)

            white_candidate=self.find_line(white_mask,self.prev_white_x)
            yellow_candidate=self.find_line(yellow_mask,self.prev_yellow_x)

            if white_candidate is not None or yellow_candidate is not None:
                selected_white_mask=white_mask
                selected_yellow_mask=yellow_mask
                white_line=white_candidate
                yellow_line=yellow_candidate
                roi_start=start_y
                roi_end=end_y
                break

        center_x=width/2.0
        target_x=None

        if white_line is not None:
            self.cx_w=white_line[0]
            self.cy_w=white_line[1]+roi_start
            self.prev_white_x=self.cx_w

        if yellow_line is not None:
            self.cx_y=yellow_line[0]
            self.cy_y=yellow_line[1]+roi_start
            self.prev_yellow_x=self.cx_y

        if white_line is not None and yellow_line is not None:
            measured_lane_width=abs(self.cx_w-self.cx_y)
            if 100<measured_lane_width<width:
                self.lane_width=0.9*self.lane_width+0.1*measured_lane_width
            target_x=(self.cx_w+self.cx_y)/2.0
            self.lost_count=0
        elif white_line is not None:
            target_x=self.cx_w-self.lane_width/2.0
            self.lost_count=0
        elif yellow_line is not None:
            target_x=self.cx_y+self.lane_width/2.0
            self.lost_count=0


        if target_x is not None:
            error_x=target_x-center_x
            linear=5.0
            angular=np.clip(-float(error_x)/5.0,-10.0,10.0)
            wheel_distance=0.2
            self.v_l=linear-angular*wheel_distance*0.9
            self.v_r=linear+angular*wheel_distance*0.9
            self.last_v_l=self.v_l
            self.last_v_r=self.v_r
        elif self.lost_count<8:
            self.lost_count+=1
            self.v_l=self.last_v_l*0.9
            self.v_r=self.last_v_r*0.9
        else:
            self.v_l=0.0
            self.v_r=0.0

        
        self.msg.linear.x=self.wheel_radius*(self.v_r+self.v_l)/2.0*0.2
        self.msg.angular.z=self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation*0.2
        self.publish_velocity()

        display=image.copy()

        if selected_white_mask is None:
            selected_white_mask=np.zeros((1,width),dtype=np.uint8)
        if selected_yellow_mask is None:
            selected_yellow_mask=np.zeros((1,width),dtype=np.uint8)

        # cv2.imshow('line tracking',display)
        # cv2.imshow('white mask',selected_white_mask)
        # cv2.imshow('yellow mask',selected_yellow_mask)
        # cv2.waitKey(3)

    def find_line(self,mask,previous_x):
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<80:
                continue
            M=cv2.moments(contour)
            if M['m00']==0:
                continue
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            candidates.append((cx,cy,area))

        if not candidates:
            return None
        if previous_x is None:
            return max(candidates,key=lambda x:x[2])
        return min(candidates,key=lambda x:abs(x[0]-previous_x))

    def publish_velocity(self):
        self.cmd_pub.publish(self.msg)
        rospy.loginfo('linear: %.3f angular: %.3f',self.msg.linear.x,self.msg.angular.z)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()

