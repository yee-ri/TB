#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.cmd_pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.img_callback,queue_size=1)

        self.bridge=CvBridge()
        self.msg=Twist()

        self.v_l=0.0
        self.v_r=0.0
        self.last_v_l=0.0
        self.last_v_r=0.0

        self.cx_w=0
        self.cy_w=0
        self.cx_y=0
        self.cy_y=0

        self.wheel_radius=0.0475
        self.wheel_separation=0.148
        self.lost_count=0
        self.lane_width=260.0

        self.mode='stop'
        self.light=None
        self.sign=None

        self.prev_white_x=None
        self.prev_yellow_x=None

        self.hide=0
        self.hide_yellow=0
        self.hide_white=0

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn2.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn2.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

    def detect_light(self,data):
        hsv_frame=cv2.cvtColor(data,cv2.COLOR_BGR2HSV)
        green_mask=cv2.inRange(hsv_frame,np.array([35,50,50]),np.array([85,255,255]))
        green_mask=cv2.erode(green_mask,None,iterations=1)
        green_mask=cv2.dilate(green_mask,None,iterations=2)

        if cv2.countNonZero(green_mask)>=200:
            rospy.loginfo('DETECTED GREEN LIGHT')
            self.light='green'
            self.mode='lane'
            return 'green'

        return None

    def detect_sign(self,data):
        self.detect_lturn(data)
        self.detect_rturn(data)

        if (self.max_val_l>self.max_val_r+0.04 and self.max_val_l>0.35) or self.max_val_l>0.4:
            return 'left'

        if (self.max_val_r>self.max_val_l+0.04 and self.max_val_r>0.35) or self.max_val_r>0.4:
            return 'right'

        return None

    def detect_lturn(self,data):
        gray=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_l,_,_=cv2.minMaxLoc(res)

    def detect_rturn(self,data):
        gray=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        res=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_r,_,_=cv2.minMaxLoc(res)

    def img_callback(self,data):
        image=cv2.imdecode(np.frombuffer(data.data,np.uint8),cv2.IMREAD_COLOR)

        if image is None:
            return

        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        yellow_mask=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))
        white_mask=cv2.inRange(hsv,np.array([0,0,200]),np.array([179,50,255]))
        blue_mask=cv2.inRange(hsv,np.array([90,80,50]),np.array([130,255,255]))

        color_mask=cv2.bitwise_or(yellow_mask,white_mask)
        color_mask=cv2.bitwise_or(color_mask,blue_mask)
        filtered=cv2.bitwise_and(image,image,mask=color_mask)

        cv2.imshow('COLOR FILTER',filtered)
        cv2.waitKey(1)

        height,width=image.shape[:2]

        if self.mode=='stop':
            if self.detect_light(image)!='green':
                return

        if self.mode=='lane':
            self.sign=self.detect_sign(filtered)

            if self.sign=='left':
                self.mode='turn_left'
                self.hide_yellow=1
                self.prev_white_x=None
                rospy.loginfo('@@@ TURN LEFT !!!! @@@')

            elif self.sign=='right':
                self.mode='turn_right'
                self.hide_white=1
                self.prev_yellow_x=None
                rospy.loginfo('@@@ TURN RIGHT !!!! @@@')

        roi_list=[
            (int(height*0.55),height),
            (int(height*0.45),int(height*0.72)),
            (int(height*0.35),int(height*0.55))
        ]

        selected_white_mask=None
        selected_yellow_mask=None
        white_line=None
        yellow_line=None
        roi_start=0

        for start_y,end_y in roi_list:
            crop_img=image[start_y:end_y,:]
            hsv_frame=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

            white_roi=cv2.inRange(hsv_frame,np.array([0,0,200]),np.array([179,50,255]))
            yellow_roi=cv2.inRange(hsv_frame,np.array([20,100,100]),np.array([50,255,255]))

            if self.hide_yellow==1:
                yellow_roi[:,3*width//4:]=0
                white_roi[:,:]=0
            elif self.hide_white==1:
                white_roi[:,:width//2]=0
                yellow_roi[:,:]=0

            white_roi=cv2.erode(white_roi,None,iterations=1)
            white_roi=cv2.dilate(white_roi,None,iterations=2)
            yellow_roi=cv2.erode(yellow_roi,None,iterations=1)
            yellow_roi=cv2.dilate(yellow_roi,None,iterations=2)

            if self.mode=='lane':
                expected_white=width/2.0+self.lane_width/2.0
                expected_yellow=width/2.0-self.lane_width/2.0
            else:
                expected_white=None
                expected_yellow=None

            white_candidate=self.find_line(white_roi,self.prev_white_x,expected_white)
            yellow_candidate=self.find_line(yellow_roi,self.prev_yellow_x,expected_yellow)

            if self.mode=='lane' and white_candidate is not None and yellow_candidate is not None:
                lane_gap=white_candidate[0]-yellow_candidate[0]
                if lane_gap<80 or lane_gap>width*0.75:
                    white_jump=self.candidate_distance(white_candidate,self.prev_white_x,expected_white)
                    yellow_jump=self.candidate_distance(yellow_candidate,self.prev_yellow_x,expected_yellow)
                    if white_jump<=yellow_jump:
                        yellow_candidate=None
                    else:
                        white_candidate=None

            if white_candidate is not None or yellow_candidate is not None:
                selected_white_mask=white_roi
                selected_yellow_mask=yellow_roi
                white_line=white_candidate
                yellow_line=yellow_candidate
                roi_start=start_y
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

        if self.mode=='turn_left':
            if yellow_line is not None:
                target_x=self.cx_y+self.lane_width/2.0
                self.lost_count=0

        elif self.mode=='turn_right':
            if white_line is not None:
                target_x=self.cx_w-self.lane_width/2.0
                self.lost_count=0

        elif white_line is not None and yellow_line is not None:
            measured_lane_width=self.cx_w-self.cx_y

            if 80<measured_lane_width<width*0.75:
                self.lane_width=0.9*self.lane_width+0.1*measured_lane_width
                target_x=(self.cx_w+self.cx_y)/2.0
                self.lost_count=0
            else:
                white_error=abs(self.cx_w-(center_x+self.lane_width/2.0))
                yellow_error=abs(self.cx_y-(center_x-self.lane_width/2.0))

                if white_error<yellow_error:
                    target_x=self.cx_w-self.lane_width/2.0
                else:
                    target_x=self.cx_y+self.lane_width/2.0
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

        if selected_white_mask is not None:
            cv2.imshow('WHITE ROI',selected_white_mask)
        if selected_yellow_mask is not None:
            cv2.imshow('YELLOW ROI',selected_yellow_mask)
        cv2.waitKey(1)

    def candidate_distance(self,candidate,previous_x,expected_x):
        if candidate is None:
            return float('inf')
        if previous_x is not None:
            return abs(candidate[0]-previous_x)
        if expected_x is not None:
            return abs(candidate[0]-expected_x)
        return 0

    def find_line(self,mask,previous_x,expected_x=None):
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

        if previous_x is not None:
            candidate=min(candidates,key=lambda x:abs(x[0]-previous_x))
            if abs(candidate[0]-previous_x)>120:
                return None
            return candidate

        if expected_x is not None:
            return min(candidates,key=lambda x:abs(x[0]-expected_x))

        return max(candidates,key=lambda x:x[2])

    def publish_velocity(self):
        self.cmd_pub.publish(self.msg)
        rospy.loginfo('linear: %.3f angular: %.3f',self.msg.linear.x,self.msg.angular.z)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()
