import rospy
import cv2
import numpy as np
import math
import subprocess
import actionlib
import os

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from std_msgs.msg import UInt8

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.bridge = CvBridge()
        self.msg = Twist()

        self.hide = 0
        self.hide_yellow = 0
        self.hide_white = 0



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


        self.exit_x=None
        self.exit_y=None
        

        self.cmd_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.img_callback,queue_size=1)
    # self.scan_sub = rospy.Subscriber('/scan',LaserScan,self.scan_callback)


    def img_callback(self,data):
        image=cv2.imdecode(np.frombuffer(data.data,np.uint8),cv2.IMREAD_COLOR)
        image = image[160:,:]
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        yellow_mask=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))

        white_mask=cv2.erode(white_mask,None,iterations=1)
        white_mask=cv2.dilate(white_mask,None,iterations=2)

        yellow_mask=cv2.erode(yellow_mask,None,iterations=1)
        yellow_mask=cv2.dilate(yellow_mask,None,iterations=2)

        mask=cv2.bitwise_or(white_mask,yellow_mask)
        result=cv2.bitwise_and(image,image,mask=mask)

        cv2.imshow('COLOR DETECTION',result)
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


    # def publish_velocity(self):
    #     self.cmd_pub.publish(self.msg)


if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()
