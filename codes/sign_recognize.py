#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class TemplateMatch:

    def __init__(self):
        rospy.init_node('template_match',anonymous=True)

        self.bridge=CvBridge()
        self.depth_sub=rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn1.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn1.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)


    def depth_callback(self,data):
        depth_image=self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        height,width=depth_image.shape
        cx=width//2
        cy=height//2
        depth=depth_image[cy,cx]
        if depth>0: rospy.loginfo("CENTER DEPTH: %.3f m",depth/1000.0)

    def image_callback(self,msg):
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')

        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))
        #white_mask=cv2.inRange(hsv,np.array([0,0,150]),np.array([179,50,255]))
        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179, 55, 255]))
        blue_mask=cv2.inRange(hsv,np.array([90,80,50]),np.array([130,255,255]))

        mask=cv2.bitwise_or(yellow_mask,white_mask)
        mask=cv2.bitwise_or(mask,blue_mask)

        filtered=cv2.bitwise_and(image,image,mask=mask)

        cv2.imshow('COLOR',image)
        # cv2.imshow('COLOR FILTER',filtered)
        cv2.waitKey(1)

        gray=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)

        left_result=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,lmax_val,_,lmax_loc=cv2.minMaxLoc(left_result)

        left_percent=lmax_val*100

        right_result=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,rmax_val,_,rmax_loc=cv2.minMaxLoc(right_result)

        right_percent=rmax_val*100

        # rospy.loginfo("LEFT sign: %.1f%%     RIGHT sign: %.1f%%",left_percent,right_percent)

if __name__=='__main__':
    node=TemplateMatch()
    rospy.spin()
    cv2.destroyAllWindows()