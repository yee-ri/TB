#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthViewer:
    def __init__(self):
        rospy.init_node('depth_viewer',anonymous=True)
        self.bridge=CvBridge()
        self.image = None
        self.image_view=None
        self.depth_view=None
        rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)
        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)

    def image_callback(self,msg):
        self.image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        height,width=self.image.shape[:2]
        self.image=self.image[:int(height*2/3),:int(width*3/4)]

        hsv=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        red_mask=cv2.inRange(hsv,np.array([0,180,100]),np.array([10,255,160]))
        red_area=cv2.countNonZero(red_mask)
        filtered=cv2.bitwise_and(self.image,self.image,mask=red_mask)

        rospy.loginfo("AREA : %.2f",red_area)
        self.image_view=filtered

    def depth_callback(self,data):
        depth=self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        height,width=depth.shape[:2]
        roi=depth[:int(height*2/3),:int(width*3/4)]
        valid=roi[(roi>0)&np.isfinite(roi)]

        if len(valid)==0:
            rospy.loginfo("DEPTH: NO DATA")
            return

        distance=np.percentile(valid,5)
        if data.encoding=='16UC1': distance/=1000.0
        rospy.loginfo("CLOSEST FRONT DEPTH: %.3f m",distance)

        roi_view=cv2.normalize(roi,None,0,255,cv2.NORM_MINMAX)
        self.depth_view=np.uint8(roi_view)

    def run(self):
        rate=rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.image is not None: cv2.imshow('image',self.image)
            if self.image_view is not None: cv2.imshow('COLOR',self.image_view)
            if self.depth_view is not None: cv2.imshow('DEPTH ROI',self.depth_view)
            cv2.waitKey(1)
            rate.sleep()

        cv2.destroyAllWindows()

if __name__=='__main__':
    viewer=DepthViewer()
    viewer.run()