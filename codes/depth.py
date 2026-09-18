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

        rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.depth_callback,queue_size=1,buff_size=2**24)
        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)

    def image_callback(self,msg):

        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([10, 255, 160]))        # mask=cv2.bitwise_or(red_mask)
        filtered=cv2.bitwise_and(image,image,mask=red_mask)
        red_area=cv2.countNonZero(red_mask)
        
        # gray=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
        rospy.loginfo("AREA : %.2f ",red_area)
        cv2.imshow('COLORED',filtered)
        cv2.imshow('COLOR',image)
        cv2.waitKey(1)

    def depth_callback(self,data):
        depth=self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')

        height,width=depth.shape[:2]

        x1=int(width*0.25)
        x2=int(width*0.75)
        y1=int(height*0.25)
        y2=int(height*0.75)
        roi=depth[::int(height*2/3),:]
        # roi=depth[y1:y2,x1:x2]

        valid=roi[(roi>0)&np.isfinite(roi)]

        if len(valid)==0:
            rospy.loginfo("DEPTH: NO DATA")
            return

        distance=np.percentile(valid,5)

        if data.encoding=='16UC1':
            distance/=1000.0

        rospy.loginfo("CLOSEST FRONT DEPTH: %.3f m",distance)

if __name__=='__main__':
    DepthViewer()
    rospy.spin()