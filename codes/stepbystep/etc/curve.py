#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CurveTest:
    def __init__(self):
        rospy.init_node('curve_test',anonymous=True)
        self.bridge=CvBridge()
        self.threshold=95
        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)

    def detect_curve(self,white_mask,yellow_mask):
        curve_values=[]

        for mask in [white_mask,yellow_mask]:
            h=mask.shape[0]
            upper_mask=mask[:h//2,:]
            lower_mask=mask[h//2:,:]

            M_upper=cv2.moments(upper_mask)
            M_lower=cv2.moments(lower_mask)

            if M_upper['m00']>0 and M_lower['m00']>0:
                cx_upper=int(M_upper['m10']/M_upper['m00'])
                cx_lower=int(M_lower['m10']/M_lower['m00'])

                curve=abs(cx_upper-cx_lower)
                curve_values.append(curve)

        if not curve_values:
            return None,0

        curve=max(curve_values)
        return curve>self.threshold,curve

    def image_callback(self,msg):
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        crop_img=image[300:,:]
        height,width=crop_img.shape[:2]

        hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179,55,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

        white_mask[:,:width//2]=0

        is_curve,curve=self.detect_curve(white_mask,yellow_mask)

        if is_curve is None:
            result="UNKNOWN"
            color=(0,255,255)
        elif is_curve:
            result="CURVE"
            color=(0,0,255)
        else:
            result="STRAIGHT"
            color=(0,255,0)

        rospy.loginfo_throttle(0.5,"LANE: %s / CURVE VALUE: %.1f / THRESHOLD: %d",result,curve,self.threshold)

        cv2.putText(crop_img,result,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,3)
        cv2.putText(crop_img,"CURVE: %.1f"%curve,(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(crop_img,"THRESHOLD: %d"%self.threshold,(30,140),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("CURVE TEST",crop_img)
        cv2.imshow("WHITE MASK",white_mask)
        cv2.imshow("YELLOW MASK",yellow_mask)
        cv2.waitKey(1)

if __name__=='__main__':
    controller=CurveTest()
    rospy.spin()