#!/usr/bin/env python3
import rospy
import cv2
import math
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StraightValueChecker:
    def __init__(self):
        rospy.init_node('straight_value_checker',anonymous=True)
        self.bridge=CvBridge()
        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)
        self.turn_direction='right'

    def get_curve(self,mask):
        y,x=np.nonzero(mask)
        if len(x)<200:return None
        coeff=np.polyfit(y,x,2)
        h=mask.shape[0]
        return abs(coeff[0])*(h**2)

    def image_callback(self,msg):
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        crop_img=image[300:,:]
        height,width=crop_img.shape[:2]

        hsv=cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)

        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179,55,255]))
        yellow_mask=cv2.inRange(hsv,np.array([15,100,100]),np.array([50,255,255]))

        if self.turn_direction=='left':
            yellow_mask[:,width//2:]=0
            yellow_mask[:height//2,:]=0
            white_mask[:height//4,:]=0

        elif self.turn_direction=='right':
            white_mask[:,:width//2]=0
        mask=cv2.bitwise_or(white_mask,yellow_mask)

        mask=cv2.erode(mask,None,iterations=1)
        mask=cv2.dilate(mask,None,iterations=2)
        edges=cv2.Canny(mask,50,150)

        lines=cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=25,maxLineGap=30)

        angles=[]
        view=crop_img.copy()

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line[0]
                dx=x2-x1
                dy=y2-y1

                if abs(dy)<20:continue

                angle=math.degrees(math.atan2(dx,dy))

                if abs(angle)<45:
                    angles.append(angle)
                    cv2.line(view,(x1,y1),(x2,y2),(0,0,255),2)

        if len(angles)>0:
            mean_angle=float(np.mean(angles))
            angle_std=float(np.std(angles))
        else:
            mean_angle=None
            angle_std=None

        white_curve=self.get_curve(white_mask)
        yellow_curve=self.get_curve(yellow_mask)

        curve_values=[v for v in [white_curve,yellow_curve] if v is not None]
        max_curve=max(curve_values) if curve_values else None

        mean_text="None" if mean_angle is None else f"{mean_angle:.2f}"
        std_text="None" if angle_std is None else f"{angle_std:.2f}"
        white_text="None" if white_curve is None else f"{white_curve:.2f}"
        yellow_text="None" if yellow_curve is None else f"{yellow_curve:.2f}"
        max_text="None" if max_curve is None else f"{max_curve:.2f}"

        rospy.loginfo("LINES: %d | MEAN: %s | STD: %s | WHITE_CURVE: %s | YELLOW_CURVE: %s | MAX_CURVE: %s",
                      len(angles),mean_text,std_text,white_text,yellow_text,max_text)

        cv2.putText(view,f"LINES: {len(angles)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(view,f"MEAN: {mean_text}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(view,f"STD: {std_text}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(view,f"WHITE CURVE: {white_text}",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(view,f"YELLOW CURVE: {yellow_text}",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(view,f"MAX CURVE: {max_text}",(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow('straight_value_checker',view)
        cv2.imshow('lane_mask',mask)
        cv2.waitKey(1)

if __name__=='__main__':
    StraightValueChecker()
    rospy.spin()
    cv2.destroyAllWindows()
