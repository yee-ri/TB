#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class turtlebot():
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub=rospy.Publisher("/cmd_vel",Twist,queue_size=10)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.img_callback,queue_size=1)

        self.bridge=CvBridge()
        self.msg=Twist()

        self.lane_width = 0.26
        self.v_l=0.0
        self.v_r=0.0

        self.cx_w=0
        self.cy_w=0
        self.cx_y=0
        self.cy_y=0

        self.wheel_radius=0.0475
        self.wheel_separation=0.148

    def img_callback(self,data):
        image=cv2.imdecode(np.frombuffer(data.data,np.uint8),cv2.IMREAD_COLOR)

        if image is None:
            return

        height,width=image.shape[:2]
        rospy.loginfo("H: %f.  W: %f",height,width) # 480 848

        self.crop_img=image[230:,:]
        display_image=self.crop_img.copy()

        hsvFrame=cv2.cvtColor(self.crop_img,cv2.COLOR_BGR2HSV)

        white_lower=np.array([0,0,150])
        white_upper=np.array([179,50,255])
        white_mask=cv2.inRange(hsvFrame,white_lower,white_upper)

        yellow_lower=np.array([20,100,100])
        yellow_upper=np.array([50,255,255])
        yellow_mask=cv2.inRange(hsvFrame,yellow_lower,yellow_upper)

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

        h,w,_=self.crop_img.shape
        center_x=w/2.0

        if M_w["m00"]>0 and M_y["m00"]>0:
            self.cx_w=int(M_w["m10"]/M_w["m00"])
            self.cy_w=int(M_w["m01"]/M_w["m00"])

            self.cx_y=int(M_y["m10"]/M_y["m00"])
            self.cy_y=int(M_y["m01"]/M_y["m00"])

    
            self.cen=(self.cx_w+self.cx_y)/2.0
            self.err_x=self.cen-center_x
            rospy.loginfo("White:  %f.   YELLOW: %f ",self.cx_w,self.cx_y)
            rospy.loginfo("center %f.   ERR1 is %f ",self.cen,self.err_x)

            # if abs(self.err_x)<25:
            #     self.err_x=0

            linear=7
            angular=-float(self.err_x)/2*0.2

            wheel_distance=0.148

            self.v_l=linear-angular*wheel_distance*1.2
            self.v_r=linear+angular*wheel_distance*1.2

            self.publish_velocity()

        elif M_w["m00"]>0 and M_y["m00"]<=0:
            self.cx_w=int(M_w["m10"]/M_w["m00"])
            self.cy_w=int(M_w["m01"]/M_w["m00"])

            self.cen=self.cx_w-self.lane_width/2
            self.err_x=self.cen-center_x-220

            rospy.loginfo("ERR2 is %f ",self.err_x)
            # if abs(self.err_x)<40:
            #     self.err_x=0
            linear=7
            angular=-float(self.err_x)/2
            wheel_distance=0.148

            self.v_l=linear-angular*wheel_distance*0.9
            self.v_r=linear+angular*wheel_distance*0.9

            self.publish_velocity()

        elif M_w["m00"]<=0 and M_y["m00"]>0:
            self.cx_y=int(M_y["m10"]/M_y["m00"])
            self.cy_y=int(M_y["m01"]/M_y["m00"])

            self.cen=self.cx_y+self.lane_width
            self.err_x=self.cen-center_x+220
            rospy.loginfo("ERR3 is %f ",self.err_x)
            if abs(self.err_x)<42:
                 self.err_x=0

            linear=7
            angular=-float(self.err_x)/5
            wheel_distance=0.148

            self.v_l=linear-angular*wheel_distance*0.9
            self.v_r=linear+angular*wheel_distance*0.9

            self.publish_velocity()

        else:
            self.v_l=0.0
            self.v_r=0.0
            self.publish_velocity()

        cv2.circle(display_image,(self.cx_w,self.cy_w),8,(255,0,0),-1)
        cv2.circle(display_image,(self.cx_y,self.cy_y),8,(0,255,255),-1)
        cv2.line(display_image,(int(center_x),0),(int(center_x),h),(0,0,255),2)

        cv2.imshow('crop',display_image)
        cv2.imshow('white mask',white_mask)
        cv2.imshow('yellow mask',yellow_mask)
        cv2.waitKey(1)

    def publish_velocity(self):
        self.msg.linear.x=(self.wheel_radius*(self.v_r+self.v_l)/2.0)*0.1
        self.msg.angular.z=(self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation)*0.1

        self.cmd_pub.publish(self.msg)

        rospy.loginfo(
            "linear : %f       angular : %f",
            self.msg.linear.x,
            self.msg.angular.z
        )

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()