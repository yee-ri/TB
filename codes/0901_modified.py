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


        self.lane_width = 0.26
        self.msg=Twist()

        self.v_l=0.0
        self.v_r=0.0

        self.cx_w=0
        self.cy_w=0
        self.cx_y=0
        self.cy_y=0

        self.wheel_radius=0.0475
        self.wheel_separation=0.148


        self.mode='lane'

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/lturn2.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/rturn2.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))

        self.max_val_l=0.0
        self.max_val_r=0.0

    def detect_sign(self,data):
        gray=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)

        res_l=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_l,_,_=cv2.minMaxLoc(res_l)

        res_r=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,self.max_val_r,_,_=cv2.minMaxLoc(res_r)

        if (self.max_val_l>self.max_val_r+0.04 and self.max_val_l>0.35) or self.max_val_l>0.4:
            return 'left'

        if (self.max_val_r>self.max_val_l+0.04 and self.max_val_r>0.35) or self.max_val_r>0.4:
            return 'right'

        return None

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


        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        yellow_full=cv2.inRange(hsv,np.array([20,100,100]),np.array([50,255,255]))

        white_full=cv2.inRange(hsv,np.array([0,0,200]),np.array([179,50,255]))

        blue_full=cv2.inRange(hsv ,np.array([90,80,50]),np.array([130,255,255]))

        color_mask=cv2.bitwise_or(yellow_full,white_full)
        color_mask=cv2.bitwise_or(color_mask,blue_full)

        filtered=cv2.bitwise_and(image,image,mask=color_mask)

        if self.mode=='lane':
            sign=self.detect_sign(filtered)

            if sign=='left':
                self.mode='turn_left'
                rospy.loginfo("@@@ TURN LEFT @@@")

            elif sign=='right':
                self.mode='turn_right'
                rospy.loginfo("@@@ TURN RIGHT @@@")

        self.crop_img=image[180:,:]

        hsvFrame=cv2.cvtColor(self.crop_img,cv2.COLOR_BGR2HSV)

        white_lower=np.array([0,0,200])
        white_upper=np.array([179,50,255])
        white_mask=cv2.inRange(hsvFrame,white_lower,white_upper)

        yellow_lower=np.array([20,100,100])
        yellow_upper=np.array([50,255,255])
        yellow_mask=cv2.inRange(hsvFrame,yellow_lower,yellow_upper)

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

        h,w,_=self.crop_img.shape
        center_x=w/2.0

        if self.mode=='lane':
            if M_w["m00"]>0 and M_y["m00"]>0:
                self.cx_w=int(M_w["m10"]/M_w["m00"])
                self.cy_w=int(M_w["m01"]/M_w["m00"])

                self.cx_y=int(M_y["m10"]/M_y["m00"])
                self.cy_y=int(M_y["m01"]/M_y["m00"])

                if self.cy_w>self.cy_y+20:
                    self.cen=self.cx_w-140
                    rospy.loginfo("LANE -> WHITE PRIORITY")

                elif self.cy_y>self.cy_w+20:
                    self.cen=self.cx_y+145
                    rospy.loginfo("LANE -> YELLOW PRIORITY")

                else:
                    self.cen=(self.cx_w+self.cx_y)/2.0
                    rospy.loginfo("LANE -> BOTH CENTER")

                self.err_x=self.cen-center_x

                linear=5.0
                angular=-float(self.err_x)/4.0
                wheel_distance=0.2

                self.v_l=linear-angular*wheel_distance*0.7
                self.v_r=linear+angular*wheel_distance*0.7

                self.publish_velocity()

            elif M_w["m00"]>0 and M_y["m00"]<=0:
                self.cx_w=int(M_w["m10"]/M_w["m00"])
                self.cy_w=int(M_w["m01"]/M_w["m00"])

                self.cen=self.cx_w-150
                self.err_x=self.cen-center_x

                linear=5.2
                angular=-float(self.err_x)/5.0
                wheel_distance=0.2

                self.v_l=linear-angular*wheel_distance*0.9
                self.v_r=linear+angular*wheel_distance*0.9

                self.publish_velocity()

            elif M_w["m00"]<=0 and M_y["m00"]>0:
                self.cx_y=int(M_y["m10"]/M_y["m00"])
                self.cy_y=int(M_y["m01"]/M_y["m00"])

                self.cen=self.cx_y+135
                self.err_x=self.cen-center_x

                linear=4.5
                angular=-float(self.err_x)/5.0
                wheel_distance=0.2

                self.v_l=linear-angular*wheel_distance*0.9
                self.v_r=linear+angular*wheel_distance*0.9

                self.publish_velocity()

            else:
                self.stop_robot()

        elif self.mode=='turn_left':
            if M_y["m00"]>0:
                self.cx_y=int(M_y["m10"]/M_y["m00"])
                self.cy_y=int(M_y["m01"]/M_y["m00"])

                self.cen=self.cx_y+85
                self.err_x=self.cen-center_x

                linear=4.5
                angular=-float(self.err_x)/5.0
                wheel_distance=0.2

                self.v_l=linear-angular*wheel_distance*0.9
                self.v_r=linear+angular*wheel_distance*0.9

                self.publish_velocity()

            else:
                self.stop_robot()

        elif self.mode=='turn_right':
            if M_w["m00"]>0:
                self.cx_w=int(M_w["m10"]/M_w["m00"])
                self.cy_w=int(M_w["m01"]/M_w["m00"])

                self.cen=self.cx_w-90
                self.err_x=self.cen-center_x

                linear=5.2
                angular=-float(self.err_x)/5.0
                wheel_distance=0.2

                self.v_l=linear-angular*wheel_distance*0.9
                self.v_r=linear+angular*wheel_distance*0.9

                self.publish_velocity()

            else:
                self.stop_robot()

        cv2.imshow('COLOR FILTER',filtered)
        cv2.imshow('WHITE MASK',white_mask)
        cv2.imshow('YELLOW MASK',yellow_mask)
        cv2.waitKey(1)

    def stop_robot(self):
        self.v_l=0.0
        self.v_r=0.0
        self.publish_velocity()


    def publish_velocity(self):
        self.msg.linear.x=(self.wheel_radius*(self.v_r+self.v_l)/2.0)*0.1
        self.msg.angular.z=(self.wheel_radius*(self.v_r-self.v_l)/self.wheel_separation)*0.1

        self.cmd_pub.publish(self.msg)

        rospy.loginfo("linear : %f     angular : %f","mode : %s linear : %f angular : %f",self.mode,self.msg.linear.x,self.msg.angular.z)

if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()