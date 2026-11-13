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

        self.lturn_template=cv2.imread('/home/sj/Desktop/TB/images/left.png',cv2.IMREAD_GRAYSCALE)
        self.lturn_template=cv2.resize(self.lturn_template,(100,100))

        self.rturn_template=cv2.imread('/home/sj/Desktop/TB/images/right.png',cv2.IMREAD_GRAYSCALE)
        self.rturn_template=cv2.resize(self.rturn_template,(100,100))
        self.depth_point = None

        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)


    def depth_callback(self, data):
        depth_image = self.bridge.imgmsg_to_cv2( data, desired_encoding='passthrough')
        height, width = depth_image.shape[:2] 
        rospy.loginfo("%f , %df",height,width)
        # cx = width // 2 
        # cy = height // 2 -170
        cx_r = width // 2 + 290
        cy_r= height // 2 + 30
        self.depth_point = (cx_r, cy_r, width, height)

        # self.depth_point = (cx, cy, width, height)

        depth = float(depth_image[cy_r, cx_r])
        if np.isfinite(depth) and depth > 0:

            depth_m = depth / 1000.0 if data.encoding == '16UC1' else depth
            # rospy.loginfo("CENTER DEPTH: %.3f m", depth_m)

    def image_callback(self,msg):
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        height,width=image.shape[:2]

        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 218]), np.array([50, 255, 255]))

        white_mask=cv2.inRange(hsv,np.array([0,0,210]),np.array([179, 55, 255]))

        red_mask=cv2.inRange( hsv, np.array([0, 180, 107]), np.array([5, 255, 194]))

        green_mask = cv2.inRange(hsv, np.array([45, 100, 80]), np.array([85, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([96, 255, 9]), np.array([108, 255, 21]))

        width = blue_mask.shape[1]
        left_blue_area = cv2.countNonZero(blue_mask[:, :width // 2])
        right_blue_area = cv2.countNonZero(blue_mask[:, width // 2:])
        blue_area = cv2.countNonZero(blue_mask)
        # rospy.loginfo("BLUE: TOTAL=%d LEFT=%d RIGHT=%d",
        #               blue_area, left_blue_area, right_blue_area)

        mask=cv2.bitwise_or(yellow_mask,white_mask)

        mask=cv2.bitwise_or(mask,red_mask)

        mask=cv2.bitwise_or(mask,green_mask)
        red_area=cv2.countNonZero(red_mask)  
        green_area = cv2.countNonZero(green_mask)
        yellow_area = cv2.countNonZero(yellow_mask)
        blue_area = cv2.countNonZero(blue_mask)

        filtered=cv2.bitwise_and(image,image,mask=mask)

        display = image.copy()
        filtered_display = filtered.copy()
        rospy.loginfo("BLUE: %f",blue_area)
        point = self.depth_point
        if point is not None:
            cx, cy, depth_width, depth_height = point
            x = min(width - 1, int(cx * width / depth_width))
            y = min(height - 1, int(cy * height / depth_height))

            for view in (display, filtered_display):
                cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    view, 'DEPTH', (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )

        cv2.imshow('COLOR', display)
        cv2.imshow('COLOR FILTER', filtered_display)
        cv2.waitKey(1)
        gray=cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)

        left_result=cv2.matchTemplate(gray,self.lturn_template,cv2.TM_CCOEFF_NORMED)
        _,lmax_val,_,lmax_loc=cv2.minMaxLoc(left_result)

        left_percent=lmax_val*100

        right_result=cv2.matchTemplate(gray,self.rturn_template,cv2.TM_CCOEFF_NORMED)
        _,rmax_val,_,rmax_loc=cv2.minMaxLoc(right_result)

        right_percent=rmax_val*100

        # rospy.loginfo("LEFT sign: %.1f%% RIGHT sign: %.1f%%",left_percent,right_percent)
        # rospy.loginfo("GREEN is %.3f",green_area)


if __name__=='__main__':
    node=TemplateMatch()
    rospy.spin()
    cv2.destroyAllWindows()