#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,LaserScan


class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        
        # self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # self.joint_sub = rospy.Subscriber('/joint_states',JointState, self.joint_callback)
        self.image = rospy.Subscriber('/camera/image',Image,self.detect_lturn)
        self.scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.bridge = CvBridge()
        self.ignore_sign_until = rospy.Time(0)
        # self.lturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        # self.lturn_template = cv2.resize(self.lturn_template,(10,10))

        # self.rturn_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/rturn.jpg',cv2.IMREAD_GRAYSCALE)
        # self.rturn_template = cv2.resize(self.rturn_template,(10,10))

        # self.repair_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/construct.png',cv2.IMREAD_GRAYSCALE)
        # self.repair_template = cv2.resize(self.repair_template,(80,80))

        # self.noentry_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/parking.png',cv2.IMREAD_GRAYSCALE)
        # self.noentry_template = cv2.resize(self.noentry_template,(20,20))
        self.parking_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/parking.png',cv2.IMREAD_GRAYSCALE)
        self.parking_template = cv2.resize(self.parking_template,(20,20))

        # self.gate_template = cv2.imread('/home/sj/catkin_ws/src/turtlebot3_simulations/images/gate.jpg',cv2.IMREAD_GRAYSCALE)
        # self.gate_template = cv2.resize(self.gate_template,(10,10))
        self.msg = Twist()

        self.sign_detected = False
        self.sign_score = 0.0
        self.sign_box = None
        self.sign_count = 0

        self.template = self.parking_template 

    def detect_lturn(self,data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # if self.repair_template is None:
        #     print("template load fail")
        #     return

        template_h,template_w = self.template.shape
        image_h,image_w = gray.shape

        if template_h > image_h or template_w > image_w:
            return

        res = cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0]+template_w,top_left[1]+template_h)

        rospy.loginfo(" SIGN score: %.3f",max_val)

        if max_val > 0.72:
            cv2.rectangle(image,top_left,bottom_right,(0,0,255),2)
            rospy.loginfo(" SIGN DETECTED")

        cv2.imshow("sign_size",self.template )
        cv2.imshow("REPAIR",image) 
        cv2.waitKey(1)  

    def scan_callback(self,data):
        left_ranges = []
        front_ranges = []
        right_ranges = []

        for i in range(len(data.ranges)):
            distance = data.ranges[i]

            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = data.angle_min + i*data.angle_increment
            angle_deg = np.degrees(angle)

            if angle_deg > 180:
                angle_deg -= 360

            if 30 <= angle_deg <= 80:
                right_ranges.append(distance)

            elif -20 <= angle_deg <= 20:
                front_ranges.append(distance)

            elif -80 <= angle_deg <= -30:
                left_ranges.append(distance)

        self.left_distance = min(left_ranges) if len(left_ranges) > 0 else float('inf')
        self.front_distance = min(front_ranges) if len(front_ranges) > 0 else float('inf')
        self.right_distance = min(right_ranges) if len(right_ranges) > 0 else float('inf')

        rospy.loginfo( "LEFT: %.2f  FRONT: %.2f  RIGHT: %.2f",self.left_distance,self.front_distance, self.right_distance)

if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin() 