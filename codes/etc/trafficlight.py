import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class turtlebot() :
    def __init__(self):
        rospy.init_node('controller', anonymous=True)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image = rospy.Subscriber('/camera/color/image_raw',Image,self.img_callback)

        self.bridge = CvBridge()
        self.msg = Twist()

    def img_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data,"bgr8")
    
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv_frame, np.array([45, 100, 80]), np.array([85, 255, 255]))
 
        green_mask = cv2.erode(green_mask, None, iterations=1)
        green_mask = cv2.dilate(green_mask, None, iterations=2)
        green_pixel = cv2.countNonZero(green_mask)
        rospy.loginfo ("%.3f",green_pixel)

        if green_pixel >= 200:
            rospy.loginfo("DETECTED GREEN LIGHT")
            
        cv2.imshow('green mask', green_mask)
        cv2.imshow('output', image)
        cv2.waitKey(3)


if __name__ == '__main__':
    controller = turtlebot()
    rospy.spin()
