#!/usr/bin/env python
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist,Point
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker

class turtlebot():
    def __init__(self):
        rospy.init_node('controller',anonymous=True)

        self.image=rospy.Subscriber('/camera/color/image_raw',Image,self.img_callback)
        self.scan=rospy.Subscriber('/livox/lidar',PointCloud2,self.scan_callback)

        self.marker_pub=rospy.Publisher('/lidar_sector_marker',Marker,queue_size=1)

        self.bridge=CvBridge()

        self.scan_img=np.zeros((500,500,3),dtype=np.uint8)

        self.cx_w=0
        self.cy_w=0
        self.cx_y=0
        self.cy_y=0

    def img_callback(self,data):
        image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.crop_img=image[160:,:]

        hsvFrame=cv2.cvtColor(self.crop_img,cv2.COLOR_BGR2HSV)

        white_lower=np.array([0,0,200])
        white_upper=np.array([179,50,255])
        white_mask=cv2.inRange(hsvFrame,white_lower,white_upper)

        yellow_lower=np.array([20,100,100])
        yellow_upper=np.array([50,255,255])
        yellow_mask=cv2.inRange(hsvFrame,yellow_lower,yellow_upper)

        M_w=cv2.moments(white_mask)
        M_y=cv2.moments(yellow_mask)

        if M_w["m00"]>0 or M_y["m00"]>0:
            if M_w["m00"]==0.0:
                self.cx_w=0
                self.cy_w=0
            else:
                self.cx_w=int(M_w["m10"]/M_w["m00"])
                self.cy_w=int(M_w["m01"]/M_w["m00"])

            if M_y["m00"]==0.0:
                self.cx_y=0
                self.cy_y=0
            else:
                self.cx_y=int(M_y["m10"]/M_y["m00"])
                self.cy_y=int(M_y["m01"]/M_y["m00"])

        cv2.circle(self.crop_img,(self.cx_w,self.cy_w),8,(255,255,255),-1)
        cv2.circle(self.crop_img,(self.cx_y,self.cy_y),8,(0,255,255),-1)

        # cv2.imshow("CROP",self.crop_img)
        cv2.imshow("camera",image)
        cv2.waitKey(3)

    def scan_callback(self,data):
        left_ranges=[]
        front_ranges=[]
        right_ranges=[]

        for point in point_cloud2.read_points(data,field_names=("x","y","z"),skip_nans=True):
            x,y,z=point

            distance=np.sqrt(x*x+y*y)

            if distance<=0:
                continue

            angle_deg=np.degrees(np.arctan2(y,x))

            if 30<=angle_deg<=75:
                left_ranges.append(distance)

            elif -30<=angle_deg<=30:
                front_ranges.append(distance)

            elif -75<=angle_deg<=-30:
                right_ranges.append(distance)

        self.left_distance=min(left_ranges) if left_ranges else float('inf')
        self.front_distance=min(front_ranges) if front_ranges else float('inf')
        self.right_distance=min(right_ranges) if right_ranges else float('inf')

        rospy.loginfo("LEFT: %.2f FRONT: %.2f RIGHT: %.2f",self.left_distance,self.front_distance,self.right_distance)

        self.publish_lidar_sector()

    def publish_lidar_sector(self):
        marker=Marker()
        marker.header.frame_id="base_footprint"
        marker.header.stamp=rospy.Time.now()
        marker.ns="lidar_sector"
        marker.id=0
        marker.type=Marker.LINE_LIST
        marker.action=Marker.ADD

        marker.pose.orientation.w=1.0
        marker.scale.x=0.02

        marker.color.r=0.0
        marker.color.g=1.0
        marker.color.b=0.0
        marker.color.a=1.0

        line_length=3.0

        for angle in [-80,-30,30,80]:
            rad=np.radians(angle)

            p1=Point()
            p1.x=0.0
            p1.y=0.0
            p1.z=0.0

            p2=Point()
            p2.x=line_length*np.cos(rad)
            p2.y=line_length*np.sin(rad)
            p2.z=0.0

            marker.points.append(p1)
            marker.points.append(p2)

        self.marker_pub.publish(marker)
if __name__=='__main__':
    controller=turtlebot()
    rospy.spin()