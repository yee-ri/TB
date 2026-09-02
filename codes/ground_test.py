#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import tf2_ros
import tf.transformations as tft
from sensor_msgs.msg import CompressedImage,CameraInfo

class GroundTest():
    def __init__(self):
        rospy.init_node('ground_test',anonymous=True)
        self.image=None
        self.K=None
        self.D=None
        self.tf_buffer=tf2_ros.Buffer()
        self.tf_listener=tf2_ros.TransformListener(self.tf_buffer)
        self.image_sub=rospy.Subscriber('/camera/color/image_raw/compressed',CompressedImage,self.image_callback,queue_size=1)
        self.info_sub=rospy.Subscriber('/camera/color/camera_info',CameraInfo,self.info_callback,queue_size=1)
        cv2.namedWindow('camera')
        cv2.setMouseCallback('camera',self.mouse_callback)

    def info_callback(self,msg):
        self.K=np.array(msg.K,dtype=np.float64).reshape(3,3)
        self.D=np.array(msg.D,dtype=np.float64)

    def image_callback(self,msg):
        self.image=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
        if self.image is None:
            return
        display=self.image.copy()
        # cv2.putText(display,'CLICK GROUND',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow('camera',display)
        cv2.waitKey(1)

    def mouse_callback(self,event,u,v,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:
            return
        if self.K is None:
            rospy.logwarn("camera_info not received")
            return
        ground=self.pixel_to_ground(u,v)
        if ground is None:
            return
        x,y=ground
        rospy.loginfo("PIXEL (%d,%d) -> BASE x=%.3f cm, y=%.3f cm",u,v,x*100,y*100)

    def pixel_to_ground(self,u,v):
        try:
            transform=self.tf_buffer.lookup_transform('base_footprint','camera_color_optical_frame',rospy.Time(0),rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("TF ERROR: %s",str(e))
            return None

        pixel=np.array([[[float(u),float(v)]]],dtype=np.float64)
        normalized=cv2.undistortPoints(pixel,self.K,self.D)
        x_norm=normalized[0,0,0]
        y_norm=normalized[0,0,1]
        ray_camera=np.array([x_norm,y_norm,1.0])
        q=transform.transform.rotation
        rotation=tft.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
        ray_base=np.dot(rotation,ray_camera)
        t=transform.transform.translation
        camera_position=np.array([t.x,t.y,t.z])

        if ray_base[2]>=0:
            rospy.logwarn("CLICKED POINT DOES NOT HIT GROUND")
            return None
        scale=-camera_position[2]/ray_base[2]
        if scale<=0:
            rospy.logwarn("GROUND POINT IS BEHIND CAMERA")
            return None
        ground=camera_position+scale*ray_base
        return ground[0],ground[1]

if __name__=='__main__':
    node=GroundTest()
    rospy.spin()