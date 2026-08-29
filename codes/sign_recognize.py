#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class TemplateMatch:

    def __init__(self):
        rospy.init_node('template_match',anonymous=True)

        self.bridge=CvBridge()
        self.template=cv2.imread('/home/sj/Desktop/TB/images/lturn.jpg',cv2.IMREAD_GRAYSCALE)
        self.template = cv2.resize(self.template,(100,100))
        if self.template is None:
            rospy.logerr("TEMPLATE LOAD FAIL")
            rospy.signal_shutdown("template load fail")
            return
       
        self.th,self.tw=self.template.shape
      
        rospy.Subscriber('/camera/color/image_raw',Image,self.image_callback,queue_size=1,buff_size=2**24)

    def image_callback(self,msg):
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        result=cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,max_loc=cv2.minMaxLoc(result)
   
        percent=max_val*100
        x,y=max_loc
   

        rospy.loginfo("MATCH: %.1f%%",percent)
    
        # template_view=cv2.cvtColor(self.template,cv2.COLOR_GRAY2BGR)
        # template_view=cv2.resize(template_view,(200,200))
     
        # camera_view=cv2.resize(image,(640,480))
     
        # cv2.imshow('TEMPLATE',template_view)
        # cv2.imshow('CAMERA MATCH',camera_view)
        # cv2.waitKey(1)

if __name__=='__main__':
    node=TemplateMatch()
    rospy.spin()
    # cv2.destroyAllWindows()