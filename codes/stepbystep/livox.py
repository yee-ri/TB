#!/usr/bin/env python3
"""ROS Noetic Livox distance checker. No motion commands are published.
Run: python3 lidar_check.py
Without a GUI: python3 lidar_check.py _show:=false
Distances are coordinates from base_footprint, not clearance from the body.
Green: front; blue: left; orange: right; gray: height-filtered points outside ROI.
RViz: Fixed Frame=base_footprint, add PointCloud2 /lidar_check/filtered.
Requires rospy, numpy, OpenCV, tf2_ros, tf2_sensor_msgs.
"""
import math
import threading
import time
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class LidarCheck:
    def __init__(self):
        rospy.init_node('lidar_check')
        self.topic=rospy.get_param('~topic','/livox/lidar')
        self.frame=rospy.get_param('~frame','base_footprint')
        self.show=rospy.get_param('~show',True)
        self.z_min=float(rospy.get_param('~z_min',0.03))
        self.z_max=float(rospy.get_param('~z_max',0.30))
        self.front_min=float(rospy.get_param('~front_min',0.15))
        self.front_max=float(rospy.get_param('~front_max',3.0))
        self.safe_width=float(rospy.get_param('~robot_half_width',0.10))+float(rospy.get_param('~safety_margin',0.05))
        self.side_x_min=float(rospy.get_param('~side_x_min',-0.15))
        self.side_x_max=float(rospy.get_param('~side_x_max',0.80))
        self.side_limit=float(rospy.get_param('~side_limit',1.50))
        self.percentile=float(rospy.get_param('~percentile',10))
        self.min_points=int(rospy.get_param('~min_points',3))
        if not (self.z_min<self.z_max and 0<self.safe_width<self.side_limit and self.front_min<self.front_max and self.side_x_min<self.side_x_max and 0<=self.percentile<=100 and self.min_points>=1):
            raise ValueError('Invalid ROI/percentile/min_points parameters')
        self.buffer=tf2_ros.Buffer()
        self.listener=tf2_ros.TransformListener(self.buffer)
        self.lock=threading.Lock()
        self.points=[]
        self.stats={name:None for name in ('LEFT','FRONT','RIGHT')}
        self.last_ok=None
        self.status='WAITING FOR LIDAR'
        self.pub=rospy.Publisher('/lidar_check/filtered',PointCloud2,queue_size=1)
        self.sub=rospy.Subscriber(self.topic,PointCloud2,self.callback,queue_size=1,buff_size=2**24)
        rospy.loginfo('Topic=%s Frame=%s Z=%.2f..%.2f Front X=%.2f..%.2f |Y|<%.2f Side X=%.2f..%.2f |Y|=%.2f..%.2f',self.topic,self.frame,self.z_min,self.z_max,self.front_min,self.front_max,self.safe_width,self.side_x_min,self.side_x_max,self.safe_width,self.side_limit)

    def callback(self,msg):
        try:
            if msg.header.frame_id==self.frame:
                cloud=msg
            else:
                transform=self.buffer.lookup_transform(self.frame,msg.header.frame_id,msg.header.stamp,rospy.Duration(0.1))
                cloud=do_transform_cloud(msg,transform)
            points=[]
            groups={name:[] for name in ('LEFT','FRONT','RIGHT')}
            for x,y,z in point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True):
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and self.z_min<z<self.z_max):continue
                region=None
                if self.front_min<x<self.front_max and abs(y)<self.safe_width:
                    region='FRONT'
                    groups[region].append(x)
                elif self.side_x_min<x<self.side_x_max:
                    if self.safe_width<=y<self.side_limit:
                        region='LEFT'
                        groups[region].append(y)
                    elif -self.side_limit<y<=-self.safe_width:
                        region='RIGHT'
                        groups[region].append(-y)
                points.append((x,y,z,region))
            stats={name:(len(values),float(np.min(values)),float(np.percentile(values,self.percentile))) if values else (0,None,None) for name,values in groups.items()}
            with self.lock:
                self.points=points
                self.stats=stats
                self.last_ok=time.monotonic()
                self.status='OK'
            header=Header(stamp=msg.header.stamp,frame_id=self.frame)
            self.pub.publish(point_cloud2.create_cloud_xyz32(header,[(x,y,z) for x,y,z,region in points]))
            lines=[]
            for name,(count,minimum,distance) in stats.items():
                value='NO DATA' if count<self.min_points else '%.2f m'%distance
                lines.append('%s: %s (n=%d min=%s)'%(name,value,count,'--' if minimum is None else '%.2f'%minimum))
            rospy.loginfo_throttle(0.5,' | '.join(lines))
        except Exception as error:
            with self.lock:self.status='TF/CLOUD ERROR: '+str(error)
            rospy.logwarn_throttle(1.0,'Lidar check: %s',error)

    def run(self):
        if not self.show:
            rate=rospy.Rate(10)
            while not rospy.is_shutdown():
                with self.lock:last_ok=self.last_ok
                if last_ok is None or time.monotonic()-last_ok>1.0:rospy.logwarn_throttle(1.0,'NO FRESH VALID CLOUD')
                rate.sleep()
            return
        import cv2
        scale=150
        origin=(400,650)
        def pixel(x,y):return (int(origin[0]-y*scale),int(origin[1]-x*scale))
        colors={'FRONT':(0,220,0),'LEFT':(255,150,0),'RIGHT':(0,150,255),None:(80,80,80)}
        cv2.namedWindow('Lidar check',cv2.WINDOW_NORMAL)
        try:
            while not rospy.is_shutdown():
                with self.lock:
                    points=self.points
                    stats=self.stats.copy()
                    last_ok=self.last_ok
                    status=self.status
                stale=last_ok is None or time.monotonic()-last_ok>1.0
                image=np.zeros((800,800,3),dtype=np.uint8)
                for distance in (0.5,1.0,2.0,3.0):cv2.circle(image,origin,int(distance*scale),(40,40,40),1)
                for x1,x2,y1,y2,name in ((self.front_min,self.front_max,-self.safe_width,self.safe_width,'FRONT'),(self.side_x_min,self.side_x_max,self.safe_width,self.side_limit,'LEFT'),(self.side_x_min,self.side_x_max,-self.side_limit,-self.safe_width,'RIGHT')):
                    cv2.rectangle(image,pixel(x2,y2),pixel(x1,y1),colors[name],1)
                if not stale and status=='OK':
                    for x,y,z,region in points:
                        u,v=pixel(x,y)
                        if 0<=u<800 and 0<=v<800:image[v,u]=colors[region]
                cv2.arrowedLine(image,origin,pixel(0.35,0),(255,255,255),2)
                cv2.putText(image,'BASE',(origin[0]+12,origin[1]),0,0.5,(255,255,255),1)
                for index,name in enumerate(('LEFT','FRONT','RIGHT')):
                    item=stats[name]
                    count,minimum,distance=item if item is not None else (0,None,None)
                    value='NO DATA' if stale or status!='OK' or count<self.min_points else '%.2f m'%distance
                    label='%s: %s  n=%d  min=%s'%(name,value,count,'--' if minimum is None else '%.2f'%minimum)
                    cv2.putText(image,label,(15,30+index*28),0,0.6,colors[name],1)
                cv2.putText(image,('STALE / NO DATA' if stale else status)[:95],(15,125),0,0.5,(0,0,255) if stale or status!='OK' else (255,255,255),1)
                cv2.putText(image,'TOP VIEW: front=up, left=left | q: quit',(15,775),0,0.55,(255,255,255),1)
                cv2.imshow('Lidar check',image)
                if cv2.waitKey(30)&0xFF==ord('q'):break
        finally:
            cv2.destroyAllWindows()
            rospy.signal_shutdown('Viewer closed')


if __name__=='__main__':
    try:LidarCheck().run()
    except rospy.ROSInterruptException:pass
