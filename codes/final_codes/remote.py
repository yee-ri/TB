#!/usr/bin/env python
import rospy
import sys
import termios
import tty
from geometry_msgs.msg import Twist

def get_key():
    fd=sys.stdin.fileno()
    old=termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        key=sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old)
    return key

rospy.init_node('wasd_teleop')
pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)

linear=0.1
angular=0.4

print("W: forward  S: back  A: left  D: right  X: stop  Q: quit")

while not rospy.is_shutdown():
    key=get_key()
    msg=Twist()

    if key=='w':
        msg.linear.x=linear
    elif key=='s':
        msg.linear.x=-linear
    elif key=='a':
        msg.angular.z=angular
    elif key=='d':
        msg.angular.z=-angular
    elif key=='x':
        pass
    elif key=='q':
        pub.publish(Twist())
        break
    else:
        continue

    pub.publish(msg)