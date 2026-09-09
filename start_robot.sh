#!/bin/bash

roslaunch realsense2_camera rs_camera.launch &
sleep 2

roslaunch turtlebot3_bringup turtlebot3_core.launch &
sleep 2

roslaunch livox_ros_driver2 rviz_MID360.launch &
sleep 2

rosrun tf static_transform_publisher -0.033073 0 0.1439 0 0 0 base_footprint livox_frame 100 &
rosrun tf static_transform_publisher 0.057526051 0.009 0.083634463 0 0.2967 0 base_footprint camera_link 100 &

wait