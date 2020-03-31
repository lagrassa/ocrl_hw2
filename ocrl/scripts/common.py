#!/usr/bin/env python

import numpy as np
import math

import rospy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point, PoseArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

x_lim = [-10, 10]
y_lim = [-10, 10]
theta_lim = [-np.pi, np.pi]
num_waypoints = 10
waypoint_tol = 0.2
waypoint_ang_tol = 5*np.pi/180

wheelbase = 0.335
max_steering_angle = 30*np.pi/180
min_turning_radius = wheelbase/np.tan(max_steering_angle)
max_acc = 4
max_dec = -4
max_vel = 10
max_ang_vel = max_vel/min_turning_radius
