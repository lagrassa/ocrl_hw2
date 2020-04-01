#!/usr/bin/env python

from common import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Twist, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from angles import *
from planner import plan

import tf

def waypointCallback(msg):
  global waypoints
  for i in range(len(msg.poses)):
    waypoints[i, 0] = msg.poses[i].position.x
    waypoints[i, 1] = msg.poses[i].position.y
    waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]

def vehicleStateCallback(msg):
  global rear_axle_center, rear_axle_theta, rear_axle_velocity
  rear_axle_center.position.x = msg.pose.pose.position.x
  rear_axle_center.position.y = msg.pose.pose.position.y
  rear_axle_center.orientation = msg.pose.pose.orientation

  rear_axle_theta = euler_from_quaternion(
    [rear_axle_center.orientation.x, rear_axle_center.orientation.y, rear_axle_center.orientation.z,
     rear_axle_center.orientation.w])[2]

  rear_axle_velocity.linear = msg.twist.twist.linear
  rear_axle_velocity.angular = msg.twist.twist.angular

def go_to_waypoint(waypoint):
  print waypoint
  print("setting new goal")
  navigation_msg = PoseStamped()
  navigation_msg.header.frame_id = "base_link"
  navigation_msg.header.stamp = rospy.Time.now()
  navigation_msg.pose.position.x = waypoint[0]
  navigation_msg.pose.position.y = waypoint[1]
  q = quaternion_from_euler(0,0,waypoint[2])
  navigation_msg.pose.orientation = Quaternion(*q)
  global nav_goal_pub
  nav_goal_pub.publish(navigation_msg)

  global rear_axle_center, rear_axle_theta, rear_axle_velocity, cmd_pub
  rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)
  dx = waypoint[0] - rear_axle_center.position.x
  dy = waypoint[1] - rear_axle_center.position.y
  heading = euler_from_quaternion(rear_axle_center.orientation)[2]
  s0 = np.array([rear_axle_center.position.x, rear_axle_center.position.y, heading ])
  nc = 20
  v0 = 0
  fc, us, states = plan(s0, sf, v0, nc)
  rate = rospy.Rate(fc)
  target_distance = math.sqrt(dx*dx + dy*dy)
  i = 0
  while target_distance > waypoint_tol:
    cmd =  AckermannDriveStamped()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "base_link"
    cmd.drive.steering_angle =us[i][1] 
    cmd.drive.speed = us[i][0]
    cmd_pub.publish(cmd)
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)
    dx = waypoint[0] - rear_axle_center.position.x
    dy = waypoint[1] - rear_axle_center.position.y
    target_distance = math.sqrt(dx * dx + dy * dy)
    i+=1 
    rate.sleep() 


if __name__ == '__main__':

  rospy.init_node('pure_pursuit')
  cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)
  nav_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

  waypoints = np.zeros((num_waypoints, 3))
  rospy.Subscriber("/ackermann_vehicle/waypoints",
                   PoseArray,
                   waypointCallback)
  rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)


  rear_axle_center = Pose()
  rear_axle_velocity = Twist()
  rospy.Subscriber("/ackermann_vehicle/ground_truth/state",
                   Odometry, vehicleStateCallback)
  rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)

  for w in waypoints:
    go_to_waypoint(w)




