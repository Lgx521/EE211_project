#!/usr/bin/env python3

import rclpy
import random
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration

class WaypointQuatNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_quat_navigator')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.waypoints = self.get_waypoints()
        self.current_goal_idx = 0
        self.init_sent = False
        self.goal_in_progress = False
        
        self.current_route = None

    def get_waypoints(self):
        return [
            [6.7132, -2.0496, -0.483441, 0.875377],
            [7.2471, -3.7436, -0.587796, 0.809009],
            [8.7445, -4.2035, 0.381371, 0.924422],
            [10.8847, -2.4878, -0.210088, 0.977683],
            [9.8938, -0.7154, 0.834495, 0.551015],
            [6.5807, -1.7804, 0.948298, 0.317381]
        ]
    
    def grasp_object(self):
        self.get_logger().info("="*50)
        self.get_logger().info("Grasping Signal Triggered.")
        self.get_logger().info("Grasping OK.")
        self.get_logger().info("="*50)
    
    def release_object(self):
        self.get_logger().info("="*50)
        self.get_logger().info("Put Down Signal Triggered.")
        self.get_logger().info("Put Down OK.")
        self.get_logger().info("="*50)
    
    def detect_stop_sign(self):
        # True if stop sign detected, else False
        stop_detected = False
        
        if stop_detected:
            self.get_logger().info("Detected STOP sign on route!")
            return True
        return False
    
    def check_traffic_light(self):
        light_color = "green"
        self.get_logger().info(f"Traffic light status: {light_color.upper()}")
        return light_color

    def publish_init_pose(self):
        if self.init_sent:
            return
        x, y, z, w = self.waypoints[0]
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w
        
        for _ in range(3):
            self.init_pose_pub.publish(msg)
            self.get_logger().info("Published /initialpose for localization...")
            rclpy.spin_once(self, timeout_sec=0.5)
        
        self.init_sent = True
        self.get_logger().info("Waiting 1s for AMCL localization to settle...")
        rclpy.spin_once(self, timeout_sec=1.0)

    def send_next_goal(self):
        if not self.init_sent:
            self.publish_init_pose()
        
        if self.current_goal_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints completed!')
            rclpy.shutdown()
            return
        
        if self.goal_in_progress:
            self.get_logger().warn("A goal is already in progress, skip sending new goal")
            return
        
        self.current_route = f"waypoint{self.current_goal_idx+1}"
        
        # 1. Waypoint1-3 : detect stop sign
        if 1 <= self.current_goal_idx + 1 <= 3:
            self.get_logger().info(f"Scanning for STOP sign on route to {self.current_route}...")
            if self.detect_stop_sign():
                self.get_logger().info("Stopping for 2 seconds due to STOP sign...")
                rclpy.spin_once(self, timeout_sec=2.0)
                self.get_logger().info("Resume navigation after STOP sign pause")
        
        # 2. Waypoint4-5 : check traffic light
        elif self.current_goal_idx + 1 == 5:
            self.get_logger().info(f"Checking traffic light on route to {self.current_route}...")
            while True:
                light_status = self.check_traffic_light()
                if light_status == "red":
                    self.get_logger().info("Red light - waiting for green...")
                    rclpy.spin_once(self, timeout_sec=1.0)
                else:
                    self.get_logger().info("Green light - proceeding!")
                    break
        
        x, y, z, w = self.waypoints[self.current_goal_idx]
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(f"Sending goal {self.current_goal_idx+1}/{len(self.waypoints)}: (x={x:.2f}, y={y:.2f}, z={z:.3f}, w={w:.3f})")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("NavigateToPose action server not available!")
            rclpy.shutdown()
            return
        
        self.goal_in_progress = True
        self._send_goal_future = self.action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Goal {self.current_goal_idx+1} rejected by server!")
            self.goal_in_progress = False
            self.current_goal_idx += 1
            self.send_next_goal()
            return

        self.get_logger().info(f"Goal {self.current_goal_idx+1} accepted by server, navigating...")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self.goal_in_progress = False
        current_goal_number = self.current_goal_idx + 1
        self.get_logger().info(f"Goal {current_goal_number} reached!")
        
        if current_goal_number == 4:
            self.grasp_object()
            rclpy.spin_once(self, timeout_sec=0.2)
        
        elif current_goal_number == 5:
            self.release_object()
            rclpy.spin_once(self, timeout_sec=0.2)
        
        self.current_goal_idx += 1
        self.send_next_goal()

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointQuatNavigator()
    navigator.send_next_goal()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()