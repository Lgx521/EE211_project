#!/usr/bin/env python3
"""
🚫 Navigation terminated with status code: X
  - Nav2 NavigateToPose 标准状态码:
    0 = UNKNOWN          - 未知状态 (极少出现)
    1 = CANCELED         - 被取消 - 通常是收到停止指令/手动取消
    2 = ABORTED          - 被中止 - 导航过程中出错 (如避障失败/机器人卡住)
    3 = FAILED           - 失败 - 路径规划失败/目标点无可行路径
    4 = SUCCEEDED        - 成功 - 导航完成, 到达目标位置
    5 = RECALLED         - 已撤回 - 目标在执行前被服务器撤回
    6 = REJECTED         - 已拒绝 - 目标被服务器拒绝 (同显式rejected日志)
    7 = PREEMPTED        - 被抢占 - 新的导航目标替换了当前目标
    8 = EXECUTING        - 执行中 - 临时状态 (正常不会出现在最终结果)
    9 = WAITING          - 等待中 - 临时状态 (等待服务器资源)
    10= TIMED_OUT        - 超时 - 导航超出设定时间未完成
"""

import rclpy
import threading
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool
import time

class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')
        
        # 1. Configure fixed target pose (x, y, z(yaw), w(yaw)) - Modify these values as needed
        self.target_pose = [8.7445, -4.2035, 0.381371, 0.924422]  # Target pose quaternion
        
        # 2. Action client (navigate to pose)
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 3. Subscribe to stop_control signal (Bool: True=stop, False=resume)
        self.stop_control_sub = self.create_subscription(
            Bool,
            'traffic_light/stop_control',
            self.stop_control_callback,
            10
        )
        
        # 4. State variables
        self.stop_flag = False          # Stop flag
        self.goal_in_progress = False   # Whether a navigation goal is in execution
        self.current_goal_handle = None # Current goal handle (for cancellation)
        self.navigation_completed = False  # Flag for navigation completion
        self.wait_time_after_goal = 5.0  # Wait time (seconds) after reaching goal
        
    def stop_control_callback(self, msg):
        new_stop_flag = msg.data
        
        # Only perform actions on state changes
        if new_stop_flag != self.stop_flag:
            self.stop_flag = new_stop_flag
            
            if self.stop_flag:
                self.get_logger().info("🔴 Stop command received, pausing navigation!")
                # Cancel current navigation goal if in progress
                if self.current_goal_handle and self.goal_in_progress:
                    self.current_goal_handle.cancel_goal_async()
                    self.get_logger().info("Current navigation goal cancelled")
                    self.goal_in_progress = False
            else:
                self.get_logger().info("🟢 Resume command received, resuming navigation!")
                # Resume navigation
                if not self.goal_in_progress and not self.navigation_completed:
                    self.send_nav_goal()

    def send_nav_goal(self):
        # Return immediately if in stop state or navigation already completed
        if self.stop_flag or self.navigation_completed:
            self.get_logger().warn("Currently in stop state or navigation completed, skipping goal transmission")
            return
        
        # Return if goal is already in progress
        if self.goal_in_progress:
            self.get_logger().warn("Navigation goal already in progress, skipping duplicate transmission")
            return
        
        # Wait for navigation action server to be ready
        if not self.nav_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Navigation action server not available!")
            rclpy.shutdown()
            return
        
        # Construct target pose
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.target_pose[0]
        pose.pose.position.y = self.target_pose[1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = self.target_pose[2]
        pose.pose.orientation.w = self.target_pose[3]
        
        # Construct navigation goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        # Send goal (asynchronous)
        self.get_logger().info(f"Sending navigation goal: x={self.target_pose[0]:.2f}, y={self.target_pose[1]:.2f}")
        self.goal_in_progress = True
        send_future = self.nav_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Navigation goal rejected by server!")
                self.goal_in_progress = False
                return
            
            self.get_logger().info("Navigation goal accepted by server, starting navigation...")
            self.current_goal_handle = goal_handle
            
            # Register result callback
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.goal_result_callback)
        except Exception as e:
            self.get_logger().error(f"Failed to process goal response: {str(e)}")
            self.goal_in_progress = False

    def goal_result_callback(self, future):
        try:
            self.goal_in_progress = False
            self.current_goal_handle = None
            
            result = future.result()
            # Nav2 success status code: 4 (SUCCEEDED)
            if result.status == 4:
                self.get_logger().info("✅ Navigation to target pose completed!")
                self.navigation_completed = True
                
                # Start waiting thread before exiting
                self.get_logger().info(f"⌛ Waiting {self.wait_time_after_goal} seconds before exiting...")
                exit_thread = threading.Thread(target=self.exit_after_wait)
                exit_thread.daemon = True
                exit_thread.start()
            else:
                # Status code explanation: 1=cancelled, 2=aborted, 3=failed
                self.get_logger().info(f"🚫 Navigation terminated with status code: {result.status}")
                
                # Retry sending goal if termination wasn't caused by stop command and not completed
                if not self.stop_flag and not self.navigation_completed:
                    self.get_logger().info("Attempting to resend navigation goal...")
                    self.send_nav_goal()
        except Exception as e:
            self.get_logger().error(f"Failed to process result callback: {str(e)}")
    
    def exit_after_wait(self):
        time.sleep(self.wait_time_after_goal)
        
        self.get_logger().info("🔚 Waiting period completed, exiting program...")
        
        self.destroy_node()
        rclpy.shutdown()
        
        import sys
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleNavigator()
    
    navigator.send_nav_goal()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("User interruption, exiting program")
    finally:
        if rclpy.ok():
            navigator.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()