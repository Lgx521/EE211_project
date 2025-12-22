#!/usr/bin/env python3
"""
Nav2 Goal Controller for TurtleBot4 (ROS2 Humble)
True Persistent Stop/Resume Logic:
- Hard stop when stop_control = True (no auto-retry, no stuttering)
- Resume only when stop_control = False (explicit resume)
- No automatic navigation in stop mode
- Persistent hold state until resume command
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool

# Goal status constants (ROS2 Humble compatible)
GOAL_STATUS_SUCCEEDED = 4
GOAL_STATUS_CANCELED = 2
GOAL_STATUS_ABORTED = 5
GOAL_STATUS_EXECUTING = 1
GOAL_STATUS_ACCEPTED = 0

class Nav2GoalController(Node):
    def __init__(self):
        super().__init__('nav2_goal_controller')
        
        # Core state variables (persistent stop/resume)
        self.stop_flag = False                  # Persistent stop state
        self.current_goal_handle = None         # Active goal handle
        self.saved_goal_pose = None             # Persistent goal storage
        self.is_navigating = False              # Navigation state
        self.navigation_held = False            # Flag for held navigation
        
        # 1. Subscribe to global plan (no auto-execute in stop mode)
        self.global_plan_sub = self.create_subscription(
            Path,
            '/received_global_plan',
            self.global_plan_callback,
            10
        )
        
        # 2. Subscribe to RViz goal input
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # 3. Subscribe to stop/resume control (persistent state)
        self.stop_sub = self.create_subscription(
            Bool,
            'traffic_light/stop_control',
            self.stop_control_callback,
            10
        )
        
        # 4. Create action client
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Wait for Nav2 action server
        self.get_logger().info("Waiting for Nav2 Action Server (/navigate_to_pose)...")
        if not self.action_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("Nav2 Action Server not available! Please start Nav2 first")
            rclpy.shutdown()
            return
        
        self.get_logger().info("=== Persistent Stop/Resume Controller Initialized ===")
        self.get_logger().info("Logic:")
        self.get_logger().info("  - STOP (True) = Hard stop, no auto-retry, hold position")
        self.get_logger().info("  - RESUME (False) = Continue to saved goal")
        self.get_logger().info("  - Will stay stopped until explicit resume command")

    def global_plan_callback(self, msg: Path):
        """Update goal but DO NOT auto-execute in stop mode"""
        if len(msg.poses) == 0:
            return
        
        # Update saved goal (always preserve latest goal)
        goal_pose_stamped = msg.poses[-1]
        self.saved_goal_pose = goal_pose_stamped
        
        # self.get_logger().info("\nUpdated goal from global plan:")
        # self.get_logger().info(f"   Position: x={goal_pose_stamped.pose.position.x:.2f}, y={goal_pose_stamped.pose.position.y:.2f}")
        # self.get_logger().info(f"   Frame ID: {msg.header.frame_id}")
        
        # ONLY auto-execute if NOT in stop mode and NOT navigating
        if not self.stop_flag and not self.is_navigating and not self.navigation_held:
            self.get_logger().info("Auto-executing new goal (stop mode OFF)")
            self.send_nav_goal(goal_pose_stamped)

    def goal_pose_callback(self, msg: PoseStamped):
        """Handle RViz goal - store only (no auto-execute in stop mode)"""
        self.get_logger().info(f"\nReceived new goal from RViz: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")
        self.saved_goal_pose = msg
        
        # ONLY execute immediately if NOT in stop mode
        if not self.stop_flag and not self.is_navigating:
            self.get_logger().info("Executing new RViz goal (stop mode OFF)")
            self.send_nav_goal(msg)
        else:
            self.get_logger().info("Goal saved - will execute when resume command received")

    def stop_control_callback(self, msg: Bool):
        """True Persistent Stop/Resume Logic (core fix)"""
        new_stop_flag = msg.data
        
        # Ignore duplicate commands (prevent multiple triggers)
        if new_stop_flag == self.stop_flag:
            return
        
        # Update persistent stop state
        self.stop_flag = new_stop_flag
        
        if self.stop_flag:
            self.get_logger().info("\n=====================================")
            self.get_logger().info("=== HARD STOP COMMAND RECEIVED ===")
            self.get_logger().info("=== WILL STAY STOPPED UNTIL RESUME ===")
            self.get_logger().info("=====================================")
            
            # 1. Cancel ALL active navigation immediately
            if self.current_goal_handle and self.is_navigating:
                self.get_logger().info("Canceling active navigation...")
                self.current_goal_handle.cancel_goal_async()
            
            # 2. Force state reset (critical for stop persistence)
            self.is_navigating = False
            self.navigation_held = True
            self.current_goal_handle = None
            
            # 3. Confirm stop state
            if self.saved_goal_pose:
                self.get_logger().info(f"Goal preserved: x={self.saved_goal_pose.pose.position.x:.2f}, y={self.saved_goal_pose.pose.position.y:.2f}")
            self.get_logger().info("✅ System in HARD STOP state - no auto-retry")
                
        else:
            self.get_logger().info("\n=====================================")
            self.get_logger().info("=== RESUME COMMAND RECEIVED ===")
            self.get_logger().info("=====================================")
            
            # 1. Reset hold state
            self.navigation_held = False
            
            # 2. Resume to saved goal (if exists)
            if self.saved_goal_pose:
                self.get_logger().info(f"Resuming navigation to goal: x={self.saved_goal_pose.pose.position.x:.2f}, y={self.saved_goal_pose.pose.position.y:.2f}")
                self.send_nav_goal(self.saved_goal_pose)
            else:
                self.get_logger().error("❌ No saved goal! Send new goal via RViz first")
            self.get_logger().info("✅ System in RESUME state - normal navigation")

    def send_nav_goal(self, pose: PoseStamped):
        """Send goal ONLY if NOT in stop mode (prevent stuttering)"""
        # HARD CHECK: Do NOT send goal if in stop mode
        if self.stop_flag:
            self.get_logger().warn("⛔ STOP MODE ACTIVE - REJECTING goal send")
            self.navigation_held = True
            return
        
        # Cancel existing navigation (clean start)
        if self.current_goal_handle and self.is_navigating:
            self.get_logger().info("Stopping existing navigation for new goal...")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        
        # Build goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        self.is_navigating = True
        self.navigation_held = False
        self.get_logger().info(f"\n=== Sending Goal to Nav2 ===")
        self.get_logger().info(f"Position: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}")
        self.get_logger().info(f"Stop Mode: {self.stop_flag} | Held: {self.navigation_held}")
        
        # Send goal with error protection
        try:
            send_future = self.action_client.send_goal_async(goal_msg)
            send_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.get_logger().error(f"Failed to send goal: {str(e)}")
            self.is_navigating = False
            self.navigation_held = True

    def goal_response_callback(self, future):
        """Handle goal response (stop mode check)"""
        try:
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error("❌ Goal rejected by Nav2!")
                self.is_navigating = False
                self.navigation_held = True
                return
            
            # Double-check stop mode before starting
            if self.stop_flag:
                self.get_logger().warn("⛔ STOP mode activated - canceling accepted goal")
                goal_handle.cancel_goal_async()
                self.is_navigating = False
                self.navigation_held = True
                return
            
            self.get_logger().info("✅ Goal accepted by Nav2 - starting navigation")
            self.current_goal_handle = goal_handle
            
            # Listen for result
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.goal_result_callback)
            
        except Exception as e:
            self.get_logger().error(f"Goal response error: {str(e)}")
            self.is_navigating = False
            self.navigation_held = True

    def goal_result_callback(self, future):
        """Result handling with stop mode persistence"""
        try:
            result = future.result()
            status_code = result.status
            
            # Always reset navigation state
            self.is_navigating = False
            self.current_goal_handle = None
            
            # Status handling (stop mode aware)
            if status_code == GOAL_STATUS_SUCCEEDED:
                self.get_logger().info("\n🎉 Navigation Completed Successfully")
                self.get_logger().info(f"Reached goal: x={self.saved_goal_pose.pose.position.x:.2f}, y={self.saved_goal_pose.pose.position.y:.2f}")
                self.navigation_held = False  # Ready for new goals
                
            elif status_code == GOAL_STATUS_CANCELED or status_code == GOAL_STATUS_ABORTED:
                self.get_logger().info("\n🛑 Navigation Stopped (as requested)")
                # Maintain hold state if in stop mode
                if self.stop_flag:
                    self.navigation_held = True
                    self.get_logger().info("🔒 System remains in STOP state - no auto-retry")
                else:
                    self.navigation_held = False
                
            else:
                self.get_logger().warn(f"\n⚠️ Navigation Failed (Code: {status_code})")
                # ONLY retry if NOT in stop mode (critical fix)
                if not self.stop_flag and self.saved_goal_pose and not self.navigation_held:
                    self.get_logger().info("🔄 Auto-retrying navigation (stop mode OFF)")
                    self.send_nav_goal(self.saved_goal_pose)
                else:
                    self.get_logger().info("⛔ Stop mode active - skipping auto-retry")
                    
        except Exception as e:
            self.get_logger().error(f"Result processing error: {str(e)}")
            self.is_navigating = False
            self.navigation_held = self.stop_flag  # Match hold state to stop mode

def main(args=None):
    """Main function with persistent stop/resume logic"""
    rclpy.init(args=args)
    node = Nav2GoalController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n🛑 Program terminated by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()