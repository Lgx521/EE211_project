#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Bool
import time

class WaypointQuatNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_quat_navigator')
        # Action client for NavigateToPose action
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Publisher for initial pose (AMCL localization)
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        # Subscriber for stop control topic (traffic light/stop signal)
        self.stop_control_sub = self.create_subscription(
            Bool,
            'traffic_light/stop_control',
            self.stop_control_callback,
            10
        )
        
        # 新增：发布cmd_vel控制机器人移动
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers for grasp/place success signals
        self.grasp_success_received = False  # MODIFIED: Separate grasp flag
        self.place_success_received = False  # MODIFIED: Separate place flag
        self.grasp_sub = self.create_subscription(
            Bool,
            '/grasp/success',
            self.grasp_success_callback,
            100  # MODIFIED: Larger queue to prevent missed signals
        )
        self.place_sub = self.create_subscription(
            Bool,
            '/place/success',
            self.place_success_callback,
            100  # MODIFIED: Larger queue to prevent missed signals
        )
        
        # List of waypoints (x, y, z, w quaternion)
        self.waypoints = self.get_waypoints()
        self.current_goal_idx = 0  # Index of current target waypoint (preserved during stop)
        self.init_sent = False
        self.goal_in_progress = False  # Whether a goal is actively being executed
        self.stop_flag = False  # True = pause navigation, False = normal operation
        self.current_goal_handle = None  # Current goal handle (for canceling)
        
        self.current_route = None
        # 新增：向前移动配置（针对SLAM保护）
        self.forward_move_distance = 0.3  # 向前移动距离（米）
        self.forward_move_speed = 0.1     # 向前移动速度（m/s）
        self.is_moving_forward = False    # 标记是否正在向前移动

    def get_waypoints(self):
        """Define navigation waypoints with x, y, z (orientation), w (orientation)"""
        return [
            [6.7132, -2.0496, -0.483441, 0.875377],
            [6.5974, -2.8870, -0.524038, 0.851695],
            [8.7445, -4.2035, 0.381371, 0.924422],
            [10.7641, -2.5770, -0.194092, 0.980983],
            [9.6205, -1.1276, 0.956089, 0.293078],
            [6.5915, -1.7414, 0.911765, 0.410712]
        ]
    
    # 新增：直接发布cmd_vel让机器人向前移动指定距离
    def move_forward_directly(self):
        """通过cmd_vel直接控制机器人向前移动（规避SLAM保护）"""
        if self.stop_flag or self.is_moving_forward:
            return False
        
        self.is_moving_forward = True
        self.get_logger().info(f"Start moving forward {self.forward_move_distance}m via cmd_vel (speed: {self.forward_move_speed}m/s)")
        
        # 计算需要移动的时间
        move_duration = self.forward_move_distance / self.forward_move_speed
        start_time = time.time()
        
        # 构造向前移动的速度指令
        twist = Twist()
        twist.linear.x = self.forward_move_speed
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        # 持续发布速度指令直到达到指定距离（此阶段不响应stop_control）
        while rclpy.ok():
            # 检查是否移动时间已到
            elapsed_time = time.time() - start_time
            if elapsed_time >= move_duration:
                break
            
            # 发布速度指令
            self.cmd_vel_pub.publish(twist)
            
            # 处理ROS回调，但忽略stop_flag
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # 发布停止指令
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.get_logger().info("Stop moving forward, published zero velocity")
        
        self.get_logger().info(f"Successfully moved forward {self.forward_move_distance}m")
        self.is_moving_forward = False
        return True
    
    def grasp_success_callback(self, msg):
        """Callback for grasp success signal"""
        if msg.data:  # MODIFIED: Only set flag on True (ignore False)
            self.grasp_success_received = True
            self.get_logger().info("Received grasp success signal (/grasp/success=True)")
    
    def place_success_callback(self, msg):
        """Callback for place success signal"""
        if msg.data:  # MODIFIED: Only set flag on True (ignore False)
            self.place_success_received = True
            self.get_logger().info("Received place success signal (/place/success=True)")
    
    def wait_for_grasp_success(self):
        """Wait until grasp success signal is received"""
        self.get_logger().info("Waiting for grasp success signal (/grasp/success=True)...")
        self.grasp_success_received = False  # MODIFIED: Use separate grasp flag
        
        # 等待阶段不响应stop_control
        while rclpy.ok() and not self.grasp_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.grasp_success_received:
                self.get_logger().debug("Still waiting for grasp success...")
        
        self.get_logger().info("Grasp success received - resuming navigation")
        return True
    
    def wait_for_place_success(self):
        """Wait until place success signal is received"""
        self.get_logger().info("Waiting for place success signal (/place/success=True)...")
        self.place_success_received = False  # MODIFIED: Use separate place flag
        
        # 等待阶段不响应stop_control
        while rclpy.ok() and not self.place_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.place_success_received:
                self.get_logger().debug("Still waiting for place success...")
        
        self.get_logger().info("Place success received - resuming navigation")
        return True
    
    def stop_control_callback(self, msg):
        """Callback function for stop control topic
        Args:
            msg (Bool): True = stop navigation, False = resume navigation
        """
        new_stop_flag = msg.data
        
        # 核心修改：仅在「nav2正常导航中（goal_in_progress=True）」时响应stop_control
        if not self.goal_in_progress:
            self.get_logger().debug(f"Ignore stop_control (not in nav2 navigation): {new_stop_flag}")
            return
        
        # Only act on state changes (avoid repeated actions)
        if new_stop_flag != self.stop_flag:
            self.stop_flag = new_stop_flag
            
            if self.stop_flag:
                self.get_logger().info("Received stop command (stop_control=True), pausing navigation!")
                # 仅取消nav2的当前导航目标（无其他操作）
                if self.current_goal_handle:
                    self.current_goal_handle.cancel_goal_async()
                    self.get_logger().info(f"Cancelled current goal: waypoint {self.current_goal_idx + 1}")
                    self.goal_in_progress = False  # Mark goal as not in progress (critical fix)
            else:
                self.get_logger().info("Received resume command (stop_control=False), resuming navigation!")
                # 仅恢复nav2导航（无其他操作）
                if not self.goal_in_progress and self.current_goal_idx < len(self.waypoints):
                    self.send_current_goal()

    def publish_init_pose(self):
        """Publish initial pose for AMCL localization"""
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
        
        # Publish initial pose multiple times to ensure reception
        for _ in range(3):
            self.init_pose_pub.publish(msg)
            self.get_logger().info("Published /initialpose for localization...")
            rclpy.spin_once(self, timeout_sec=1.0)
        
        self.init_sent = True
        self.get_logger().info("Waiting 5s for AMCL localization to settle...")
        rclpy.spin_once(self, timeout_sec=5.0)

    def send_current_goal(self):
        """Send the CURRENT waypoint goal (not next) to NavigateToPose action server"""
        # Return immediately if in stop state
        if self.stop_flag:
            self.get_logger().warn("Currently in stop state, skipping new goal transmission")
            return
        
        # Publish initial pose if not sent yet
        if not self.init_sent:
            self.publish_init_pose()
        
        # Check if all waypoints are completed
        if self.current_goal_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints completed!')
            rclpy.shutdown()
            return
        
        # Check if a goal is already in progress
        if self.goal_in_progress or self.is_moving_forward:
            self.get_logger().warn("A goal is already in progress, skip sending new goal")
            return
        
        self.current_route = f"waypoint{self.current_goal_idx + 1}"
        
        # Construct target pose for CURRENT waypoint
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

        self.get_logger().info(f"Sending goal {self.current_goal_idx + 1}/{len(self.waypoints)}: (x={x:.2f}, y={y:.2f}, z={z:.3f}, w={w:.3f})")
        # Wait for action server to become available
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("NavigateToPose action server not available!")
            rclpy.shutdown()
            return
        
        self.goal_in_progress = True
        # Send goal asynchronously
        self._send_goal_future = self.action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Callback for goal response from action server"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f"Goal {self.current_goal_idx + 1} rejected by server!")
                self.goal_in_progress = False
                return  # Do NOT increment index (critical fix)

            self.get_logger().info(f"Goal {self.current_goal_idx + 1} accepted by server, navigating...")
            # Save current goal handle for potential cancellation
            self.current_goal_handle = goal_handle
            # Register callback for result retrieval
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.result_callback)
        except Exception as e:
            self.get_logger().error(f"Goal response error: {str(e)}")
            self.goal_in_progress = False

    def result_callback(self, future):
        """Callback for navigation result (only increment index if goal completed successfully)"""
        try:
            # Reset navigation state
            self.goal_in_progress = False
            self.current_goal_handle = None
            
            # Only increment index if goal was completed (not cancelled)
            # Check if the goal was actually achieved (nav2 returns result code 4 for success)
            result = future.result()
            if result.status == 4:  # Nav2 SUCCESS status code
                current_goal_number = self.current_goal_idx + 1
                self.get_logger().info(f"Goal {current_goal_number} reached!")
                
                # 核心逻辑：第四、第五个点到达后直接发cmd_vel向前移动
                proceed_to_next = True
                if current_goal_number in [4, 5]:
                    self.get_logger().info(f"Waypoint {current_goal_number} reached, moving forward via cmd_vel (SLAM protection workaround)")
                    # 直接发布cmd_vel向前移动（此阶段不响应stop_control）
                    proceed_to_next = self.move_forward_directly()
                    
                    # 移动完成后执行原有等待逻辑
                    if proceed_to_next and not self.stop_flag:
                        # Waypoint 4: Wait for grasp success
                        if current_goal_number == 4:
                            proceed_to_next = self.wait_for_grasp_success()
                        # Waypoint 5: Wait for place success
                        elif current_goal_number == 5:
                            proceed_to_next = self.wait_for_place_success()
                
                # Move to next waypoint ONLY if current goal succeeded and we're clear to proceed
                if proceed_to_next and not self.stop_flag:
                    self.current_goal_idx += 1
                    self.send_current_goal()
            else:
                self.get_logger().info(f"Goal {self.current_goal_idx + 1} cancelled/interrupted - preserving index")
                
        except Exception as e:
            self.get_logger().error(f"Result callback error: {str(e)}")
            self.goal_in_progress = False

    def send_next_goal(self):
        """Legacy wrapper for initial start (uses send_current_goal)"""
        self.send_current_goal()

def main(args=None):
    """Main function to initialize and run the navigator node"""
    rclpy.init(args=args)
    navigator = WaypointQuatNavigator()
    
    # Start navigation with first waypoint
    navigator.send_next_goal()
    
    # Keep node spinning
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        # 新增：键盘中断时发布停止指令
        stop_twist = Twist()
        navigator.cmd_vel_pub.publish(stop_twist)
        pass
    
    # Cleanup
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()