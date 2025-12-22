#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Bool
from rclpy.executors import MultiThreadedExecutor
import threading

class WaypointQuatNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_quat_navigator')
        # Action client for NavigateToPose action
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Publisher for initial pose (AMCL localization)
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        
        # Publishers for grasp/place start signals
        self.grasp_start_pub = self.create_publisher(Bool, '/grasp/start', 10)
        self.place_start_pub = self.create_publisher(Bool, '/place/start', 10)
        
        # Subscribers for grasp/place success signals
        self.grasp_success_received = False
        self.place_success_received = False
        self.grasp_sub = self.create_subscription(
            Bool,
            '/grasp/success',
            self.grasp_success_callback,
            100
        )
        self.place_sub = self.create_subscription(
            Bool,
            '/place/success',
            self.place_success_callback,
            100
        )
        
        # 交通灯停止控制相关 - 仅控制导航
        self.traffic_light_stop = False  # 交通灯停止标志
        self.traffic_light_prev_state = False  # 上一状态，用于边沿检测
        self.traffic_light_sub = self.create_subscription(
            Bool,
            '/traffic_light/stop_control',
            self.traffic_light_callback,
            10
        )
        self.stop_lock = threading.Lock()  # 线程锁，保护状态变量
        
        # List of waypoints (x, y, z, w quaternion)
        self.waypoints = self.get_waypoints()
        self.current_goal_idx = 0
        self.init_sent = False
        self.goal_in_progress = False
        self.current_goal_handle = None
        self.current_route = None

    def get_waypoints(self):
        """Define navigation waypoints with x, y, z (orientation), w (orientation)"""
        return [
            [6.7132, -2.0496, -0.483441, 0.875377],
            [6.7167, -3.2733, -0.351566, 0.936163],
            [8.7445, -4.2035, 0.381371, 0.924422],
            [11.1434, -2.4972, -0.212324, 0.954916],
            [9.0319, -0.5364, -0.998852, 0.047912],
            [6.5915, -1.7414, 0.911765, 0.410712]
        ]
    
    def grasp_success_callback(self, msg):
        """Callback for grasp success signal"""
        if msg.data:
            self.grasp_success_received = True
            self.get_logger().info("Received grasp success signal (/grasp/success=True)")
    
    def place_success_callback(self, msg):
        """Callback for place success signal"""
        if msg.data:
            self.place_success_received = True
            self.get_logger().info("Received place success signal (/place/success=True)")
    
    def traffic_light_callback(self, msg):
        """Callback for traffic light stop control signal (边沿检测) - 仅控制导航"""
        with self.stop_lock:
            # 检测上升沿 (False -> True) - 需要停止导航
            if msg.data and not self.traffic_light_prev_state:
                self.traffic_light_stop = True
                self.get_logger().warn("=== TRAFFIC LIGHT STOP: True - Cancelling current navigation goal ===")
                self.cancel_current_goal()
            
            # 检测下降沿 (True -> False) - 恢复导航
            elif not msg.data and self.traffic_light_prev_state:
                self.traffic_light_stop = False
                self.get_logger().info("=== TRAFFIC LIGHT STOP: False - Resuming navigation ===")
                # 重新发送当前目标（仅当不在抓取/放置过程中时）
                if not self.goal_in_progress and self.current_goal_idx < len(self.waypoints):
                    self.send_current_goal()
            
            # 更新上一状态
            self.traffic_light_prev_state = msg.data
    
    def cancel_current_goal(self):
        """取消当前正在执行的导航目标（仅导航）"""
        if self.current_goal_handle and self.goal_in_progress:
            try:
                # 发送取消请求
                cancel_future = self.current_goal_handle.cancel_goal_async()
                # 等待取消完成
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                self.get_logger().info("Current navigation goal cancelled successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to cancel goal: {str(e)}")
            finally:
                self.goal_in_progress = False
                self.current_goal_handle = None
        else:
            self.get_logger().info("No active navigation goal to cancel")
    
    def wait_for_grasp_success(self):
        """Wait until grasp success signal is received - 不受交通灯控制"""
        self.get_logger().info("Waiting for grasp success signal (/grasp/success=True)...")
        self.grasp_success_received = False
        
        while rclpy.ok() and not self.grasp_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.grasp_success_received:
                self.get_logger().debug("Still waiting for grasp success...")
        
        self.get_logger().info("Grasp success received - resuming navigation")
        return True
    
    def wait_for_place_success(self):
        """Wait until place success signal is received - 不受交通灯控制"""
        self.get_logger().info("Waiting for place success signal (/place/success=True)...")
        self.place_success_received = False
        
        while rclpy.ok() and not self.place_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.place_success_received:
                self.get_logger().debug("Still waiting for place success...")
        
        self.get_logger().info("Place success received - resuming navigation")
        return True

    def publish_init_pose(self):
        """Publish initial pose for AMCL localization - 受交通灯控制"""
        if self.init_sent:
            return
        
        # 检查交通灯停止信号
        with self.stop_lock:
            if self.traffic_light_stop:
                self.get_logger().warn("Cannot publish initial pose - traffic light stop active")
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
            with self.stop_lock:
                if self.traffic_light_stop:
                    break
            self.init_pose_pub.publish(msg)
            self.get_logger().info("Published /initialpose for localization...")
            rclpy.spin_once(self, timeout_sec=1.0)
        
        self.init_sent = True
        self.get_logger().info("Waiting 5s for AMCL localization to settle...")
        rclpy.spin_once(self, timeout_sec=5.0)

    def send_current_goal(self):
        """Send the CURRENT waypoint goal to NavigateToPose action server - 受交通灯控制"""
        # 检查交通灯停止信号
        with self.stop_lock:
            if self.traffic_light_stop:
                self.get_logger().warn("Cannot send navigation goal - traffic light stop active")
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
        if self.goal_in_progress:
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
        """Callback for goal response from action server - 受交通灯控制"""
        try:
            # 检查交通灯停止信号
            with self.stop_lock:
                if self.traffic_light_stop:
                    self.get_logger().warn("Goal response received but traffic light stop active - cancelling goal")
                    self.goal_in_progress = False
                    return
                
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f"Goal {self.current_goal_idx + 1} rejected by server!")
                self.goal_in_progress = False
                return

            self.get_logger().info(f"Goal {self.current_goal_idx + 1} accepted by server, navigating...")
            # Save current goal handle
            self.current_goal_handle = goal_handle
            # Register callback for result retrieval
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.result_callback)
        except Exception as e:
            self.get_logger().error(f"Goal response error: {str(e)}")
            self.goal_in_progress = False

    def result_callback(self, future):
        """Callback for navigation result"""
        try:
            # 检查交通灯停止信号（防止在结果返回时已经触发停止）
            with self.stop_lock:
                if self.traffic_light_stop:
                    self.get_logger().warn("Result received but traffic light stop active - resetting state")
                    self.goal_in_progress = False
                    self.current_goal_handle = None
                    return
            
            # Reset navigation state
            self.goal_in_progress = False
            self.current_goal_handle = None
            
            # Check if the goal was actually achieved (nav2 returns result code 4 for success)
            result = future.result()
            if result.status == 4:  # Nav2 SUCCESS status code
                current_goal_number = self.current_goal_idx + 1
                self.get_logger().info(f"Goal {current_goal_number} reached!")
                
                proceed_to_next = True
                
                # arrived at fourth waypoint: send /grasp/start True signal, then wait for grasp success
                if current_goal_number == 4:
                    self.get_logger().info("Publishing /grasp/start=True signal")
                    grasp_start_msg = Bool()
                    grasp_start_msg.data = True
                    self.grasp_start_pub.publish(grasp_start_msg)
                    proceed_to_next = self.wait_for_grasp_success()
                
                # arrived at fifth waypoint: send /place/start True signal, then wait for place success
                elif current_goal_number == 5:
                    self.get_logger().info("Publishing /place/start=True signal")
                    place_start_msg = Bool()
                    place_start_msg.data = True
                    self.place_start_pub.publish(place_start_msg)
                    proceed_to_next = self.wait_for_place_success()
                
                # Move to next waypoint
                if proceed_to_next:
                    self.current_goal_idx += 1
                    # 检查交通灯状态后再发送下一个导航目标
                    with self.stop_lock:
                        if not self.traffic_light_stop:
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
    # 使用多线程执行器，确保回调函数及时处理
    executor = MultiThreadedExecutor(num_threads=4)
    navigator = WaypointQuatNavigator()
    
    # Start navigation with first waypoint
    navigator.send_next_goal()
    
    # Keep node spinning with multi-threaded executor
    try:
        rclpy.spin(navigator, executor=executor)
    except KeyboardInterrupt:
        navigator.get_logger().info("Navigation interrupted by user")
    except Exception as e:
        navigator.get_logger().error(f"Navigation error: {str(e)}")
    
    # Cleanup
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()