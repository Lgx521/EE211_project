#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Bool

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
        
        # Subscriber for traffic light stop control
        self.traffic_light_stop = False
        self.previous_traffic_light_stop = False
        self.traffic_light_sub = self.create_subscription(
            Bool,
            '/traffic_light/stop_control',
            self.traffic_light_callback,
            10
        )
        
        # List of waypoints (x, y, z, w quaternion)
        self.waypoints = self.get_waypoints()
        self.current_goal_idx = 0
        self.init_sent = False
        self.goal_in_progress = False
        self.current_goal_handle = None
        self.current_route = None
        
        # State machine for handling grasp/place operations
        # States: 'navigating', 'waiting_grasp', 'waiting_place', 'idle', 'traffic_stopped'
        self.state = 'idle'
        
        # Traffic light control: remember state before stop
        self.state_before_traffic_stop = None

    def get_waypoints(self):
        """Define navigation waypoints with x, y, z (orientation), w (orientation)"""
        return [
            [6.7132, -2.0496, -0.483441, 0.875377],
            [6.7167, -3.2733, -0.351566, 0.936163],
            [8.7445, -4.2035, 0.381371, 0.924422],
            [11.1434, -2.4972, -0.212324, 0.954916],
            [9.0319, -0.5364, -0.998852, 0.047912],
            [6.7132, -2.0496, -0.483441, 0.875377]
        ]
    
    def grasp_success_callback(self, msg):
        """Callback for grasp success signal"""
        if msg.data:
            self.grasp_success_received = True
            self.get_logger().info("Received grasp success signal (/grasp/success=True)")
            
            # If waiting for grasp, proceed to next waypoint
            if self.state == 'waiting_grasp':
                self.get_logger().info("Grasp completed, proceeding to next waypoint")
                self.state = 'idle'
                self.current_goal_idx += 1
                self.send_current_goal()
    
    def place_success_callback(self, msg):
        """Callback for place success signal"""
        if msg.data:
            self.place_success_received = True
            self.get_logger().info("Received place success signal (/place/success=True)")
            
            # If waiting for place, proceed to next waypoint
            if self.state == 'waiting_place':
                self.get_logger().info("Place completed, proceeding to next waypoint")
                self.state = 'idle'
                self.current_goal_idx += 1
                self.send_current_goal()
    
    def traffic_light_callback(self, msg):
        """Callback for traffic light stop control signal"""
        self.traffic_light_stop = msg.data
        
        # Detect rising edge (False -> True): Stop signal
        if self.traffic_light_stop and not self.previous_traffic_light_stop:
            self.get_logger().warn("[object Object] STOP signal received - cancelling current goal")
            self.handle_traffic_stop()
        
        # Detect falling edge (True -> False): Resume signal
        elif not self.traffic_light_stop and self.previous_traffic_light_stop:
            self.get_logger().info("🚦 Traffic light GO signal received - resuming navigation")
            self.handle_traffic_resume()
        
        # Update previous state for edge detection
        self.previous_traffic_light_stop = self.traffic_light_stop
    
    def handle_traffic_stop(self):
        """Handle traffic light stop signal - cancel current navigation goal"""
        # Only cancel if there's an active goal
        if self.goal_in_progress and self.current_goal_handle is not None:
            self.get_logger().info(f"Cancelling navigation to waypoint {self.current_goal_idx + 1}")
            
            # Save current state before stopping
            self.state_before_traffic_stop = self.state
            self.state = 'traffic_stopped'
            
            # Cancel the current goal
            try:
                cancel_future = self.current_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
            except Exception as e:
                self.get_logger().error(f"Failed to cancel goal: {str(e)}")
                self.goal_in_progress = False
                self.current_goal_handle = None
        else:
            # No active navigation, just update state
            self.get_logger().info("No active navigation goal to cancel")
            self.state_before_traffic_stop = self.state
            self.state = 'traffic_stopped'
    
    def handle_traffic_resume(self):
        """Handle traffic light resume signal - continue to the same waypoint"""
        if self.state == 'traffic_stopped':
            self.get_logger().info(f"Resuming navigation to waypoint {self.current_goal_idx + 1}")
            
            # Restore previous state
            if self.state_before_traffic_stop is not None:
                self.state = self.state_before_traffic_stop
                self.state_before_traffic_stop = None
            else:
                self.state = 'idle'
            
            # Resume navigation to the SAME waypoint (index not changed)
            self.send_current_goal()
        else:
            self.get_logger().info("Not in traffic_stopped state, ignoring resume signal")
    
    def cancel_done_callback(self, future):
        """Callback when goal cancellation is complete"""
        try:
            cancel_response = future.result()
            if len(cancel_response.goals_canceling) > 0:
                self.get_logger().info("Goal successfully cancelled")
            else:
                self.get_logger().warn("Goal cancellation may have failed")
        except Exception as e:
            self.get_logger().error(f"Cancel callback error: {str(e)}")
        finally:
            # Reset goal state
            self.goal_in_progress = False
            self.current_goal_handle = None

    def publish_init_pose(self):
        """Publish initial pose for AMCL localization"""
        if self.init_sent:
            return
        
        # Wait for /initialpose topic to have subscribers (AMCL)
        self.get_logger().info("Waiting for AMCL to be ready (checking /initialpose subscribers)...")
        wait_count = 0
        max_wait = 30  # Wait up to 30 seconds
        while self.init_pose_pub.get_subscription_count() == 0 and wait_count < max_wait:
            self.get_logger().info(f"Waiting for /initialpose subscribers... ({wait_count + 1}/{max_wait})")
            rclpy.spin_once(self, timeout_sec=1.0)
            wait_count += 1
        
        if self.init_pose_pub.get_subscription_count() == 0:
            self.get_logger().warn("No subscribers on /initialpose topic! AMCL may not be running.")
            self.get_logger().warn("Proceeding anyway, but localization may fail.")
        else:
            self.get_logger().info(f"Found {self.init_pose_pub.get_subscription_count()} subscriber(s) on /initialpose")
        
        # Construct initial pose message
        x, y, z, w = self.waypoints[0]
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        
        # Set orientation
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w
        
        # Set covariance matrix (6x6 = 36 elements)
        # Diagonal elements: x, y, z, rotation_x, rotation_y, rotation_z
        # Small values indicate high confidence in the initial pose
        msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  # x variance
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,  # y variance
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # z variance (not used in 2D)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # rotation x (not used in 2D)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # rotation y (not used in 2D)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.068  # rotation z (yaw) variance
        ]
        
        # Publish initial pose multiple times to ensure reception
        self.get_logger().info(f"Publishing initial pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, w={w:.3f}")
        for i in range(5):
            msg.header.stamp = self.get_clock().now().to_msg()  # Update timestamp
            self.init_pose_pub.publish(msg)
            self.get_logger().info(f"Published /initialpose ({i + 1}/5)")
            rclpy.spin_once(self, timeout_sec=0.5)
        
        self.init_sent = True
        self.get_logger().info("Waiting 3s for AMCL localization to settle...")
        
        # Wait for AMCL to process the initial pose
        for i in range(6):
            rclpy.spin_once(self, timeout_sec=0.5)
        
        self.get_logger().info("Initial pose published successfully!")

    def send_current_goal(self):
        """Send the CURRENT waypoint goal (not next) to NavigateToPose action server"""
        # Publish initial pose if not sent yet
        if not self.init_sent:
            self.publish_init_pose()
        
        # Check if traffic light is stopping us
        if self.state == 'traffic_stopped':
            self.get_logger().warn("Traffic light is RED, cannot send new goal")
            return
        
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
        """Callback for goal response from action server"""
        try:
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
            # Reset navigation state
            self.goal_in_progress = False
            self.current_goal_handle = None
            
            # Check if the goal was actually achieved (nav2 returns result code 4 for success)
            result = future.result()
            if result.status == 4:  # Nav2 SUCCESS status code
                current_goal_number = self.current_goal_idx + 1
                self.get_logger().info(f"Goal {current_goal_number} reached!")
                
                # arrived at fourth waypoint: send /grasp/start True signal, then wait for grasp success
                if current_goal_number == 4:
                    self.get_logger().info("Publishing /grasp/start=True signal")
                    grasp_start_msg = Bool()
                    grasp_start_msg.data = True
                    self.grasp_start_pub.publish(grasp_start_msg)
                    
                    # Enter waiting_grasp state - will proceed when grasp_success_callback is triggered
                    self.state = 'waiting_grasp'
                    self.grasp_success_received = False
                    self.get_logger().info("Waiting for grasp success signal (/grasp/success=True)...")
                
                # arrived at fifth waypoint: send /place/start True signal, then wait for place success
                elif current_goal_number == 5:
                    self.get_logger().info("Publishing /place/start=True signal")
                    place_start_msg = Bool()
                    place_start_msg.data = True
                    self.place_start_pub.publish(place_start_msg)
                    
                    # Enter waiting_place state - will proceed when place_success_callback is triggered
                    self.state = 'waiting_place'
                    self.place_success_received = False
                    self.get_logger().info("Waiting for place success signal (/place/success=True)...")
                
                # For other waypoints, proceed directly to next
                else:
                    self.state = 'idle'
                    self.current_goal_idx += 1
                    self.send_current_goal()
            else:
                # Goal was cancelled or failed
                self.get_logger().info(f"Goal {self.current_goal_idx + 1} cancelled/interrupted - preserving index")
                
                # If not in traffic_stopped state, reset to idle
                # If in traffic_stopped state, keep that state (will resume later)
                if self.state != 'traffic_stopped':
                    self.state = 'idle'
                
        except Exception as e:
            self.get_logger().error(f"Result callback error: {str(e)}")
            self.goal_in_progress = False
            if self.state != 'traffic_stopped':
                self.state = 'idle'

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
        pass
    
    # Cleanup
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()