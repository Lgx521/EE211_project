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
    
    def wait_for_grasp_success(self):
        """Wait until grasp success signal is received"""
        self.get_logger().info("Waiting for grasp success signal (/grasp/success=True)...")
        self.grasp_success_received = False
        
        while rclpy.ok() and not self.grasp_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.grasp_success_received:
                self.get_logger().debug("Still waiting for grasp success...")
        
        self.get_logger().info("Grasp success received - resuming navigation")
        return True
    
    def wait_for_place_success(self):
        """Wait until place success signal is received"""
        self.get_logger().info("Waiting for place success signal (/place/success=True)...")
        self.place_success_received = False
        
        while rclpy.ok() and not self.place_success_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.place_success_received:
                self.get_logger().debug("Still waiting for place success...")
        
        self.get_logger().info("Place success received - resuming navigation")
        return True

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
                
                proceed_to_next = True
                
                # arrived at fourth waypoint: send /grasp/start True signal, then wait for grasp success
                if current_goal_number == 4:
                    self.get_logger().info("Publishing /grasp/start=True signal")
                    grasp_start_msg = Bool()
                    grasp_start_msg.data = True
                    self.grasp_start_pub.publish(grasp_start_msg)
                    proceed_to_next = self.wait_for_grasp_success()
                
                # arrived at sixth waypoint: send /place/start True signal, then wait for place success
                elif current_goal_number == 5:
                    self.get_logger().info("Publishing /place/start=True signal")
                    place_start_msg = Bool()
                    place_start_msg.data = True
                    self.place_start_pub.publish(place_start_msg)
                    proceed_to_next = self.wait_for_place_success()
                
                # Move to next waypoint
                if proceed_to_next:
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
        pass
    
    # Cleanup
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()