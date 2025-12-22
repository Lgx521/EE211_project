#!/usr/bin/env python3
"""
Test script for initial pose publishing
Tests the /initialpose publication to AMCL
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import time

class InitPoseTester(Node):
    def __init__(self):
        super().__init__('init_pose_tester')
        
        # Publisher for initial pose
        self.init_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )
        
        self.get_logger().info("Init Pose Tester started")
        
    def publish_test_pose(self, x, y, yaw_z, yaw_w):
        """Publish a test initial pose"""
        
        # Wait for subscribers
        self.get_logger().info("Waiting for /initialpose subscribers (AMCL)...")
        wait_count = 0
        max_wait = 10
        
        while self.init_pose_pub.get_subscription_count() == 0 and wait_count < max_wait:
            self.get_logger().info(f"Waiting... ({wait_count + 1}/{max_wait})")
            rclpy.spin_once(self, timeout_sec=1.0)
            wait_count += 1
        
        if self.init_pose_pub.get_subscription_count() == 0:
            self.get_logger().error("❌ No subscribers found on /initialpose!")
            self.get_logger().error("   Make sure AMCL is running:")
            self.get_logger().error("   ros2 launch turtlebot4_navigation localization.launch.py map:=<your_map>")
            return False
        
        self.get_logger().info(f"✓ Found {self.init_pose_pub.get_subscription_count()} subscriber(s)")
        
        # Create message
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set pose
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = yaw_z
        msg.pose.pose.orientation.w = yaw_w
        
        # Set covariance (same as RViz default)
        msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.068
        ]
        
        # Publish multiple times
        self.get_logger().info(f"📍 Publishing initial pose:")
        self.get_logger().info(f"   Position: x={x:.3f}, y={y:.3f}")
        self.get_logger().info(f"   Orientation: z={yaw_z:.3f}, w={yaw_w:.3f}")
        
        for i in range(5):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.init_pose_pub.publish(msg)
            self.get_logger().info(f"   Published ({i + 1}/5)")
            time.sleep(0.2)
        
        self.get_logger().info("✅ Initial pose published successfully!")
        self.get_logger().info("   Check RViz to see if the robot pose was updated")
        return True

def main(args=None):
    rclpy.init(args=args)
    tester = InitPoseTester()
    
    # Give time for node to initialize
    time.sleep(0.5)
    
    # Test with first waypoint coordinates
    # You can modify these values to test different positions
    test_x = 6.7132
    test_y = -2.0496
    test_z = -0.483441
    test_w = 0.875377
    
    print("\n" + "="*60)
    print("Initial Pose Test Script")
    print("="*60)
    print(f"Testing with coordinates:")
    print(f"  x={test_x}, y={test_y}")
    print(f"  orientation: z={test_z}, w={test_w}")
    print("="*60 + "\n")
    
    success = tester.publish_test_pose(test_x, test_y, test_z, test_w)
    
    if success:
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
        print("Next steps:")
        print("1. Check RViz - the robot should appear at the specified position")
        print("2. If the robot doesn't appear, check:")
        print("   - Is AMCL running? (ros2 node list | grep amcl)")
        print("   - Is the map loaded? (check /map topic)")
        print("   - Are the coordinates within the map bounds?")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("❌ Test failed - AMCL not ready")
        print("="*60 + "\n")
    
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

