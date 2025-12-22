#!/usr/bin/env python3
"""
Diagnostic script for navigation system
Checks if all required nodes and topics are available
"""

import rclpy
from rclpy.node import Node
import time

class NavigationDiagnostics(Node):
    def __init__(self):
        super().__init__('navigation_diagnostics')
        self.get_logger().info("Navigation Diagnostics Tool")
        
    def check_topics(self):
        """Check if required topics exist"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Checking Topics...")
        self.get_logger().info("="*60)
        
        required_topics = [
            '/initialpose',
            '/map',
            '/tf',
            '/tf_static',
            '/amcl_pose',
            '/scan',
            '/cmd_vel'
        ]
        
        # Get list of topics
        topic_list = self.get_topic_names_and_types()
        topic_names = [name for name, _ in topic_list]
        
        all_ok = True
        for topic in required_topics:
            if topic in topic_names:
                self.get_logger().info(f"✓ {topic} - Found")
            else:
                self.get_logger().warn(f"✗ {topic} - NOT FOUND")
                all_ok = False
        
        return all_ok
    
    def check_nodes(self):
        """Check if required nodes are running"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Checking Nodes...")
        self.get_logger().info("="*60)
        
        required_nodes = [
            'amcl',
            'bt_navigator',
            'controller_server',
            'planner_server',
            'behavior_server'
        ]
        
        # Get list of nodes
        node_list = self.get_node_names()
        
        all_ok = True
        for node in required_nodes:
            # Check if node name contains the required string
            found = any(node in n for n in node_list)
            if found:
                self.get_logger().info(f"✓ {node} - Running")
            else:
                self.get_logger().warn(f"✗ {node} - NOT RUNNING")
                all_ok = False
        
        return all_ok
    
    def check_action_servers(self):
        """Check if NavigateToPose action server is available"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Checking Action Servers...")
        self.get_logger().info("="*60)
        
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose
        
        action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.get_logger().info("Waiting for NavigateToPose action server...")
        server_available = action_client.wait_for_server(timeout_sec=5.0)
        
        if server_available:
            self.get_logger().info("✓ NavigateToPose action server - Available")
            return True
        else:
            self.get_logger().warn("✗ NavigateToPose action server - NOT AVAILABLE")
            return False
    
    def check_map(self):
        """Check if map is loaded"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Checking Map...")
        self.get_logger().info("="*60)
        
        from nav_msgs.msg import OccupancyGrid
        
        map_received = [False]
        
        def map_callback(msg):
            map_received[0] = True
            self.get_logger().info(f"✓ Map received: {msg.info.width}x{msg.info.height} cells")
            self.get_logger().info(f"  Resolution: {msg.info.resolution} m/cell")
            self.get_logger().info(f"  Origin: ({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})")
        
        map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            map_callback,
            10
        )
        
        # Wait for map
        for i in range(10):
            if map_received[0]:
                break
            rclpy.spin_once(self, timeout_sec=0.5)
        
        if not map_received[0]:
            self.get_logger().warn("✗ No map received on /map topic")
            self.get_logger().warn("  Make sure to load a map:")
            self.get_logger().warn("  ros2 launch turtlebot4_navigation localization.launch.py map:=<your_map>")
            return False
        
        return True
    
    def run_diagnostics(self):
        """Run all diagnostic checks"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("NAVIGATION SYSTEM DIAGNOSTICS")
        self.get_logger().info("="*60)
        
        time.sleep(1.0)  # Give time for discovery
        
        results = {
            'topics': self.check_topics(),
            'nodes': self.check_nodes(),
            'map': self.check_map(),
            'action_servers': self.check_action_servers()
        }
        
        # Summary
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("SUMMARY")
        self.get_logger().info("="*60)
        
        all_passed = all(results.values())
        
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            self.get_logger().info(f"{check.upper()}: {status}")
        
        self.get_logger().info("="*60)
        
        if all_passed:
            self.get_logger().info("✅ All checks passed! Navigation system is ready.")
        else:
            self.get_logger().warn("⚠️  Some checks failed. Please fix the issues above.")
            self.get_logger().info("\nCommon fixes:")
            self.get_logger().info("1. Start localization:")
            self.get_logger().info("   ros2 launch turtlebot4_navigation localization.launch.py map:=<map_file>")
            self.get_logger().info("2. Start Nav2:")
            self.get_logger().info("   ros2 launch turtlebot4_navigation nav2.launch.py")
        
        self.get_logger().info("="*60 + "\n")
        
        return all_passed

def main(args=None):
    rclpy.init(args=args)
    diagnostics = NavigationDiagnostics()
    
    try:
        diagnostics.run_diagnostics()
    except Exception as e:
        diagnostics.get_logger().error(f"Error during diagnostics: {str(e)}")
    
    diagnostics.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

