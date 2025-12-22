#!/usr/bin/env python3
"""
Test script for traffic light control
Publishes True/False to /traffic_light/stop_control to test the navigation stop/resume functionality
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

class TrafficLightTester(Node):
    def __init__(self):
        super().__init__('traffic_light_tester')
        self.publisher = self.create_publisher(Bool, '/traffic_light/stop_control', 10)
        self.get_logger().info("Traffic Light Tester started")
        self.get_logger().info("Commands:")
        self.get_logger().info("  Press 's' + Enter to send STOP (True)")
        self.get_logger().info("  Press 'g' + Enter to send GO (False)")
        self.get_logger().info("  Press 'q' + Enter to quit")
    
    def send_stop(self):
        """Send stop signal (True)"""
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)
        self.get_logger().info("🔴 Sent STOP signal (True) to /traffic_light/stop_control")
    
    def send_go(self):
        """Send go signal (False)"""
        msg = Bool()
        msg.data = False
        self.publisher.publish(msg)
        self.get_logger().info("🟢 Sent GO signal (False) to /traffic_light/stop_control")

def main(args=None):
    rclpy.init(args=args)
    tester = TrafficLightTester()
    
    # Give time for publisher to establish
    time.sleep(0.5)
    
    try:
        while rclpy.ok():
            # Process ROS callbacks
            rclpy.spin_once(tester, timeout_sec=0.1)
            
            # Check for user input (non-blocking)
            import select
            import sys
            
            if select.select([sys.stdin], [], [], 0)[0]:
                command = sys.stdin.readline().strip().lower()
                
                if command == 's':
                    tester.send_stop()
                elif command == 'g':
                    tester.send_go()
                elif command == 'q':
                    tester.get_logger().info("Quitting...")
                    break
                else:
                    tester.get_logger().warn(f"Unknown command: {command}")
    
    except KeyboardInterrupt:
        tester.get_logger().info("Interrupted by user")
    
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

