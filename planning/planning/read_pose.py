#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class CurrentPosePrinter(Node):
    def __init__(self):
        super().__init__('current_pose_printer')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10)

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.get_logger().info(
            f"Current: x={pos.x:.3f}, y={pos.y:.3f}, z={ori.z:.6f}, w={ori.w:.6f}")
        print(f"{pos.x:.4f}, {pos.y:.4f}, {ori.z:.6f}, {ori.w:.6f}")

def main(args=None):
    rclpy.init(args=args)
    node = CurrentPosePrinter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()