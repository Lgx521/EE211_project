#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from std_msgs.msg import Bool

class PlaceBoxNode(Node):
    def __init__(self):
        super().__init__('place_box_node')
        # Publishers
        self.pub_arm = self.create_publisher(JointGroupCommand, '/px100/commands/joint_group', 10)
        self.pub_gripper = self.create_publisher(JointSingleCommand, '/px100/commands/joint_single', 10)
        self.pub_place_success = self.create_publisher(Bool, '/place/success', 10)

        # Simple state machine
        self.state = 'MOVE_OUT'
        self.wait_ticks = 2  # give some time for publishers to connect

        # Timer
        self.timer = self.create_timer(0.5, self.loop)
        self.get_logger().info('PlaceBoxNode started. Will extend arm and open gripper.')

    def send_arm(self, joints):
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = joints
        self.pub_arm.publish(msg)

    def send_gripper(self, val):
        msg = JointSingleCommand()
        msg.name = 'gripper'
        msg.cmd = float(val)
        self.pub_gripper.publish(msg)

    def publish_place_success(self):
        """发布放置成功话题"""
        msg = Bool()
        msg.data = True
        self.pub_place_success.publish(msg)
        self.get_logger().info("✅ 发布放置成功话题: /place/success")

    def loop(self):
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        if self.state == 'MOVE_OUT':
            # A safe forward "place" pose; gripper level (shoulder+elbow+wrist≈0)
            # [waist, shoulder, elbow, wrist]
            place_pose = [-1.57, -0.4, 1.1, -0.7]
            self.get_logger().info('Extending arm to place pose...')
            self.send_arm(place_pose)
            self.state = 'OPEN'
            self.wait_ticks = 6  # wait ~3s for the arm to reach

        elif self.state == 'OPEN':
            self.get_logger().info('Opening gripper to release object...')
            self.send_gripper(1.5)  # fully open
            self.state = 'DONE'
            self.wait_ticks = 4  # short wait then exit

        elif self.state == 'DONE':
            self.get_logger().info('Done. Exiting...')
            # 发布放置成功话题
            self.publish_place_success()
            # Cleanly shutdown
            # self.destroy_node()
            # rclpy.shutdown()


def main():
    rclpy.init()
    node = PlaceBoxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

