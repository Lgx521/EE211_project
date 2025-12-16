#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time

from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class PX100Kinematics:
    def __init__(self):
        self.L1 = 0.0445
        self.L2 = 0.100
        self.L3 = 0.100
        self.L4 = 0.110 
        
        # [修改点 1] 抓取偏移量 (单位: 米)
        # 如果觉得还伸太远，把这个数改大 (例如 0.05)
        # 如果觉得抓不到物体(太短)，把这个数改小 (例如 0.02)
        self.GRASP_OFFSET = 0.035 

    def solve_ik(self, x, y, z, debug=False):
        waist = math.atan2(y, x)
        
        # 计算平面距离
        r_target_raw = math.sqrt(x**2 + y**2)
        
        # [修改点 2] 全局回退：防止夹爪怼到物体里面
        # 我们希望夹爪包住物体，而不是尖端顶住物体中心
        r_target = r_target_raw - self.GRASP_OFFSET

        z_w = z - self.L1 

        if debug:
            print(f"[IK] 原目标距离: {r_target_raw:.3f}, 修正后距离: {r_target:.3f}")

        # --- 策略 A: 水平抓取 ---
        r_wrist_h = r_target - self.L4
        z_wrist_h = z_w
        joints = self._compute_2link_ik(r_wrist_h, z_wrist_h, pitch_goal=0.0)
        if joints:
            return [waist] + joints

        # --- 策略 B: 指向抓取 ---
        if debug: print("[IK] 尝试指向抓取...")
        angle_to_origin = math.atan2(z_w, r_target)
        
        # 修正: 在指向模式下，也需要确保不要伸过头
        # 这里的 L4 实际上是把手腕向后推
        r_wrist_p = r_target - (self.L4 * math.cos(angle_to_origin))
        z_wrist_p = z_w - (self.L4 * math.sin(angle_to_origin))
        
        joints = self._compute_2link_ik(r_wrist_p, z_wrist_p, pitch_goal=angle_to_origin)
        if joints:
            return [waist] + joints

        return None

    def _compute_2link_ik(self, r, z, pitch_goal):
        d = math.sqrt(r**2 + z**2)
        if d > (self.L2 + self.L3) or d < 0.02: return None

        alpha = math.atan2(z, r)
        try:
            cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
            if abs(cos_beta) > 1.0: return None
            beta = math.acos(cos_beta)

            theta_shoulder = alpha + beta # Elbow Up

            cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
            if abs(cos_gamma) > 1.0: return None
            gamma = math.acos(cos_gamma)

            theta_elbow = -1 * (math.pi - gamma)
            theta_wrist = pitch_goal - theta_shoulder - theta_elbow

            # 简单的安全限位检查
            if not (-2.0 < theta_shoulder < 2.0): return None
            if not (-2.5 < theta_elbow < 2.5): return None
            
            return [theta_shoulder, theta_elbow, theta_wrist]
        except ValueError:
            return None

class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        self.pub_arm = self.create_publisher(JointGroupCommand, "/px100/commands/joint_group", 10)
        self.pub_gripper = self.create_publisher(JointSingleCommand, "/px100/commands/joint_single", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_poses = self.create_subscription(PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)

        self.ik_solver = PX100Kinematics()
        
        self.state = "SEARCHING"
        self.target_pose_base = None
        self.stable_counter = 0
        
        # 为了防止 control_loop 阻塞，我们稍微加快频率，但用状态机控制等待
        self.timer = self.create_timer(0.5, self.control_loop)
        
        # 增加一个通用计数器用于非阻塞延时
        self.wait_ticks = 0
        
        self.get_logger().info("✅ 节点启动")

    def cb_poses(self, msg: PoseArray):
        if self.state != "SEARCHING": return
        if len(msg.poses) == 0: return

        try:
            target_cam = PoseStamped()
            target_cam.header = msg.header
            target_cam.pose = msg.poses[0]
            
            transform = self.tf_buffer.lookup_transform("px100/base_link", target_cam.header.frame_id, rclpy.time.Time())
            pose_base = do_transform_pose(target_cam.pose, transform)
            
            self.target_pose_base = pose_base
            self.stable_counter += 1
            if self.stable_counter > 5: # 稍微多等几帧稳定
                self.state = "PREPARE_GRASP" # 新状态
                self.stable_counter = 0

        except Exception as e:
            pass

    def send_arm(self, joints):
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = joints
        self.pub_arm.publish(msg)

    def send_gripper(self, val):
        msg = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd = float(val)
        self.pub_gripper.publish(msg)

    def control_loop(self):
        # 简单的非阻塞延时逻辑
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        if self.state == "SEARCHING":
            # 持续发送张开指令，防止夹爪在大力矩下松脱
            # PX100 夹爪范围通常是 1.5 (Open) 到 -1.0 (Close) 左右的PWM/Current
            # 如果是 Position 模式，范围是 0.03(Open) 到 0.015(Close)
            # 这里假设你是PWM模式，发送 1.0
            pass 

        elif self.state == "PREPARE_GRASP":
            # [修改点 3] 显式张开夹爪
            self.get_logger().info("👐 准备抓取：强制张开夹爪")
            self.send_gripper(1.5) # 给大一点的值确保完全张开
            self.state = "CALCULATE"
            self.wait_ticks = 2 # 等待 1秒 (0.5s * 2)

        elif self.state == "CALCULATE":
            if not self.target_pose_base: 
                self.state = "SEARCHING"
                return

            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z = self.target_pose_base.position.z

            self.get_logger().info("🧮 计算逆运动学...")
            joints = self.ik_solver.solve_ik(x, y, z, debug=True)
            
            if joints:
                self.get_logger().info(f"🚀 移动机械臂! 关节: {[round(j,2) for j in joints]}")
                self.send_arm(joints)
                self.state = "GRASP_WAIT"
                self.wait_ticks = 6 # 等待 3秒，给机械臂运动时间
            else:
                self.get_logger().error("❌ IK 无法解算 (目标不可达)")
                self.state = "SEARCHING"

        elif self.state == "GRASP_WAIT":
            # 运动结束，开始闭合
            self.state = "CLOSE"

        elif self.state == "CLOSE":
            self.get_logger().info("✊ 闭合夹爪")
            self.send_gripper(-0.8) # 闭合力度加大一点
            self.state = "RETRACT_WAIT"
            self.wait_ticks = 3 # 等待 1.5秒让夹爪夹紧

        elif self.state == "RETRACT_WAIT":
            self.state = "RETRACT"

        elif self.state == "RETRACT":
            self.get_logger().info("⬅️ 收回")
            self.send_arm([0.0, -0.3, 1.57, -0.5])
            self.state = "DONE"
            self.wait_ticks = 4

        elif self.state == "DONE":
            # 这里的逻辑可以改成放下物体，或者回到SEARCHING
            self.get_logger().info("✅ 完成")
            # self.state = "SEARCHING" 

def main():
    rclpy.init()
    node = ArucoGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()