#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import time

from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class PX100Kinematics:
    def __init__(self):
        # 机械臂物理参数
        self.L1 = 0.0445
        self.L2 = 0.100
        self.L3 = 0.100
        self.L4 = 0.110 
        
        # 水平回退偏移 (保持这个不变，这是结构需要的)
        self.GRASP_X_OFFSET = 0.035 

    def solve_ik(self, x, y, z, debug=False):
        # --- 这里的 z 就是绝对目标高度，没有任何隐藏偏移 ---
        
        # 1. 计算腰部角度
        waist = math.atan2(y, x)
        
        # 2. 水平距离修正
        r_target_raw = math.sqrt(x**2 + y**2)
        r_target = r_target_raw - self.GRASP_X_OFFSET
        
        # 3. 转换为 Shoulder 坐标系 (只减底座高度)
        # z_w = z - self.L1 
        z_w = z 

        # --- 策略 A: 优先尝试水平抓取 ---
        r_wrist_h = r_target - self.L4
        z_wrist_h = z_w
        joints = self._compute_2link_ik(r_wrist_h, z_wrist_h, pitch_goal=0.0)
        if joints: return [waist] + joints

        # --- 策略 B: 尝试指向抓取 ---
        angle_to_origin = math.atan2(z_w, r_target)
        r_wrist_p = r_target - (self.L4 * math.cos(angle_to_origin))
        z_wrist_p = z_w - (self.L4 * math.sin(angle_to_origin))
        joints = self._compute_2link_ik(r_wrist_p, z_wrist_p, pitch_goal=angle_to_origin)
        if joints: return [waist] + joints

        return None

    def _compute_2link_ik(self, r, z, pitch_goal):
        d = math.sqrt(r**2 + z**2)
        if d > (self.L2 + self.L3) or d < 0.02: return None

        alpha = math.atan2(z, r)
        try:
            cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
            if abs(cos_beta) > 1.0: return None
            beta = math.acos(cos_beta)
            theta_shoulder = alpha + beta 

            cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
            if abs(cos_gamma) > 1.0: return None
            gamma = math.acos(cos_gamma)

            theta_elbow = -1 * (math.pi - gamma)
            theta_wrist = pitch_goal - theta_shoulder - theta_elbow

            if not (-2.0 < theta_shoulder < 2.0): return None
            if not (-2.4 < theta_elbow < 2.4): return None
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
        self.wait_ticks = 0
        self.timer = self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info("✅ 节点启动: 直接在 State Machine 里加减 Z 值")

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
            if self.stable_counter > 5:
                self.state = "PREPARE_GRASP"
                self.stable_counter = 0
        except Exception:
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
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        if self.state == "SEARCHING":
            pass 

        elif self.state == "PREPARE_GRASP":
            self.get_logger().info("👐 准备抓取")
            self.send_gripper(1.5) 
            self.state = "MOVE_HOVER" 
            self.wait_ticks = 2 

        # --- 步骤 1: 移动到上方 (z + 0.05) ---
        elif self.state == "MOVE_HOVER":
            if not self.target_pose_base: 
                self.state = "SEARCHING"
                return

            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            # 直接在这里加 5cm
            z_target = z_aruco - 0.07

            self.get_logger().info(f"🚁 悬停: Z={z_target:.3f}")
            joints = self.ik_solver.solve_ik(x, y, z_target)
            
            if joints:
                self.send_arm(joints)
                self.state = "MOVE_DOWN" 
                self.wait_ticks = 5     
            else:
                self.get_logger().error("❌ 悬停 IK 无解")
                self.state = "SEARCHING"

        # --- 步骤 2: 下落抓取 (z - 0.025) ---
        elif self.state == "MOVE_DOWN":
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            # 这里直接减 0.025
            z_target = z_aruco - 0.025

            self.get_logger().info(f"⬇️ 下落: Z={z_target:.3f} (已偏移 -0.025)")
            joints = self.ik_solver.solve_ik(x, y, z_target)

            if joints:
                self.send_arm(joints)
                self.state = "CLOSE"
                self.wait_ticks = 4
            else:
                self.get_logger().error("❌ 下落 IK 无解")
                self.state = "SEARCHING"

        elif self.state == "CLOSE":
            self.get_logger().info("✊ 闭合")
            self.send_gripper(0.65) 
            self.state = "RETRACT"
            self.wait_ticks = 3 

        elif self.state == "RETRACT":
            self.get_logger().info("⬅️ 收回")
            self.send_arm([1.57, -0.3, 1.57, -0.5])
            self.state = "DONE"
            self.wait_ticks = 4

        elif self.state == "DONE":
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