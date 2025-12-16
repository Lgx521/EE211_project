#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time

# 消息类型
from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class PX100Kinematics:
    """
    修改版 PX100 逆运动学求解器 (几何法)
    适用于 Interbotix PX100 (4 DOF)
    已调整为：水平向前抓取
    """
    def __init__(self):
        # 机械臂物理参数 (单位: 米) - 基于 CAD 图纸
        self.L1 = 0.0445  # Base height to Shoulder
        self.L2 = 0.100   # Shoulder to Elbow (Humerus)
        self.L3 = 0.100   # Elbow to Wrist (Forearm)
        # Wrist joint to Gripper Tip 
        # 根据 CAD, 腕关节中心到法兰是 43.2+20=63.2，再加上手指长度
        # 110mm 是一个合理的经验值
        self.L4 = 0.110   

    def solve_ik(self, x, y, z):
        """
        计算 (x, y, z) 对应的 4 个关节角
        目标：水平向前抓取 (End-effector pitch = 0度)
        """
        # 1. Waist (关节1)
        waist = math.atan2(y, x)

        # 2. 将 3D 问题转换为 2D 平面问题 (r, z)
        # r 是目标点在 XY 平面上的投影距离 (圆柱坐标系下的半径)
        r_target = math.sqrt(x**2 + y**2)
        
        # --- 核心修改开始 ---
        # 目标：水平抓取 (Pitch = 0)
        # 手腕位置 (Wrist) 应该在目标点的 "后面" L4 距离处
        # Wrist_r = Target_r - L4
        # Wrist_z = Target_z (高度与目标一致)
        
        r_w = r_target - self.L4
        z_w = z - self.L1 # 转换到 Shoulder 坐标系 (减去底座高度)

        # 物理限制检查：如果目标太近，减去 L4 后 r_w 变成负数，说明目标在肚子里，不可达
        if r_w < 0.02: 
            return None 

        # --- 核心修改结束 ---
        
        # 3. 计算 Shoulder 和 Elbow (几何法 - 余弦定理)
        # 三角形两边为 L2, L3，第三边(斜边)长度为 d
        d = math.sqrt(r_w**2 + z_w**2)
        
        # 物理限制检查 (臂展不够)
        if d > (self.L2 + self.L3):
            return None 

        # alpha 是斜边 d 与水平线的夹角
        alpha = math.atan2(z_w, r_w)
        
        # 根据余弦定理计算内角 beta (肩部三角形内角)
        cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        if abs(cos_beta) > 1.0: return None
        beta = math.acos(cos_beta)
        
        # Shoulder (关节2)
        # 几何上 theta_shoulder = alpha + beta
        shoulder = alpha + beta
        
        # 根据余弦定理计算内角 gamma (肘部三角形内角)
        cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
        if abs(cos_gamma) > 1.0: return None
        gamma = math.acos(cos_gamma)
        
        # Elbow (关节3)
        # 几何上通常是负的补角
        elbow = -1 * (math.pi - gamma)

        # 4. Wrist Angle (关节4)
        # 我们希望末端水平 (Global Pitch = 0)
        # Global Pitch = Shoulder_Angle + Elbow_Angle + Wrist_Angle
        # 0 = shoulder + elbow + wrist
        # wrist = - (shoulder + elbow)
        wrist = -(shoulder + elbow)

        return [waist, shoulder, elbow, wrist]

class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        # --- 参数设置 ---
        self.gripper_open = 0.06  # 张开
        self.gripper_close = 0.024 # 闭合 (根据物体调整)
        
        # 水平抓取时，approach_distance 代表抓取前缩回的距离
        self.approach_dist = 0.05 # 5cm

        # --- 初始化 TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 发布者 ---
        self.pub_arm_cmd = self.create_publisher(
            JointGroupCommand, "/ArmController/target_joints", 10)
        self.pub_gripper_cmd = self.create_publisher(
            JointGroupCommand, "/ArmController/target_gripper", 10)

        # --- 订阅者 ---
        self.sub_poses = self.create_subscription(
            PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)

        self.ik_solver = PX100Kinematics()
        
        # --- 状态机 ---
        self.state = "SEARCHING" 
        self.target_pose_base = None 
        self.stable_counter = 0      

        self.get_logger().info("📐 Horizontal Grasp Planner Initialized")
        
        # 定时器
        self.control_timer = self.create_timer(1.0, self.control_loop)

    def send_arm_joints(self, joints):
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = joints
        self.pub_arm_cmd.publish(msg)

    def send_gripper(self, val):
        msg = JointGroupCommand()
        msg.name = "gripper"
        msg.cmd = [float(val)]
        self.pub_gripper_cmd.publish(msg)

    def cb_poses(self, msg: PoseArray):
        if self.state != "SEARCHING":
            return 

        if len(msg.poses) == 0:
            return

        target_pose_cam = PoseStamped()
        target_pose_cam.header = msg.header
        target_pose_cam.pose = msg.poses[0]

        try:
            # 确保转换到 px100/base_link
            transform = self.tf_buffer.lookup_transform(
                "px100/base_link", 
                target_pose_cam.header.frame_id, 
                rclpy.time.Time())
            
            pose_base = do_transform_pose(target_pose_cam.pose, transform)
            
            self.target_pose_base = pose_base
            self.stable_counter += 1
            
            if self.stable_counter > 5: 
                self.get_logger().info(f"🎯 Target Found: ({pose_base.position.x:.2f}, {pose_base.position.y:.2f}, {pose_base.position.z:.2f})")
                self.state = "READY_TO_PICK"

        except Exception as e:
            self.stable_counter = 0

    def control_loop(self):
        """主控逻辑 - 水平抓取版"""
        if self.state == "SEARCHING":
            self.send_gripper(self.gripper_open)

        elif self.state == "READY_TO_PICK":
            if not self.target_pose_base: return
            
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z = self.target_pose_base.position.z
            
            # --- 策略修改：计算预抓取点 (Pre-Grasp) ---
            # 水平抓取：预抓取点在目标点连接底座的连线上，向后退 5cm
            # 利用相似三角形原理计算
            r_total = math.sqrt(x**2 + y**2)
            if r_total == 0: return

            scale = (r_total - self.approach_dist) / r_total
            x_pre = x * scale
            y_pre = y * scale
            z_pre = z # 高度不变

            self.get_logger().info("🔙 Aligning (Pre-Grasp)...")
            joints = self.ik_solver.solve_ik(x_pre, y_pre, z_pre)
            
            if joints:
                self.send_arm_joints(joints)
                self.state = "APPROACH"
            else:
                self.get_logger().error("❌ IK Failed (Target too close/far?)")
                self.state = "SEARCHING"
                self.stable_counter = 0

        elif self.state == "APPROACH":
            # --- 策略修改：执行抓取 (Forward) ---
            self.get_logger().info("👉 Inserting...")
            
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z = self.target_pose_base.position.z
            
            # 直接解算目标点 IK
            joints = self.ik_solver.solve_ik(x, y, z)
            
            if joints:
                self.send_arm_joints(joints)
                self.state = "GRASP"
            else:
                self.get_logger().error("❌ IK Failed at Final Reach")
                self.state = "SEARCHING"

        elif self.state == "GRASP":
            self.get_logger().info("✊ Grasping...")
            self.send_gripper(self.gripper_close)
            self.state = "LIFT"

        elif self.state == "LIFT":
            # 抓到后，最好向上抬一点，而不是直接回零，避免打到桌子
            self.get_logger().info("⬆️ Lifting...")
            
            # 保持当前 XY，只增加 Z
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_lift = self.target_pose_base.position.z + 0.10 # 抬高10cm
            
            # 这里要注意，如果抬太高可能会超出机械臂工作空间
            # 如果解算失败，就直接回 Home
            joints = self.ik_solver.solve_ik(x, y, z_lift)
            
            if not joints:
                joints = [0.0, -1.0, 1.0, 0.0] # 一个安全的中间姿态

            self.send_arm_joints(joints)
            self.state = "FINISHED"

        elif self.state == "FINISHED":
            self.get_logger().info("✅ Done. Resetting...")
            time.sleep(4)
            self.send_gripper(self.gripper_open)
            self.state = "SEARCHING"
            self.stable_counter = 0

def main(args=None):
    rclpy.init(args=args)
    node = ArucoGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()