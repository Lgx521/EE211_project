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
        
        # ==========================================
        # [核心修改区域] 贴纸 -> 物体中心 的修正参数
        # ==========================================
        
        # 1. 左右/前后偏移 (X/Y)
        # 如果贴纸贴在正对你的面上，物体的中心其实在贴纸的“后面”
        # 所以 X 通常需要加一点 (例如 +0.02 表示物体中心在贴纸后方 2cm)
        # 如果贴纸偏左了，你需要往右修 (调整 Y)
        self.OFFSET_X = 0.00   # 前后修正 (米)
        self.OFFSET_Y = 0.00   # 左右修正 (米)

        # 2. 高度偏移 (Z)
        # 之前的 GRASP_HEIGHT_OFFSET 合并到这里
        # 如果贴纸在物体顶面，你要抓下面，这里设为负数 (例如 -0.03)
        self.OFFSET_Z = -0.02  # 上下修正 (负数表示向下抓)

        # 3. 抓取回退距离 (防撞)
        # 计算出中心后，手腕要停在这个距离之外，留出夹爪长度
        self.WRIST_BACKOFF = 0.035 

    def solve_ik(self, x_aruco, y_aruco, z_aruco, debug=False):
        """
        :param x_aruco, y_aruco, z_aruco: ArUco 贴纸的坐标
        """
        
        # --- 第一步：从“贴纸坐标”修正为“物体中心坐标” ---
        x_center = x_aruco + self.OFFSET_X
        y_center = y_aruco + self.OFFSET_Y
        z_center = z_aruco + self.OFFSET_Z
        
        if debug:
            print(f"[IK] 贴纸:({x_aruco:.3f}, {y_aruco:.3f}, {z_aruco:.3f}) -> 中心:({x_center:.3f}, {y_center:.3f}, {z_center:.3f})")

        # --- 第二步：基于物体中心计算 IK ---
        
        # 1. 腰部角度 (瞄准物体中心)
        waist = math.atan2(y_center, x_center)
        
        # 2. 计算平面距离
        r_center = math.sqrt(x_center**2 + y_center**2)
        
        # 3. 手腕目标位置 (Wrist Goal)
        # 我们希望夹爪中心到达物体中心，所以手腕要退后 L4 + Backoff
        # 注意：这里我们假设是用“水平抓取”去对准中心
        r_wrist_target = r_center - self.WRIST_BACKOFF - self.L4
        
        # Z 转换到 Shoulder 坐标系
        z_wrist_target = z_center - self.L1 

        # --- 策略：水平抓取 (Pitch=0) ---
        # 如果物体在地上，可能需要指向抓取，但一般修正了中心后，水平抓取最稳
        joints = self._compute_2link_ik(r_wrist_target, z_wrist_target, pitch_goal=0.0)
        
        if joints:
            return [waist] + joints
        
        # 如果水平抓不到，尝试根据高度自动调整 Pitch
        # (备用逻辑：指向物体中心)
        if debug: print("[IK] 水平不可达，尝试指向中心...")
        angle_to_center = math.atan2(z_wrist_target, r_wrist_target + self.WRIST_BACKOFF) # 指向中心的仰角
        
        # 重新计算手腕位置（带仰角）
        # 这里的几何关系比较复杂，简化处理：让手腕停在连线上
        dist_to_center = math.sqrt(r_center**2 + z_wrist_target**2)
        dist_wrist = dist_to_center - self.L4 - self.WRIST_BACKOFF
        
        r_wrist_p = dist_wrist * math.cos(angle_to_center)
        z_wrist_p = dist_wrist * math.sin(angle_to_center)
        
        joints = self._compute_2link_ik(r_wrist_p, z_wrist_p, pitch_goal=angle_to_center)
        
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
        self.get_logger().info("✅ 节点启动: 带中心偏置修正")

    def cb_poses(self, msg: PoseArray):
        if self.state != "SEARCHING": return
        if len(msg.poses) == 0: return
        try:
            target_cam = PoseStamped()
            target_cam.header = msg.header
            target_cam.pose = msg.poses[0]
            transform = self.tf_buffer.lookup_transform("px100/base_link", target_cam.header.frame_id, rclpy.time.Time())
            self.target_pose_base = do_transform_pose(target_cam.pose, transform)
            self.stable_counter += 1
            if self.stable_counter > 5:
                self.state = "PREPARE_GRASP"
                self.stable_counter = 0
        except Exception: pass

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
            self.get_logger().info("👐 张开夹爪")
            self.send_gripper(1.7) 
            self.state = "CALCULATE"
            self.wait_ticks = 2 

        elif self.state == "CALCULATE":
            if not self.target_pose_base: 
                self.state = "SEARCHING"
                return

            # 直接取 ArUco 原始坐标
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z = self.target_pose_base.position.z

            self.get_logger().info("🧮 计算修正后的 IK...")
            # 偏移修正逻辑现在都在 solve_ik 内部
            joints = self.ik_solver.solve_ik(x, y, z, debug=True)
            
            if joints:
                self.get_logger().info(f"🚀 执行移动")
                self.send_arm(joints)
                self.state = "GRASP_WAIT"
                self.wait_ticks = 6 
            else:
                self.get_logger().error("❌ IK 失败 (修正后目标不可达)")
                self.state = "SEARCHING"

        elif self.state == "GRASP_WAIT":
            self.state = "CLOSE"

        elif self.state == "CLOSE":
            self.get_logger().info("✊ 闭合")
            self.send_gripper(0.65) 
            self.state = "RETRACT_WAIT"
            self.wait_ticks = 3 

        elif self.state == "RETRACT_WAIT":
            self.state = "RETRACT"

        elif self.state == "RETRACT":
            self.get_logger().info("⬅️ 收回")
            self.send_arm([0.0, -0.3, 1.57, -0.5])
            self.state = "DONE"
            self.wait_ticks = 4

        elif self.state == "DONE":
            pass

def main():
    rclpy.init()
    node = ArucoGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()