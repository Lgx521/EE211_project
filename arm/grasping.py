#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time

from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class PX100Kinematics:
    """
    PX100 逆运动学求解器 (斜向抓取版)
    目标：以与铅垂线成 30 度夹角的方式抓取 (Pitch = -60 度)
    """
    def __init__(self):
        # 机械臂物理参数 (单位: 米)
        self.L1 = 0.0445  # Base -> Shoulder
        self.L2 = 0.100   # Shoulder -> Elbow
        self.L3 = 0.100   # Elbow -> Wrist
        self.L4 = 0.110   # Wrist -> Tip (包含夹爪长度)

    def solve_ik_slanted(self, x, y, z, angle_from_vertical_deg=30):
        """
        计算斜向抓取的 IK
        :param angle_from_vertical_deg: 与铅垂线的夹角 (默认30度)
                                        0度 = 垂直向下抓
                                        90度 = 水平向前抓
        """
        # 1. 计算目标 Pitch 角度 (弧度)
        # 垂直向下是 -90度。我们要抬起 30度，所以是 -60度。
        pitch_deg = -90 + angle_from_vertical_deg
        pitch_rad = math.radians(pitch_deg)

        # 2. Waist (腰部关节) - 始终指向目标方向
        waist = math.atan2(y, x)

        # 3. 计算手腕 (Wrist) 的目标位置
        # 我们已知指尖(Tip)的坐标 (x,y,z)，需要反推手腕在哪里。
        # r_target 是指尖在平面上的投影半径
        r_target = math.sqrt(x**2 + y**2)
        
        # 将 Z 转换到 Shoulder 坐标系 (减去底座高度)
        z_target_shoulder = z - self.L1

        # 反推手腕坐标 (r_wrist, z_wrist)
        # Wrist = Tip - Vector_of_Gripper
        r_wrist = r_target - self.L4 * math.cos(pitch_rad)
        z_wrist = z_target_shoulder - self.L4 * math.sin(pitch_rad)

        print(f"[IK] 目标: ({r_target:.3f}, {z_target_shoulder:.3f}) | 手腕应在: ({r_wrist:.3f}, {z_wrist:.3f}) | Pitch: {pitch_deg}°")

        # 4. 解算 2-Link IK (Shoulder, Elbow)
        # 三角形两边 L2, L3，斜边 d 是从 Shoulder(0,0) 到 Wrist(r_wrist, z_wrist)
        d = math.sqrt(r_wrist**2 + z_wrist**2)

        # 物理限制检查
        if d > (self.L2 + self.L3):
            print("[IK Error] 目标太远，手臂够不着")
            return None
        if d < 0.02:
            print("[IK Error] 目标太近 (在底座内)")
            return None

        # alpha: 手腕向量的角度
        alpha = math.atan2(z_wrist, r_wrist)

        # Cosine Law for Beta (肩部内角)
        cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        # 浮点数误差保护
        if cos_beta > 1.0: cos_beta = 1.0
        if cos_beta < -1.0: cos_beta = -1.0
        beta = math.acos(cos_beta)

        # --- 强制 Elbow Up (吊车姿态) ---
        # Shoulder = alpha + beta (向上抬起)
        shoulder = alpha + beta

        # Cosine Law for Gamma (肘部内角)
        cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
        if cos_gamma > 1.0: cos_gamma = 1.0
        if cos_gamma < -1.0: cos_gamma = -1.0
        gamma = math.acos(cos_gamma)

        # Elbow Angle
        # 几何修正: elbow = - (pi - gamma)
        elbow = -1 * (math.pi - gamma)

        # 5. Wrist Angle (关节4)
        # Global Pitch = Shoulder + Elbow + Wrist
        # Wrist = Goal_Pitch - (Shoulder + Elbow)
        wrist = pitch_rad - (shoulder + elbow)

        return [waist, shoulder, elbow, wrist]


class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        # Publishers
        self.pub_arm = self.create_publisher(JointGroupCommand, "/px100/commands/joint_group", 10)
        self.pub_gripper = self.create_publisher(JointSingleCommand, "/px100/commands/joint_single", 10)

        # TF & Subscribers
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_poses = self.create_subscription(PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)

        self.ik_solver = PX100Kinematics()
        
        self.state = "SEARCHING"
        self.target_pose_base = None
        self.stable_cnt = 0
        
        # 参数: 抓取角度 (与铅垂线夹角)
        self.GRASP_ANGLE_FROM_VERTICAL = 30 # degrees
        
        self.timer = self.create_timer(0.5, self.control_loop)
        self.get_logger().info(f"✅ 斜向抓取节点启动 (倾角 {self.GRASP_ANGLE_FROM_VERTICAL}°)")

    def cb_poses(self, msg: PoseArray):
        if self.state != "SEARCHING": return
        if not msg.poses: return

        try:
            target_cam = PoseStamped()
            target_cam.header = msg.header
            target_cam.pose = msg.poses[0]

            transform = self.tf_buffer.lookup_transform("px100/base_link", target_cam.header.frame_id, rclpy.time.Time())
            pose_base = do_transform_pose(target_cam.pose, transform)
            
            self.target_pose_base = pose_base
            self.stable_cnt += 1
            
            # 简单滤波：连续看到5次
            if self.stable_cnt > 5:
                self.get_logger().info(f"🎯 锁定目标: X={pose_base.position.x:.2f}, Y={pose_base.position.y:.2f}, Z={pose_base.position.z:.2f}")
                self.state = "CALCULATE"
                
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
        if self.state == "SEARCHING":
            self.send_gripper(1.0) # Open

        elif self.state == "CALCULATE":
            if not self.target_pose_base:
                self.state = "SEARCHING"
                return

            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z = self.target_pose_base.position.z
            
            # 修正: 假设 ArUco 在物体表面，我们要抓取中心
            # 30度斜着抓，我们需要让指尖稍微深入物体一点点，或者对准中心
            # 这里不做过多修正，直接抓 TF 给出的点
            
            self.get_logger().info("🧮 计算斜向 IK...")
            joints = self.ik_solver.solve_ik_slanted(x, y, z, self.GRASP_ANGLE_FROM_VERTICAL)
            
            if joints:
                self.get_logger().info("🚀 移动到预备位置")
                # 策略: 先发指令，等待时间稍长一点确保到位
                self.send_arm(joints)
                self.state = "WAIT_FOR_MOVE"
            else:
                self.get_logger().error("❌ 无法到达 (尝试移动物体或调整角度)")
                self.state = "SEARCHING"
                self.stable_cnt = 0

        elif self.state == "WAIT_FOR_MOVE":
            time.sleep(3.0)
            self.state = "CLOSE"

        elif self.state == "CLOSE":
            self.get_logger().info("✊ 抓取")
            self.send_gripper(-0.5) # Close
            time.sleep(1.5)
            self.state = "LIFT"

        elif self.state == "LIFT":
            self.get_logger().info("⬆️ 抬起")
            # 抬起时，保持当前关节，只动 Shoulder 或使用 Home
            self.send_arm([0.0, -1.5, 1.2, 0.5]) # Sleep/Carry pose
            time.sleep(2.0)
            self.state = "DONE"

        elif self.state == "DONE":
            self.get_logger().info("✅ 完成，重置中...")
            time.sleep(3.0)
            self.send_gripper(1.0)
            self.state = "SEARCHING"
            self.stable_cnt = 0

def main():
    rclpy.init()
    node = ArucoGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()