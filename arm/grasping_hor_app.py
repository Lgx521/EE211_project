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
        # 机械臂连杆长度
        self.L1 = 0.0445
        self.L2 = 0.100
        self.L3 = 0.100
        self.L4 = 0.110 
        
        # [参数] 水平方向回退距离
        # 水平抓取时，这个值决定了夹爪尖端距离物体中心的距离
        self.GRASP_X_OFFSET = 0.035 

    def solve_ik(self, x, y, z):
        """
        只计算水平抓取 (Pitch = 0) 的逆运动学解
        """
        # 1. 腰部角度 (Waist)
        waist = math.atan2(y, x)
        
        # 2. 坐标转换 (从 Base 到 Shoulder)
        # 水平距离 r
        r_target_raw = math.sqrt(x**2 + y**2)
        r_target = r_target_raw - self.GRASP_X_OFFSET # 给夹爪留出空间
        
        # 垂直高度 z_w (相对于 Shoulder 关节)
        z_w = z - self.L1 

        # 3. 计算关节角
        # --- 强制策略: 仅水平抓取 (Pitch = 0) ---
        # 此时 Wrist 的位置很好算：就在目标点后方 L4 处
        r_wrist = r_target - self.L4
        z_wrist = z_w
        
        # 调用 2-Link IK 解算 Shoulder 和 Elbow
        joints = self._compute_2link_ik(r_wrist, z_wrist, pitch_goal=0.0)
        
        if joints:
            return [waist] + joints
        
        # 如果水平够不到，直接返回失败，绝不尝试 30 度
        return None

    def _compute_2link_ik(self, r, z, pitch_goal):
        d = math.sqrt(r**2 + z**2)
        # 物理限位检查
        if d > (self.L2 + self.L3) or d < 0.02: return None

        alpha = math.atan2(z, r)
        try:
            # 余弦定理算 Elbow
            cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
            if abs(cos_beta) > 1.0: return None
            beta = math.acos(cos_beta)
            theta_shoulder = alpha + beta # Elbow Up 构型

            # 余弦定理算 Shoulder
            cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
            if abs(cos_gamma) > 1.0: return None
            gamma = math.acos(cos_gamma)

            theta_elbow = -1 * (math.pi - gamma)
            
            # 关键：Pitch = Shoulder + Elbow + Wrist
            # 所以 Wrist = Pitch_Goal - Shoulder - Elbow
            theta_wrist = pitch_goal - theta_shoulder - theta_elbow

            # 关节角限位
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
        
        # --- 高度配置 ---
        self.HOVER_ADD_Z = 0.03  # 悬停高度: 比物体高 6cm
        self.GRASP_ADD_Z = -0.025 # 抓取高度: 比物体低 2.5cm (下压)

        self.get_logger().info("✅ 节点启动: 强制水平抓取 (Pitch=0)")

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
            self.get_logger().info("1. 准备姿态：抬起手臂")
            self.send_gripper(1.5) 
            
            # 先移动到一个安全的高位 (Sleep 姿态的变体，抬高一点)
            # 防止从低处直接铲过去
            self.send_arm([0.0, -0.6, 1.2, -0.5]) 
            
            self.state = "MOVE_HOVER" 
            self.wait_ticks = 4 

        # --- 步骤 2: 保持水平姿态，移动到物体上方 ---
        elif self.state == "MOVE_HOVER":
            if not self.target_pose_base: 
                self.state = "SEARCHING"
                return

            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            # 悬停目标 Z
            z_target = z_aruco + self.HOVER_ADD_Z

            self.get_logger().info(f"2. 水平悬停 (Pitch=0, Z={z_target:.3f})")
            joints = self.ik_solver.solve_ik(x, y, z_target)
            
            if joints:
                self.send_arm(joints)
                self.state = "MOVE_DOWN" 
                self.wait_ticks = 5     
            else:
                self.get_logger().error("❌ 悬停点 IK 无解 (可能太远或太高)")
                self.state = "SEARCHING"

        # --- 步骤 3: 保持水平姿态，垂直下降 ---
        elif self.state == "MOVE_DOWN":
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            # 抓取目标 Z
            z_target = z_aruco + self.GRASP_ADD_Z

            self.get_logger().info(f"3. 垂直下落 (Pitch=0, Z={z_target:.3f})")
            joints = self.ik_solver.solve_ik(x, y, z_target)

            if joints:
                self.send_arm(joints)
                self.state = "CLOSE"
                self.wait_ticks = 4
            else:
                self.get_logger().error("❌ 下落点 IK 无解 (可能太低)")
                self.state = "SEARCHING"

        elif self.state == "CLOSE":
            self.get_logger().info("4. 闭合夹爪")
            self.send_gripper(0.65) 
            self.state = "RETRACT"
            self.wait_ticks = 3 

        elif self.state == "RETRACT":
            self.get_logger().info("5. 抬起收回")
            self.send_arm([1.57, -0.3, 1.57, -1.3])
            self.state = "DONE"
            self.wait_ticks = 4

        elif self.state == "DONE":
            self.get_logger().info("✅ 抓取完成")
            # self.state = "SEARCHING"

def main():
    rclpy.init()
    node = ArucoGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()