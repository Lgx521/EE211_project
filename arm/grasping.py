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
    简易的 PX100 逆运动学求解器 (几何法)
    适用于 Interbotix PX100 (4 DOF)
    """
    def __init__(self):
        # 机械臂物理参数 (单位: 米) - 参考 PX100 官方图纸
        self.L1 = 0.0445  # Base height to Shoulder
        self.L2 = 0.100   # Shoulder to Elbow (Humerus)
        self.L3 = 0.100   # Elbow to Wrist (Forearm)
        self.L4 = 0.110   # Wrist to Gripper Tip (需根据实际抓取点调整)

    def solve_ik(self, x, y, z):
        """
        计算 (x, y, z) 对应的 4 个关节角
        目标：垂直向下抓取 (End-effector pitch = -90度)
        """
        # 1. Waist (关节1)
        waist = math.atan2(y, x)

        # 2. 将 3D 问题转换为 2D 平面问题 (r, z)
        # r 是目标点在 XY 平面上的投影距离
        r = math.sqrt(x**2 + y**2)
        
        # 目标手腕位置 (Wrist Position)
        # 我们希望末端垂直向下，所以手腕的位置就在指尖的正上方 L4 处
        # Wrist_r = r
        # Wrist_z = z + L4
        # 转换到 Shoulder 坐标系 (减去 L1 高度)
        z_w = (z + self.L4) - self.L1
        r_w = r
        
        # 3. 计算 Shoulder 和 Elbow (余弦定理)
        # 三角形两边为 L2, L3，第三边(斜边)长度为 d
        d = math.sqrt(r_w**2 + z_w**2)
        
        # 物理限制检查
        if d > (self.L2 + self.L3) or d == 0:
            return None # 目标不可达

        # alpha 是斜边 d 与水平线的夹角
        alpha = math.atan2(z_w, r_w)
        
        # 根据余弦定理计算内角
        cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        if abs(cos_beta) > 1.0: return None
        beta = math.acos(cos_beta)
        
        # Shoulder (关节2) = alpha + beta (注意 PX100 关节定义方向)
        # PX100 Shoulder: 0是直立, 还是水平? 
        # Interbotix定义: 0是直立向上? 不，通常0是水平或者对应URDF。
        # 假设: 0是水平向前. 
        # 在几何计算中，通常计算出的是相对于水平线的角度
        shoulder = (math.pi / 2) - (alpha + beta) 
        # 修正: 上面公式计算的是相对于Z轴的夹角? 
        # 让我们使用标准几何: 
        # theta2 (shoulder) = alpha + beta
        shoulder = alpha + beta
        
        cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
        if abs(cos_gamma) > 1.0: return None
        gamma = math.acos(cos_gamma)
        
        # Elbow (关节3)
        # 几何角度是 gamma, 对应的关节角通常是 - (pi - gamma) 或者根据零位调整
        # 对于 PX100, elbow 0度通常是前臂与大臂垂直? 需要根据实际调整
        # 这里使用标准 Elbow-down configuration
        elbow = -1 * (math.pi - gamma)

        # 4. Wrist Angle (关节4)
        # 我们希望末端垂直向下 (Global Pitch = -pi/2)
        # Global Pitch = Shoulder + Elbow + Wrist
        # -pi/2 = shoulder + elbow + wrist
        # wrist = -pi/2 - shoulder - elbow
        wrist = -math.pi/2 - shoulder - elbow

        return [waist, shoulder, elbow, wrist]

class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        # --- 参数设置 ---
        self.grasp_height_offset = 0.02 # 抓取时，指尖比物体中心低多少(或高多少)
        # 假设方块高度 5cm，中心在 2.5cm。ArUco 在表面(5cm)。
        # 我们希望指尖接触 ArUco 下方一点点
        
        self.gripper_open = 0.06  # 6cm (根据实际物体大小)
        self.gripper_close = 0.025 # 2.5cm (方块宽度)

        # --- 初始化 TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 发布者 (连接到你的 ArmController) ---
        # 注意：你的 ArmController 订阅的是 "~/target_joints"，如果是从外部发
        # 话题通常是 /ArmController/target_joints (取决于节点名)
        self.pub_arm_cmd = self.create_publisher(
            JointGroupCommand, "/ArmController/target_joints", 10)
        
        self.pub_gripper_cmd = self.create_publisher(
            JointGroupCommand, "/ArmController/target_gripper", 10)

        # --- 订阅者 (来自 ArUco Detector) ---
        self.sub_poses = self.create_subscription(
            PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)

        self.ik_solver = PX100Kinematics()
        
        # --- 状态机 ---
        self.state = "SEARCHING" # SEARCHING, ALIGNING, GRASPING, FINISHED
        self.target_pose_base = None # 存储转换后的坐标
        self.stable_counter = 0      # 简单的滤波

        self.get_logger().info("🧲 ArUco Grasp Planner Initialized")
        
        # 定时器用于执行抓取逻辑 (1Hz 慢速逻辑)
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
        """接收 ArUco 位姿并尝试转换到 Base 坐标系"""
        if self.state != "SEARCHING":
            return # 正在抓取时忽略新数据

        if len(msg.poses) == 0:
            return

        # 取第一个检测到的 ArUco
        target_pose_cam = PoseStamped()
        target_pose_cam.header = msg.header
        target_pose_cam.pose = msg.poses[0]

        try:
            # 等待变换关系可用
            # 这里假设 相机 frame 到 px100/base_link 的变换存在
            # 这里的 'px100/base_link' 必须与你的机器人 URDF 一致
            transform = self.tf_buffer.lookup_transform(
                "px100/base_link", 
                target_pose_cam.header.frame_id, 
                rclpy.time.Time())
            
            # 执行坐标变换
            pose_base = do_transform_pose(target_pose_cam.pose, transform)
            
            self.target_pose_base = pose_base
            self.stable_counter += 1
            
            if self.stable_counter > 5: # 连续看到5次才算稳定
                self.get_logger().info(f"🎯 Target Locked at: {pose_base.position.x:.2f}, {pose_base.position.y:.2f}, {pose_base.position.z:.2f}")
                self.state = "READY_TO_PICK"

        except Exception as e:
            self.get_logger().warn(f"TF Transform Error: {e}")
            self.stable_counter = 0

    def control_loop(self):
        """主控逻辑"""
        if self.state == "SEARCHING":
            # 可以让机械臂做一个简单的扫描动作 (这里省略)
            self.send_gripper(self.gripper_open) # 保持张开
            pass

        elif self.state == "READY_TO_PICK":
            if not self.target_pose_base: return
            
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_floor = self.target_pose_base.position.z 
            
            # 1. 移动到物体上方 10cm (Pre-Grasp)
            self.get_logger().info("🚀 Moving to Pre-Grasp...")
            joints = self.ik_solver.solve_ik(x, y, z_floor + 0.10)
            
            if joints:
                self.send_arm_joints(joints)
                self.state = "APPROACH"
            else:
                self.get_logger().error("❌ IK Failed for Pre-Grasp (Out of reach?)")
                self.state = "SEARCHING"
                self.stable_counter = 0

        elif self.state == "APPROACH":
            # 等待机械臂到位 (简单延时逻辑)
            # 2. 下降到抓取高度
            self.get_logger().info("⬇️ Descending...")
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            # 目标 Z = ArUco表面高度 - 一点点偏移(为了抓稳)
            # 注意: 如果 z_floor 是 ArUco 表面，我们需要往下抓一点吗？
            # 假设 ArUco 在方块顶面。我们希望指尖包裹方块两侧。
            # 指尖需要在 ArUco 平面下方。
            z_grasp = self.target_pose_base.position.z - 0.01 
            
            # 地面安全检查 (假设 base_link z=0 是地面)
            if z_grasp < 0.01: z_grasp = 0.01 

            joints = self.ik_solver.solve_ik(x, y, z_grasp)
            if joints:
                self.send_arm_joints(joints)
                self.state = "GRASP"
            else:
                self.state = "SEARCHING"

        elif self.state == "GRASP":
            # 3. 闭合夹爪
            self.get_logger().info("✊ Grasping...")
            self.send_gripper(self.gripper_close)
            self.state = "LIFT"

        elif self.state == "LIFT":
            # 4. 抬起
            self.get_logger().info("⬆️ Lifting...")
            # 回到 Home 姿态 或 抬高
            joints = [0.0, 0.0, 0.0, 0.0] # Sleep pose / Home pose
            # 或者只抬高 Z
            # x = self.target_pose_base.position.x
            # y = self.target_pose_base.position.y
            # joints = self.ik_solver.solve_ik(x, y, 0.2)
            
            self.send_arm_joints(joints)
            self.state = "FINISHED"

        elif self.state == "FINISHED":
            self.get_logger().info("✅ Mission Complete. Resetting in 5s...")
            time.sleep(5)
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