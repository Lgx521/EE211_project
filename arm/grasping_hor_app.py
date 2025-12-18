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
        # 根据CAD图纸修正的参数（单位：米）
        self.L1 = 0.08945  # 基座高度：89.45mm
        self.L2 = 0.10055  # 肩到肘：100.55mm
        self.L3 = 0.100    # 肘到腕：100mm
        self.L4 = 0.12915  # 腕到夹爪末端：129.15mm 
        
        # 水平回退距离（几何IK使用）
        self.GRASP_X_OFFSET = 0.02

        # 优先使用 robotics-toolbox 的数值IK（按用户文档方法）
        self._use_rtb = False
        try:
            import roboticstoolbox as rtb
            from spatialmath import SE3
            self._rtb = rtb
            self._SE3 = SE3
            # 内置 PX-100 模型
            self._robot = rtb.models.px100()
            # ee_gripper_link 的索引（与示例一致）
            self._ee_index = 11
            self._use_rtb = True
            print("[IK] 使用 robotics-toolbox 数值IK (ikine_LM)")
        except Exception as e:
            print(f"[IK] 未检测到 robotics-toolbox，回退到几何IK。原因: {e}")

    def solve_ik(self, x, y, z):
        # 若可用，走 robotics-toolbox 的方法
        if self._use_rtb:
            try:
                # 初始猜测让腰关节面向目标，利于收敛
                waist = math.atan2(y, x)
                q0 = [waist, -0.5, 1.0, -0.5]
                # 仅约束位置 + 工具俯仰（Ry），其余自由
                T = self._SE3(x, y, z)
                mask = [1, 1, 1, 0, 1, 0]
                sol = self._robot.ikine_LM(T, q0=q0, end=self._robot[self._ee_index], mask=mask)
                if sol.success:
                    return list(sol.q)
                else:
                    print("[IK-rtb] 求解失败，回退到几何IK")
            except Exception as e:
                print(f"[IK-rtb] 异常: {e}，回退到几何IK")
        
        # 几何IK回退：保持历史逻辑（水平抓取）
        waist = math.atan2(y, x)
        r_target_raw = math.sqrt(x**2 + y**2)
        r_target = r_target_raw - self.GRASP_X_OFFSET
        z_w = z - self.L1 
        r_wrist = r_target - self.L4
        z_wrist = z_w
        if r_wrist < 0.01:
            print(f"[IK 警告] 目标太近! (Dist={r_target_raw:.3f}), 需要至少 {self.L4 + self.GRASP_X_OFFSET + 0.02:.3f}m")
            return None
        joints = self._compute_2link_ik(r_wrist, z_wrist, pitch_goal=0.0)
        if joints:
            return [waist] + joints
        return None

    def _compute_2link_ik(self, r, z, pitch_goal):
        d = math.sqrt(r**2 + z**2)
        # 检查最大臂展
        if d > (self.L2 + self.L3): 
            # print("[IK] 太远够不着")
            return None
        # 检查最小折叠 (防止自相交)
        if d < 0.05: return None

        alpha = math.atan2(z, r)
        try:
            cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
            if abs(cos_beta) > 1.0: return None
            beta = math.acos(cos_beta)
            # 使用 alpha - beta 让机械臂向下弯曲（肘部向下）
            theta_shoulder = alpha - beta 

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
        
        # 高度配置
        self.HOVER_ADD_Z = 0.02
        self.GRASP_ADD_Z = -0.02 

        self.get_logger().info("✅ 节点启动: 已修复近距离IK翻转BUG")

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
            self.get_logger().info("1. 准备：抬起手臂")
            self.send_gripper(1.5) 
            self.send_arm([0.0, -0.6, 1.2, -0.5]) 
            self.state = "MOVE_HOVER" 
            self.wait_ticks = 4 

        elif self.state == "MOVE_HOVER":
            if not self.target_pose_base: 
                self.state = "SEARCHING"
                return

            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            z_target = z_aruco + self.HOVER_ADD_Z
            
            # Debug: 打印距离
            dist = math.sqrt(x**2 + y**2)
            self.get_logger().info(f"2. 悬停 (Z={z_target:.3f}, Dist={dist:.3f})")
            
            joints = self.ik_solver.solve_ik(x, y, z_target)
            
            if joints:
                self.send_arm(joints)
                self.state = "MOVE_DOWN" 
                self.wait_ticks = 5     
            else:
                self.get_logger().error("❌ 悬停 IK 失败 (物体可能太近)")
                self.state = "SEARCHING"

        elif self.state == "MOVE_DOWN":
            x = self.target_pose_base.position.x
            y = self.target_pose_base.position.y
            z_aruco = self.target_pose_base.position.z

            z_target = z_aruco + self.GRASP_ADD_Z

            self.get_logger().info(f"3. 下落 (Z={z_target:.3f})")
            joints = self.ik_solver.solve_ik(x, y, z_target)

            if joints:
                self.send_arm(joints)
                self.state = "CLOSE"
                self.wait_ticks = 4
            else:
                self.get_logger().error("❌ 下落 IK 失败 (物体太近或太低)")
                self.state = "SEARCHING"

        elif self.state == "CLOSE":
            self.get_logger().info("4. 闭合")
            self.send_gripper(0.65) 
            self.state = "RETRACT"
            self.wait_ticks = 3 

        elif self.state == "RETRACT":
            self.get_logger().info("5. 收回")
            self.send_arm([1.57, -0.3, 1.57, -1.3])
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