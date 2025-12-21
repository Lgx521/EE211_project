#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import time
import subprocess
import threading

from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from pan_tilt_msgs.msg import PanTiltCmdDeg
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class PX100Kinematics:
    def __init__(self):
        # 使用 robotics-toolbox 的 px100 内置模型
        import roboticstoolbox as rtb
        from spatialmath import SE3
        self._rtb = rtb
        self._SE3 = SE3
        self._robot = rtb.models.px100()
        self._ee_index = 11
        # 保存上一次解，保证下落时姿态连续（保持水平，yaw不跳变）
        self._last_q = None

    def solve_ik(self, x, y, z, keep_level=True):
        # 保持夹爪水平（pitch=0），只约束 XYZ + Ry；yaw/roll 不约束
        pitch = 0.0 if keep_level else 0.0
        T = self._SE3(x, y, z) * self._SE3.Ry(pitch)
        mask = [1, 1, 1, 0, 1, 0]
        if self._last_q is not None:
            q0 = list(self._last_q)
        else:
            yaw0 = math.atan2(y, x)
            q0 = [yaw0, -0.5, 1.0, -0.5]
        sol = self._robot.ikine_LM(T, q0=q0, end=self._robot[self._ee_index], mask=mask)
        if sol.success:
            self._last_q = sol.q
            return list(sol.q)
        return None


class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        # Publishers
        self.pub_arm = self.create_publisher(JointGroupCommand, "/px100/commands/joint_group", 10)
        self.pub_gripper = self.create_publisher(JointSingleCommand, "/px100/commands/joint_single", 10)
        self.pub_pan_tilt = self.create_publisher(PanTiltCmdDeg, "/pan_tilt_cmd_deg", 10)
        self.pub_grasp_success = self.create_publisher(String, "/grasp/success", 10)

        # Subscribers
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_poses = self.create_subscription(PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)

        self.ik_solver = PX100Kinematics()
        
        # State management
        self.state = "INIT_PAN_TILT"
        self.target_pose_base = None
        self.stable_counter = 0
        self.wait_ticks = 0
        self.timer = self.create_timer(0.5, self.control_loop)
        
        # 云台等待配置
        self.pan_tilt_wait_ticks = 10  # 等待5秒 (10 * 0.5s)
        
        # 高度配置
        self.HOVER_ADD_Z = 0.02
        self.GRASP_ADD_Z = -0.02 
        
        # Aruco detection process
        self.aruco_process = None
        self.aruco_running = False

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

    def move_pan_tilt(self, yaw=0.0, pitch=30.0, speed=5):
        """移动云台到指定角度"""
        msg = PanTiltCmdDeg()
        msg.yaw = yaw
        msg.pitch = pitch
        msg.speed = speed
        self.pub_pan_tilt.publish(msg)
        self.get_logger().info(f"云台移动: yaw={yaw}°, pitch={pitch}°, speed={speed}")

    def start_aruco_detection(self):
        """启动aruco检测脚本"""
        if not self.aruco_running:
            try:
                cmd = ["python3", "/home/tony/ros2_ws/src/EE211_project/detection/aruco_detection_ros.py"]
                self.aruco_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.aruco_running = True
                self.get_logger().info("✅ 启动aruco检测脚本")
            except Exception as e:
                self.get_logger().error(f"❌ 启动aruco检测失败: {e}")

    def stop_aruco_detection(self):
        """停止aruco检测脚本"""
        if self.aruco_running and self.aruco_process:
            try:
                self.aruco_process.terminate()
                self.aruco_process.wait(timeout=5)
                self.get_logger().info("✅ 停止aruco检测脚本")
            except Exception as e:
                self.get_logger().warning(f"⚠️ 停止aruco检测时出错: {e}")
                try:
                    self.aruco_process.kill()
                except:
                    pass
            finally:
                self.aruco_process = None
                self.aruco_running = False

    def publish_grasp_success(self):
        """发布抓取成功话题"""
        msg = String()
        msg.data = "grasp_success"
        self.pub_grasp_success.publish(msg)
        self.get_logger().info("✅ 发布抓取成功话题: /grasp/success")

    def control_loop(self):
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        if self.state == "INIT_PAN_TILT":
            self.get_logger().info("初始化阶段: 设置云台和机械臂初始位置")
            # 初始化云台到0度
            self.move_pan_tilt(yaw=0.0, pitch=30.0, speed=5)
            # 初始化机械臂到安全位置
            # self.send_arm([0.0, -0.5, 1.0, -0.5])
            # 初始化夹爪张开
            self.send_gripper(1.5)
            # 等待初始化完成，然后进入搜索状态
            self.state = "SEARCHING"
            self.get_logger().info("✅ 初始化完成，进入搜索状态")

        elif self.state == "SEARCHING":
            pass 

        elif self.state == "PREPARE_GRASP":
            self.get_logger().info("0. 准备阶段: 开启aruco")
            # 移动云台30度
            # self.move_pan_tilt(yaw=0.0, pitch=30.0, speed=5)
            # 等待云台转动到位
            self.wait_ticks = self.pan_tilt_wait_ticks
            self.state = "WAIT_PAN_TILT"

        elif self.state == "WAIT_PAN_TILT":
            self.get_logger().info("等待云台转动到位...")
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                self.get_logger().info("✅ 云台转动完成，启动aruco检测")
                # 启动aruco检测
                self.start_aruco_detection()
                self.state = "MOVE_ARM_UP"
                self.wait_ticks = 4

        elif self.state == "MOVE_ARM_UP":
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
            # 恢复云台到原位
            self.move_pan_tilt(yaw=0.0, pitch=0.0, speed=5)
            # 停止aruco检测
            self.stop_aruco_detection()
            # 发布抓取成功话题
            self.publish_grasp_success()
            # 清理状态
            self.state = "SEARCHING"

def main():
    rclpy.init()
    node = ArucoGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()