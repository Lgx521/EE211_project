#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import time
import subprocess
import os
import signal

from geometry_msgs.msg import PoseArray, PoseStamped
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import Bool

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
        
        # 准备多组初始猜测
        yaw0 = math.atan2(y, x)
        dist = math.sqrt(x**2 + y**2)
        
        # 根据目标高度和距离，提供更智能的初始猜测
        initial_guesses = []
        
        # 首选：使用上一次的解（如果存在）
        if self._last_q is not None:
            initial_guesses.append(list(self._last_q))
        
        # 备选1：基于目标位置的智能猜测
        # 对于较低的物体，肘部需要更弯曲
        if z < 0.05:  # 物体很低
            initial_guesses.append([yaw0, -0.2, 1.4, -1.2])
        elif z < 0.1:  # 物体中等高度
            initial_guesses.append([yaw0, -0.4, 1.2, -0.8])
        else:  # 物体较高
            initial_guesses.append([yaw0, -0.5, 1.0, -0.5])
        
        # 备选2：标准姿态
        initial_guesses.append([yaw0, -0.5, 1.0, -0.5])
        
        # 备选3：更激进的姿态（用于极端情况）
        initial_guesses.append([yaw0, 0.0, 1.5, -1.5])
        
        # 尝试每一组初始猜测
        for i, q0 in enumerate(initial_guesses):
            sol = self._robot.ikine_LM(
                T, 
                q0=q0, 
                end=self._robot[self._ee_index], 
                mask=mask,
                ilimit=500,      # 增加迭代次数限制（默认约100）
                slimit=200,      # 增加搜索步数限制
                tol=1e-5         # 稍微放宽收敛容差（默认1e-6）
            )
            if sol.success:
                self._last_q = sol.q
                if i > 0:  # 如果不是第一次尝试就成功了
                    print(f"✓ IK求解成功（尝试#{i+1}）")
                return list(sol.q)
        
        # 所有尝试都失败
        return None


class ArucoGraspNode(Node):
    def __init__(self):
        super().__init__("aruco_grasp_node")
        
        self.pub_arm = self.create_publisher(JointGroupCommand, "/px100/commands/joint_group", 10)
        self.pub_gripper = self.create_publisher(JointSingleCommand, "/px100/commands/joint_single", 10)
        # 发布抓取成功话题
        self.pub_grasp_success = self.create_publisher(Bool, "/grasp/success", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub_poses = self.create_subscription(PoseArray, "/aruco_detector/marker_poses", self.cb_poses, 10)
        
        # 订阅启动话题
        self.sub_start = self.create_subscription(Bool, "/grasp/start", self.cb_start, 10)

        self.ik_solver = PX100Kinematics()
        
        # 等待启动信号
        self.started = False
        self.state = "WAITING_START"
        self.target_pose_base = None
        self.stable_counter = 0
        self.wait_ticks = 0
        self.timer = self.create_timer(0.5, self.control_loop)
        
        # 高度配置
        self.HOVER_ADD_Z = 0.02
        self.GRASP_ADD_Z = -0.02
        
        # ArUco检测进程
        self.aruco_process = None

        self.get_logger().info("✅ 节点启动: 等待 /grasp/start 话题中的 True 指令...")

    def cb_start(self, msg: Bool):
        """接收启动信号"""
        if msg.data and not self.started:
            self.get_logger().info("🚀 收到启动信号，开始执行抓取任务...")
            self.started = True
            # 执行抓取前准备
            self.setup_before_grasp()
            # 切换到搜索状态
            self.state = "SEARCHING"

    def cb_poses(self, msg: PoseArray):
        if not self.started: return  # 未收到启动信号，不处理
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

    def setup_before_grasp(self):
        """抓取前准备：云台转动30度并启动ArUco检测"""
        self.get_logger().info("🔧 抓取前准备：云台转动 + ArUco检测启动")
        
        # 1. 云台转动到30度
        try:
            cmd = 'ros2 topic pub --once /pan_tilt_cmd_deg pan_tilt_msgs/msg/PanTiltCmdDeg "{yaw: 0.0, pitch: 30.0, speed: 20}"'
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("✅ 云台转动命令已发送: pitch=30°")
        except Exception as e:
            self.get_logger().error(f"❌ 云台转动失败: {e}")
        
        # 等待云台转动到位
        time.sleep(4.0)
        
        # 2. 启动ArUco检测脚本
        try:
            # 获取当前脚本所在目录的上级目录（项目根目录）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            aruco_script = os.path.join(project_root, "detection", "aruco_detection_ros.py")
            
            # 使用subprocess.Popen启动ArUco检测（非阻塞）
            self.aruco_process = subprocess.Popen(
                ["python3", aruco_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # 创建新的进程组，便于后续终止
            )
            self.get_logger().info(f"✅ ArUco检测已启动 (PID: {self.aruco_process.pid})")
        except Exception as e:
            self.get_logger().error(f"❌ ArUco检测启动失败: {e}")
        
        # 等待ArUco检测初始化
        time.sleep(2.0)

    def cleanup_after_grasp(self):
        """抓取后清理：云台恢复原位并关闭ArUco检测"""
        self.get_logger().info("🧹 抓取后清理：云台恢复 + ArUco检测关闭")
        
        # 1. 关闭ArUco检测进程
        if self.aruco_process is not None:
            try:
                # 终止整个进程组
                os.killpg(os.getpgid(self.aruco_process.pid), signal.SIGTERM)
                self.aruco_process.wait(timeout=3.0)
                self.get_logger().info("✅ ArUco检测已关闭")
            except Exception as e:
                self.get_logger().warn(f"⚠️ ArUco检测关闭异常: {e}")
                try:
                    # 强制终止
                    os.killpg(os.getpgid(self.aruco_process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self.aruco_process = None
        
        # 2. 云台恢复到原位（pitch=0）
        try:
            cmd = 'ros2 topic pub --once /pan_tilt_cmd_deg pan_tilt_msgs/msg/PanTiltCmdDeg "{yaw: 0.0, pitch: 0.0, speed: 20}"'
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("✅ 云台恢复命令已发送: pitch=0°")
        except Exception as e:
            self.get_logger().error(f"❌ 云台恢复失败: {e}")
        
        # 3. 发布抓取成功话题
        success_msg = Bool()
        success_msg.data = True
        self.pub_grasp_success.publish(success_msg)
        self.get_logger().info("✅ 抓取成功话题已发布")

    def control_loop(self):
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return

        if self.state == "WAITING_START":
            # 等待启动信号，不执行任何操作
            pass

        elif self.state == "SEARCHING":
            pass 

        elif self.state == "PREPARE_GRASP":
            self.get_logger().info("1. 准备：抬起手臂")
            self.send_gripper(1.57) 
            self.send_arm([0.0, -0.3, 0.48, 0.0]) 
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
            self.send_gripper(0.62) 
            self.state = "RETRACT"
            self.wait_ticks = 3 

        elif self.state == "RETRACT":
            self.get_logger().info("5. 收回")
            self.send_arm([1.57, -0.3, 1.57, -1.3])
            self.state = "DONE"
            self.wait_ticks = 4

        elif self.state == "DONE":
            self.get_logger().info("✅ 抓取完成")
            # 执行抓取后清理
            self.cleanup_after_grasp()
            self.state = "FINISHED"
            
        elif self.state == "FINISHED":
            # 保持在完成状态，不再执行任何操作
            pass

def main():
    rclpy.init()
    node = ArucoGraspNode()
    
    # 运行主程序（等待 /grasp/start 话题）
    rclpy.spin(node)
    
    # 清理资源
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()