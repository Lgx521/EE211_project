#!/usr/bin/env python3
"""
最终可运行版（ROS2 Humble 兼容）
核心：从 /received_global_plan 提取目标位姿，实现停止/恢复导航
适配你的 TurtleBot4 环境，无导入错误
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool

# 手动定义 GoalStatus 常量（兼容 ROS2 Humble）
GOAL_STATUS_SUCCEEDED = 4
GOAL_STATUS_CANCELED = 2
GOAL_STATUS_ABORTED = 5

class Nav2GoalController(Node):
    def __init__(self):
        super().__init__('nav2_goal_controller')
        
        # 核心状态
        self.stop_flag = False                  # 停止标志
        self.current_goal_handle = None         # Action句柄（无类型注解）
        self.saved_goal_pose = None             # 保存的目标位姿
        self.is_navigating = False              # 导航状态
        
        # 1. 监听全局规划（核心！提取目标）
        self.global_plan_sub = self.create_subscription(
            Path,
            '/received_global_plan',  # 你的核心话题
            self.global_plan_callback,
            10
        )
        
        # 2. 监听RViz目标（补充）
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )
        
        # 3. 监听停止控制指令
        self.stop_sub = self.create_subscription(
            Bool,
            'traffic_light/stop_control',
            self.stop_control_callback,
            10
        )
        
        # 4. 创建Action客户端（兼容Humble）
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 等待Action Server就绪
        self.get_logger().info("等待Nav2 Action Server (/navigate_to_pose)...")
        if not self.action_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("Nav2 Action Server 未启动！请先启动Nav2")
            rclpy.shutdown()
            return
        
        self.get_logger().info("✅ 初始化完成！")
        self.get_logger().info("使用说明：")
        self.get_logger().info("1. RViz发送Nav2 Goal → 自动提取目标（/received_global_plan）")
        self.get_logger().info("2. 停止：ros2 topic pub /traffic_light/stop_control std_msgs/msg/Bool '{data: true}' --once")
        self.get_logger().info("3. 恢复：ros2 topic pub /traffic_light/stop_control std_msgs/msg/Bool '{data: false}' --once")

    def global_plan_callback(self, msg: Path):
        """从 /received_global_plan 提取目标位姿（最后一个路径点）"""
        if len(msg.poses) == 0:
            self.get_logger().warn("⚠️ /received_global_plan 路径为空")
            return
        
        # 核心：最后一个路径点 = Nav2的目标位姿
        goal_pose_stamped = msg.poses[-1]
        self.saved_goal_pose = goal_pose_stamped
        
        # 日志输出目标信息（验证）
        self.get_logger().info("\n🎯 从全局规划提取到目标：")
        self.get_logger().info(f"   位置：x={goal_pose_stamped.pose.position.x:.2f}, y={goal_pose_stamped.pose.position.y:.2f}")
        self.get_logger().info(f"   坐标系：{msg.header.frame_id}")
        
        # 非停止状态下自动启动导航
        if not self.stop_flag and not self.is_navigating:
            self.get_logger().info("🚀 自动发送目标到Nav2执行导航")
            self.send_nav_goal(goal_pose_stamped)

    def goal_pose_callback(self, msg: PoseStamped):
        """补充：监听RViz直接发送的目标"""
        self.get_logger().info(f"\n📥 收到RViz直接发送的目标：x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")
        self.saved_goal_pose = msg

    def stop_control_callback(self, msg: Bool):
        """处理停止/恢复指令（核心逻辑）"""
        new_stop_flag = msg.data
        
        # 忽略重复指令
        if new_stop_flag == self.stop_flag:
            return
        
        self.stop_flag = new_stop_flag
        
        if self.stop_flag:
            self.get_logger().info("\n🛑 收到停止指令")
            # 取消当前活跃的导航
            if self.current_goal_handle and self.is_navigating:
                self.current_goal_handle.cancel_goal_async()
                self.get_logger().info("🛑 已取消当前导航任务")
                self.is_navigating = False
            # 确认目标已保存
            if self.saved_goal_pose:
                self.get_logger().info(f"📌 目标已保存：x={self.saved_goal_pose.pose.position.x:.2f}, y={self.saved_goal_pose.pose.position.y:.2f}")
            else:
                self.get_logger().warn("⚠️ 无目标可保存（未收到全局规划）")
        else:
            self.get_logger().info("\n▶️ 收到恢复指令")
            # 恢复导航（使用保存的目标）
            if self.saved_goal_pose:
                self.get_logger().info(f"🚗 恢复导航到目标：x={self.saved_goal_pose.pose.position.x:.2f}, y={self.saved_goal_pose.pose.position.y:.2f}")
                self.send_nav_goal(self.saved_goal_pose)
            else:
                self.get_logger().error("❌ 无保存的目标！请先在RViz发送Nav2 Goal（触发全局规划）")

    def send_nav_goal(self, pose: PoseStamped):
        """发送目标到Nav2的 navigate_to_pose Action Server"""
        # 先取消已有导航（避免冲突）
        if self.current_goal_handle and self.is_navigating:
            self.get_logger().info("🔄 取消已有导航，执行新目标")
            self.current_goal_handle.cancel_goal_async()
        
        # 停止状态下不发送目标
        if self.stop_flag:
            self.get_logger().warn("⚠️ 当前为停止状态，拒绝发送目标")
            return
        
        # 构建Nav2 Action目标消息
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        # 标记为导航中
        self.is_navigating = True
        
        # 异步发送目标
        self.get_logger().info(f"\n📤 发送目标到Nav2：x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}")
        send_future = self.action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """处理Nav2对目标的响应（接受/拒绝）"""
        try:
            # 移除GoalHandle类型注解（兼容Humble）
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error("❌ 目标被Nav2拒绝！请检查Nav2状态")
                self.is_navigating = False
                return
            
            self.get_logger().info("✅ 目标已被Nav2接受，开始导航")
            self.current_goal_handle = goal_handle
            
            # 监听导航结果
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.goal_result_callback)
            
        except Exception as e:
            self.get_logger().error(f"❌ 处理目标响应失败：{str(e)}")
            self.is_navigating = False

    def goal_result_callback(self, future):
        """处理导航完成/失败结果"""
        try:
            result = future.result()
            self.is_navigating = False
            self.current_goal_handle = None
            
            # 解析导航状态（使用手动定义的常量）
            status_code = result.status
            if status_code == GOAL_STATUS_SUCCEEDED:
                self.get_logger().info("\n🎉 导航成功！已到达目标位置")
            elif status_code == GOAL_STATUS_CANCELED:
                self.get_logger().info("\n⏹️ 导航已被取消（停止指令触发）")
            else:
                self.get_logger().warn(f"\n❌ 导航失败（状态码：{status_code}）")
                # 自动重试（非停止状态且有目标）
                if not self.stop_flag and self.saved_goal_pose:
                    self.get_logger().info("🔄 自动重试导航到目标...")
                    self.send_nav_goal(self.saved_goal_pose)
                    
        except Exception as e:
            self.get_logger().error(f"❌ 处理导航结果失败：{str(e)}")

def main(args=None):
    """主函数（兼容ROS2 Humble）"""
    rclpy.init(args=args)
    
    # 创建节点
    node = Nav2GoalController()
    
    try:
        # 自旋节点
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n🛑 用户中断程序，正在退出...")
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()