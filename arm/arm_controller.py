import rclpy
from rclpy.node import Node
# 必需导入 JointSingleCommand，这是在不改Config情况下控制夹爪的唯一方式
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from sensor_msgs.msg import JointState

class ArmController(Node):
    def __init__(self):
        super().__init__("ArmController")
        
        # 1. Arm 发布器 (使用 Group 指令)
        # 对应 config 中的 groups: arm
        self.pub_arm = self.create_publisher(
            JointGroupCommand, 
            "/px100/commands/joint_group", 
            10
        )
        
        # 2. Gripper 发布器 (使用 Single 指令)
        # ⚠️ 关键修改：对应 config 中的 motors: gripper
        # 必须使用 SingleCommand，因为 config 里没有 gripper 的 group 定义
        self.pub_gripper = self.create_publisher(
            JointSingleCommand, 
            "/px100/commands/joint_single", 
            10
        )

        # 3. 订阅关节状态
        self.sub_states = self.create_subscription(
            JointState, 
            "/px100/joint_states", 
            self.cb_states, 
            10
        )
        
        # 4. 接收外部指令的接口
        self.sub_cmd_arm = self.create_subscription(
            JointGroupCommand, "~/target_joints", self.cb_cmd_arm, 10)
            
        self.sub_cmd_gripper = self.create_subscription(
            JointGroupCommand, "~/target_gripper", self.cb_cmd_gripper, 10)

        # 初始化变量
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.last_arm_cmd = None
        self.last_gripper_cmd = None # 存储为 float 值
        self.joint_positions = []

        self.get_logger().info("Arm controller node initilized")
        self.get_logger().info("Arm Control: JointGroupCommand ('arm')")
        self.get_logger().info("Gripper Control: JointSingleCommand ('gripper')")

        self.move_to_default_position()
        self.get_logger().info("MOVED to default position")


    def cb_states(self, msg):
        """解析关节状态"""
        # 你的 config 顺序是 [waist, shoulder, elbow, wrist_angle, gripper]
        # 我们只取前4个作为 arm 的状态
        if len(msg.position) >= 4:
            self.joint_positions = msg.position[:4]

    def cb_cmd_arm(self, msg: JointGroupCommand):
        """处理 Arm 指令"""
        if msg.name == "arm" and len(msg.cmd) == 4:
            self.last_arm_cmd = msg
            self.get_logger().info(f"收到 Arm 目标: {msg.cmd}")

    def cb_cmd_gripper(self, msg: JointGroupCommand):
        """处理 Gripper 指令"""
        # 虽然输入是 GroupCommand (方便终端发送)，但必须转为 Single
        if msg.name == "gripper" and len(msg.cmd) >= 1:
            self.last_gripper_cmd = msg.cmd[0]
            self.get_logger().info(f"收到 Gripper 目标: {self.last_gripper_cmd} (已转换为单电机指令)")

    def timer_cb(self):
        """控制循环"""
        # 1. 发送 Arm 指令 (Group)
        if self.last_arm_cmd is not None:
            self.pub_arm.publish(self.last_arm_cmd)

        # 2. 发送 Gripper 指令 (Single)
        if self.last_gripper_cmd is not None:
            msg = JointSingleCommand()
            msg.name = "gripper"  # 必须匹配 config -> motors 下的名字
            msg.cmd = self.last_gripper_cmd
            self.pub_gripper.publish(msg)

        # 打印调试信息
        # print(f"Current Joints: {self.joint_positions}")

    def move_to_default_position(self):
        """移动到默认位置"""
        default_position = [0.0, -0.3, 0.8, 1.0]
        msg = JointGroupCommand()
        msg.name = "arm"
        msg.cmd = default_position
        self.pub_arm.publish(msg)
        self.get_logger().info(f"移动到默认位置: {default_position}")

def main():
    rclpy.init()
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()