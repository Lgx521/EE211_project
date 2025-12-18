#!/bin/bash

# 机械臂抓取系统启动脚本

echo "========================================="
echo "  PX100 机械臂抓取系统启动脚本"
echo "========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}错误: ROS2环境未设置${NC}"
    echo "请先运行: source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo -e "${GREEN}✓ ROS2环境: $ROS_DISTRO${NC}"

# 功能选择
echo ""
echo "请选择功能:"
echo "  1) 运行运动学测试"
echo "  2) 启动抓取节点"
echo "  3) 查看ArUco话题"
echo "  4) 查看机械臂状态"
echo "  5) 测试夹爪控制"
echo "  6) 退出"
echo ""
read -p "请输入选项 [1-6]: " choice

case $choice in
    1)
        echo -e "${YELLOW}运行运动学测试...${NC}"
        python3 test_kinematics.py
        ;;
    2)
        echo -e "${YELLOW}启动抓取节点...${NC}"
        echo ""
        echo "请确保以下节点已启动:"
        echo "  1. ArUco检测节点: ros2 run aruco_detector aruco_detector_node"
        echo "  2. 机械臂控制节点: ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px100"
        echo ""
        read -p "是否继续? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python3 grasping_hor_app_2.py
        fi
        ;;
    3)
        echo -e "${YELLOW}查看ArUco话题...${NC}"
        ros2 topic echo /aruco_detector/marker_poses
        ;;
    4)
        echo -e "${YELLOW}查看机械臂关节状态...${NC}"
        ros2 topic echo /px100/joint_states --once
        ;;
    5)
        echo -e "${YELLOW}测试夹爪控制...${NC}"
        echo ""
        echo "1) 打开夹爪"
        echo "2) 关闭夹爪"
        read -p "请选择 [1-2]: " gripper_choice
        
        if [ "$gripper_choice" = "1" ]; then
            echo "发送打开命令..."
            ros2 topic pub --once /px100/commands/joint_single interbotix_xs_msgs/msg/JointSingleCommand "{name: 'gripper', cmd: 1.5}"
        elif [ "$gripper_choice" = "2" ]; then
            echo "发送关闭命令..."
            ros2 topic pub --once /px100/commands/joint_single interbotix_xs_msgs/msg/JointSingleCommand "{name: 'gripper', cmd: 0.6}"
        fi
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}完成${NC}"

