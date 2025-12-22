#!/usr/bin/env python3
"""
Ultra-simplified EE211 Project Launch File
No delays | No robot bringup | Core functional components only
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument

def generate_launch_description():
    # 核心路径配置
    project_root = '/home/tony/ros2_ws/src/EE211_project'

    # 1. 机械臂控制器 | 命令行: python3 /home/tony/ros2_ws/src/EE211_project/arm/arm_controller.py
    arm_controller = ExecuteProcess(
        cmd=['python3', f'{project_root}/arm/arm_controller.py'],
        output='screen',
        name='arm_controller'
    )

    # 2. 交通灯检测 | 命令行: python3 /home/tony/ros2_ws/src/EE211_project/detection/inference_nuc.py
    traffic_light_detection = ExecuteProcess(
        cmd=['python3', f'{project_root}/detection/inference_nuc.py'],
        output='screen',
        name='traffic_light_detection'
    )

    # 3. 抓取脚本(可选) | 命令行: python3 /home/tony/ros2_ws/src/EE211_project/arm/grasping_ok.py
    grasping_ok = ExecuteProcess(
        cmd=['python3', f'{project_root}/arm/grasping_ok.py'],
        output='screen',
        name='grasping_ok'
    )

    # 4. 放置盒子脚本(可选) | 命令行: python3 /home/tony/ros2_ws/src/EE211_project/arm/place_the_box.py
    place_box = ExecuteProcess(
        cmd=['python3', f'{project_root}/arm/place_the_box.py'],
        output='screen',
        name='place_box'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time if true'),
        arm_controller,         # 机械臂控制
        traffic_light_detection,# 交通灯检测
        grasping_ok,            # 抓取(可选，不需要直接删此行)
        place_box               # 放置盒子(可选，不需要直接删此行)
    ])