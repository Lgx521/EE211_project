#!/usr/bin/env python3
"""
System Launch File (Without SLAM/Bringup)
假设 Bringup 和 SLAM 已经在运行
只启动：Arm Controller, Traffic Light Detection, Grasping
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Paths
    project_root = '/home/tony/ros2_ws/src/EE211_project'
    
    # Script paths
    arm_controller_path = os.path.join(project_root, 'arm/arm_controller.py')
    traffic_light_detection_path = os.path.join(project_root, 'detection/inference_nuc.py')
    grasping_ok_path = os.path.join(project_root, 'arm/grasping_ok.py')
    place_box_path = os.path.join(project_root, 'arm/place_the_box.py')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # 1. Arm Controller - Start immediately (bringup already running)
    arm_controller = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', arm_controller_path],
                output='screen',
                name='arm_controller'
            )
        ]
    )
    
    # 2. Traffic Light Detection - Start after delay to ensure SLAM is stable
    # 使用帧跳过机制，降低相机订阅频率
    traffic_light_detection = TimerAction(
        period=5.0,  # 给SLAM一些时间稳定
        actions=[
            ExecuteProcess(
                cmd=['python3', traffic_light_detection_path],
                output='screen',
                name='traffic_light_detection',
                additional_env={'PYTHONUNBUFFERED': '1'}
            )
        ]
    )
    
    # 3. Grasping scripts
    grasping_ok = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', grasping_ok_path],
                output='screen',
                name='grasping_ok'
            )
        ]
    )
    
    place_box = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', place_box_path],
                output='screen',
                name='place_box'
            )
        ]
    )
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        
        # Launch all components
        arm_controller,             # t=2s: Arm controller
        traffic_light_detection,    # t=5s: Traffic light detection (with frame skipping)
        grasping_ok,                # t=8s: Grasping
        place_box,                  # t=10s: Place box
    ])

