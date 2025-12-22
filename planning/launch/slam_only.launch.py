#!/usr/bin/env python3
"""
SLAM Only Launch File
只启动 Bringup 和 SLAM，不启动其他可能冲突的节点
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch directories
    bringup_launch_dir = os.path.join(get_package_share_directory('iqr_tb4_bringup'), 'launch')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # 1. Bringup - Robot base initialization (includes camera)
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_launch_dir, 'bringup.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        
        # Only launch bringup
        # SLAM should be launched separately after this is stable
        bringup,
    ])

