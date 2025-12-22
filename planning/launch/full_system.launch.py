#!/usr/bin/env python3
"""
Full System Launch File for EE211 Project
Launches all components including:
- Bringup (robot base)
- Arm controller
- Traffic light detection
- Navigation stack (localization, nav2, rviz)
- Navigation script
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Paths
    map_path = '/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml'
    project_root = '/home/tony/ros2_ws/src/EE211_project'
    
    # Launch directories
    nav_launch_dir = os.path.join(get_package_share_directory('turtlebot4_navigation'), 'launch')
    viz_launch_dir = os.path.join(get_package_share_directory('turtlebot4_viz'), 'launch')
    bringup_launch_dir = os.path.join(get_package_share_directory('iqr_tb4_bringup'), 'launch')
    
    # Script paths
    arm_controller_path = os.path.join(project_root, 'arm/arm_controller.py')
    traffic_light_detection_path = os.path.join(project_root, 'detection/inference_nuc.py')
    nav_script_path = os.path.join(project_root, 'planning/scripts/navigation.py')
    
    # Optional: Grasping scripts (commented out as marked "pending viability")
    grasping_ok_path = os.path.join(project_root, 'arm/grasping_ok.py')
    place_box_path = os.path.join(project_root, 'arm/place_the_box.py')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # 1. Bringup - Robot base initialization
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_launch_dir, 'bringup.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # 4. Localization - Start 10s after bringup
    localization = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav_launch_dir, 'localization.launch.py')),
                launch_arguments={
                    'map': map_path,
                    'use_sim_time': use_sim_time
                }.items()
            )
        ]
    )
    
    # 5. Nav2 - Start 13s after bringup
    nav2 = TimerAction(
        period=13.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav_launch_dir, 'nav2.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )
    
    # 6. RViz Visualization - Start 13s after bringup
    view_robot = TimerAction(
        period=13.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(viz_launch_dir, 'view_robot.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )
    
    # 7. Navigation Script - Start 25s after bringup (give Nav2 plenty of time)
    nav_script = TimerAction(
        period=25.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', nav_script_path],
                output='screen',
                name='navigation_script',
                cwd='/home/tony/ros2_ws',
                shell=False,
                emulate_tty=True
            )
        ]
    )



    # 2. Arm Controller - Start 8s after bringup
    arm_controller = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', arm_controller_path],
                output='screen',
                name='arm_controller',
                cwd='/home/tony/ros2_ws',
                shell=False,
                emulate_tty=True
            )
        ]
    )
    
    # 3. Traffic Light Detection - Start 8s after bringup
    traffic_light_detection = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', traffic_light_detection_path],
                output='screen',
                name='traffic_light_detection',
                cwd='/home/tony/ros2_ws',
                shell=False,
                emulate_tty=True
            )
        ]
    )

    
    # Optional: Grasping scripts (uncomment if needed)
    grasping_ok = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', grasping_ok_path],
                output='screen',
                name='grasping_ok',
                cwd='/home/tony/ros2_ws',
                shell=False,
                emulate_tty=True
            )
        ]
    )
    
    place_box = TimerAction(
        period=20.0,
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
        
        # Launch all components in sequence
        # bringup,                    # t=0s: Robot base
        
        # localization,               # t=5s: Localization
        # nav2,                       # t=5s: Navigation stack
        # view_robot,                 # t=5s: RViz visualization
        # nav_script,                 # t=6s: Navigation script

        arm_controller,             # t=8s: Arm controller (wait 8s after bringup)
        traffic_light_detection,    # t=8s: Traffic light detection (wait 8s after bringup)
        grasping_ok,              # t=10s: Grasping (optional)
        place_box,                # t=10s: Place box (optional)
    ])

