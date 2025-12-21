import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_path = '/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml'
    nav_launch_dir = os.path.join(get_package_share_directory('turtlebot4_navigation'), 'launch')
    viz_launch_dir = os.path.join(get_package_share_directory('turtlebot4_viz'), 'launch')
    nav_script_path = '/home/tony/ros2_ws/src/EE211_project/planning/scripts/navigation.py'

    # ros2 launch turtlebot4_navigation localization.launch.py map:=ros2_ws/src/EE211_project/planning/map/map.yaml
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav_launch_dir, 'localization.launch.py')),
        launch_arguments={'map': map_path}.items()
    )

    # ros2 launch turtlebot4_navigation nav2.launch.py
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav_launch_dir, 'nav2.launch.py'))
    )

    # ros2 launch turtlebot4_viz view_robot.launch.py
    view_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(viz_launch_dir, 'view_robot.launch.py'))
    )

    # ~/ros2_ws/src/EE211_project/planning/scripts/navigation.py
    nav_script = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[nav_script_path],
                output='screen',
                shell=True
            )
        ]
    )

    return LaunchDescription([
        localization,
        nav2,
        view_robot,
        nav_script
    ])