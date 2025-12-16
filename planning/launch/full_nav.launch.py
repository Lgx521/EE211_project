import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_path = '/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml'
    navigation_launch_dir = os.path.join(
        get_package_share_directory('turtlebot4_navigation'),
        'launch'
    )
    my_nav_script = '/home/tony/ros2_ws/src/EE211_project/planning/resource/nodes/navigation.py'

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_launch_dir, 'localization.launch.py')
        ),
        launch_arguments={
            'map': map_path
        }.items()
    )

    nav_script = ExecuteProcess(
        cmd=['ros2', 'run', 'planning', 'navigation.py'],
        output='screen'
    )

    return LaunchDescription([
        localization,
        nav_script
    ])