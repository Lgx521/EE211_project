import os
import launch
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler,
    TimerAction, LogInfo
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessStart, OnProcessExit
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ====================== 1. 定义路径（避免硬编码，适配不同环境） ======================
    # 地图文件路径（推荐用相对路径+FindPackageShare，而非绝对路径）
    map_yaml_path = os.path.join(
        get_package_share_directory('planning'),  # 你的planning包名
        'map', 'map.yaml'  # 地图文件放在planning包的map目录下（需提前创建）
    )
    # 备选：若仍想用绝对路径，保留以下行（注释掉上面的相对路径）
    # map_yaml_path = '/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml'

    # turtlebot4相关launch文件路径
    turtlebot4_navigation_dir = get_package_share_directory('turtlebot4_navigation')
    turtlebot4_viz_dir = get_package_share_directory('turtlebot4_viz')

    # 自定义导航脚本的执行命令（优先用ros2 run，而非直接跑脚本）
    nav_script_cmd = [
        'ros2', 'run', 'planning', 'navigation',  # 对应setup.py中注册的可执行名（navigation）
        '--ros-args', '--log-level', 'INFO'  # 输出日志级别
    ]

    # ====================== 2. 启动定位（localization.launch.py） ======================
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot4_navigation_dir, 'launch', 'localization.launch.py')
        ),
        launch_arguments={
            'map': map_yaml_path,
            'use_sim_time': 'false',  # 实际机器人设为false，仿真设为true
            'autostart': 'true'
        }.items()
    )

    # ====================== 3. 启动Nav2导航（nav2.launch.py） ======================
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot4_navigation_dir, 'launch', 'nav2.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'autostart': 'true'
        }.items()
    )

    # ====================== 4. 启动可视化（view_robot.launch.py） ======================
    view_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot4_viz_dir, 'launch', 'view_robot.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false'
        }.items()
    )

    # ====================== 5. 启动自定义导航脚本（关键：等Nav2就绪后再启动） ======================
    nav_script = ExecuteProcess(
        cmd=nav_script_cmd,
        output='screen',  # 脚本日志输出到终端
        shell=False  # 避免shell解析问题
    )

    # ====================== 6. 启动顺序控制（核心：解决“脚本启动太早”问题） ======================
    # 逻辑：定位→Nav2→可视化 启动后，延迟10秒再启动自定义脚本（给Nav2/AMCL足够初始化时间）
    delayed_nav_script = TimerAction(
        period=10.0,  # 延迟10秒（可根据实际调整，比如8/15秒）
        actions=[
            LogInfo(msg='✅ Nav2/AMCL已就绪，启动自定义导航脚本...'),
            nav_script
        ]
    )

    # ====================== 7. 事件监听（可选：增加鲁棒性） ======================
    # 监听Nav2启动成功后，再触发延迟启动脚本
    nav2_start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=nav2_launch,
            on_start=[
                LogInfo(msg='🚀 Nav2 launch started, waiting for initialization...'),
                delayed_nav_script
            ]
        )
    )

    # 监听脚本退出后，打印提示
    nav_script_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=nav_script,
            on_exit=[
                LogInfo(msg='📜 自定义导航脚本执行完成/退出！')
            ]
        )
    )

    # ====================== 8. 组装所有启动项 ======================
    return LaunchDescription([
        # 先启动定位、Nav2、可视化
        localization_launch,
        nav2_launch,
        view_robot_launch,
        # 事件监听+延迟启动脚本
        nav2_start_handler,
        nav_script_exit_handler
    ])