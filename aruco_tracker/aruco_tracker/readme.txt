编译工作空间
cd ~/ros2_ws
colcon build --packages-select my_aruco_tracker
source install/setup.bash

ros2 launch realsense2_camera rs_launch.py
ros2 run aruco_tracker detect_node
ros2 run rviz2 rviz2
在 Rviz2 中：
    Change Fixed Frame to camera_link (or camera_color_optical_frame).
    Add -> Image -> Topic /aruco_image_result (查看画框的效果)。
    Add -> PoseArray -> Topic /aruco_poses (查看红色的位姿箭头)。




ros2 run aruco_tracker tracker

unset RMW_IMPLEMENTATION
ros2 run tf2_ros tf2_echo camera_color_optical_frame aruco_marker_1