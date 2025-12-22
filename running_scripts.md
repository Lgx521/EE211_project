<!-- 启动交通灯识别：
python3 /home/tony/ros2_ws/src/EE211_project/detection/trafficlight_dection2.py
/traffic_light/status (std_msgs/String): RED/GREEN/YELLOW/UNKNOWN 等状态（取最高置信度结果）
/traffic_light/boxes (std_msgs/String): JSON 数组，包含每个检测框的类别、置信度与像素坐标
/traffic_light/image_annotated (sensor_msgs/Image): 叠加检测框的可视化图像（可通过参数关闭） -->



<!-- ## Realsense Driver
```bash
ros2 launch realsense2_camera rs_launch.py
``` -->

## Bring up
```bash
ros2 launch iqr_tb4_bringup bringup.launch.py
```

## To launch the arm controller
```bash
python3 src/EE211_project/arm/arm_controller.py
```


## To launch the grasping
- Still pending the viability
```bash
python3 src/EE211_project/arm/grasping_ok.py
python3 src/EE211_project/arm/place_the_box.py
```

## Navigation
```bash
ros2 launch turtlebot4_navigation localization.launch.py map:=/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml
ros2 launch turtlebot4_navigation nav2.launch.py
ros2 launch turtlebot4_viz view_robot.launch.py
~/ros2_ws/src/EE211_project/planning/scripts/navigation.py
```

# 启动红绿灯检测脚本
```bash
python3  /home/tony/ros2_ws/src/EE211_project/detection/inference_nuc.py
```