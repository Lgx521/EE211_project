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