# EE211 Project

## Project Structure

```
EE211_project/
│
├── arm/                # Robotic arm control and grasping scripts
│   ├── arm_controller.py
│   ├── grasping.py
│   ├── ...
│
├── aruco_tracker/      # ArUco marker detection and tracking module
│   ├── aruco_node.py
│   ├── ...
│
├── detection/          # Object detection and traffic light recognition
│   ├── aruco_detection_node.py
│   ├── trafficlight_dection.py
│   ├── ...
│
├── my_nav2_planner/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── global_planner_plugin.xml
│   ├── include/
│   │   └── my_nav2_planner/
│   │       └── my_astar_planner.hpp
│   └── src/
│       └── my_astar_planner.cpp
│
├── planning/           # Main navigation package
│   ├── scripts/
│   │   ├── navigation.py      # Full-process navigation
│   │   ├── simple_nav.py      # Single-point navigation
│   │   ├── stop_at_rviz.py    # RViz navigation with traffic light support
│   │   └── read_pose.py       # Read waypoint data
│   ├── launch/
│   │   ├── navigation.launch.py   # Main navigation launch file
│   │   └── supplement.launch.py   # Supplementary launch file
│   ├── map/                # Map files
│   ├── ...
│
├── yolo_detection/     # YOLO detection scripts
│   ├── collection.py
│
└── yolo_test/          # YOLO test models and results
  ├── yolov8n.pt
  ├── yolov8n_openvino_model/
```

## Common Scripts and Launch Commands

### Automatic Navigation Process
```bash
ros2 launch iqr_tb4_bringup bringup.launch.py
ros2 launch planning navigation.launch.py
ros2 launch planning supplement.launch.py
```

### 1. System Bringup
```bash
ros2 launch iqr_tb4_bringup bringup.launch.py
```

### 2. Start Arm Controller
```bash
python3 src/EE211_project/arm/arm_controller.py
```

### 3. Start Grasping/Placing
```bash
python3 src/EE211_project/arm/grasping_ok.py
python3 src/EE211_project/arm/place_the_box.py
```

### 4. Navigation Process
```bash
ros2 launch turtlebot4_navigation localization.launch.py map:=/home/tony/ros2_ws/src/EE211_project/planning/map/map.yaml
ros2 launch turtlebot4_navigation nav2.launch.py
ros2 launch turtlebot4_viz view_robot.launch.py
python3 ~/ros2_ws/src/EE211_project/planning/scripts/navigation.py
```

### 5. Start Traffic Light Detection
```bash
python3 /home/tony/ros2_ws/src/EE211_project/detection/inference_nuc.py
```

## Authors
The project was a joint effort by Yiwen Ying, Shengzhe Gan, Zixuan Lv, and Xinxiang Duan.

<a href="https://github.com/Wendy-Ying">
  <img src="https://avatars.githubusercontent.com/u/143325815?v=4" width="100" />
</a>

<a href="https://github.com/Lgx521">
  <img src="https://avatars.githubusercontent.com/u/148550006?v=4" width="100" />
</a>

<a href="https://github.com/dxxphy">
  <img src="https://avatars.githubusercontent.com/u/180681751?v=4" width="100" />
</a>
