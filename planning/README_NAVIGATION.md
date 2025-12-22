# Navigation Package - 使用指南

## 📦 包含的功能

### 1. **主导航脚本** (`navigation.py`)
完整的路径点导航系统，包含：
- ✅ 自动初始位姿设置
- ✅ 多路径点导航
- ✅ 抓取/放置操作集成
- ✅ 交通灯停止/恢复控制

### 2. **交通灯控制**
- 订阅 `/traffic_light/stop_control` (Bool)
- `True`: 停止当前导航
- `False`: 恢复到同一目标点

### 3. **诊断和测试工具**
- `diagnose_navigation.py`: 系统诊断
- `test_init_pose.py`: 测试初始位姿
- `test_traffic_light.py`: 测试交通灯控制

## 🚀 快速开始

### 步骤 1: 编译包

```bash
cd ~/ros2_ws
colcon build --packages-select planning
source install/setup.bash
```

### 步骤 2: 启动系统

```bash
# 启动完整系统（包括 SLAM、Nav2 等）
ros2 launch planning full_system.launch.py
```

或者分别启动：

```bash
# 终端 1: 启动定位
ros2 launch turtlebot4_navigation localization.launch.py map:=/path/to/your/map.yaml

# 终端 2: 启动 Nav2
ros2 launch turtlebot4_navigation nav2.launch.py
```

### 步骤 3: 运行诊断（推荐）

```bash
# 检查系统是否就绪
ros2 run planning diagnose_navigation
```

**期望输出：**
```
✓ /initialpose - Found
✓ /map - Found
✓ amcl - Running
✓ NavigateToPose action server - Available
✅ All checks passed! Navigation system is ready.
```

### 步骤 4: 测试初始位姿（可选）

```bash
# 测试初始位姿发布是否正常
ros2 run planning test_init_pose
```

### 步骤 5: 运行导航

```bash
# 运行主导航脚本
ros2 run planning navigation
```

## 🔧 功能详解

### 初始位姿自动设置

脚本会自动：
1. 等待 AMCL 就绪（检查 `/initialpose` 订阅者）
2. 发布初始位姿（带协方差矩阵）
3. 等待定位稳定
4. 开始导航

**改进点：**
- ✅ 添加订阅者检查（最多等待 30 秒）
- ✅ 添加协方差矩阵（与 RViz 相同）
- ✅ 多次发布（5 次，每次更新时间戳）
- ✅ 详细日志输出

**日志示例：**
```
[INFO] Waiting for AMCL to be ready...
[INFO] Found 1 subscriber(s) on /initialpose
[INFO] Publishing initial pose: x=6.713, y=-2.050, z=-0.483, w=0.875
[INFO] Published /initialpose (1/5)
[INFO] Published /initialpose (2/5)
...
[INFO] Initial pose published successfully!
```

### 交通灯控制

**工作原理：**
- 监听 `/traffic_light/stop_control` 话题
- 检测边沿触发（False → True 或 True → False）
- 支持多次停止/恢复

**停止流程（收到 True）：**
1. 取消当前导航目标
2. 保存当前状态和路径点索引
3. 进入 `traffic_stopped` 状态
4. 机器人停在当前位置

**恢复流程（收到 False）：**
1. 恢复之前的状态
2. 重新发送相同的路径点目标
3. 机器人继续导航

**测试方法：**
```bash
# 终端 1: 运行导航
ros2 run planning navigation

# 终端 2: 运行交通灯测试工具
ros2 run planning test_traffic_light

# 在测试工具中输入：
# s + Enter: 发送停止信号
# g + Enter: 发送前进信号
```

### 路径点配置

在 `navigation.py` 中修改 `get_waypoints()` 函数：

```python
def get_waypoints(self):
    """Define navigation waypoints with x, y, z (orientation), w (orientation)"""
    return [
        [x1, y1, z1, w1],  # 第1个点
        [x2, y2, z2, w2],  # 第2个点
        # ... 添加更多点
    ]
```

**坐标格式：**
- `x, y`: 位置坐标（米）
- `z, w`: 四元数方向（z 和 w 分量）

**获取坐标：**
```bash
# 方法 1: 使用 RViz 的 "2D Pose Estimate" 工具
# 方法 2: 订阅 /amcl_pose 话题
ros2 topic echo /amcl_pose
```

### 抓取/放置集成

**第 4 个路径点：**
- 到达后发布 `/grasp/start = True`
- 等待 `/grasp/success = True`
- 收到成功信号后继续下一个点

**第 5 个路径点：**
- 到达后发布 `/place/start = True`
- 等待 `/place/success = True`
- 收到成功信号后继续下一个点

## 🛠️ 故障排除

### 问题 1: 初始位姿设置失败

**症状：**
```
[WARN] No subscribers on /initialpose topic!
```

**解决方案：**
```bash
# 1. 检查 AMCL 是否运行
ros2 node list | grep amcl

# 2. 如果没有，启动定位
ros2 launch turtlebot4_navigation localization.launch.py map:=<your_map>

# 3. 运行诊断
ros2 run planning diagnose_navigation
```

### 问题 2: 导航目标被拒绝

**症状：**
```
[ERROR] NavigateToPose action server not available!
```

**解决方案：**
```bash
# 1. 检查 Nav2 是否运行
ros2 node list | grep navigator

# 2. 启动 Nav2
ros2 launch turtlebot4_navigation nav2.launch.py

# 3. 等待几秒后重试
```

### 问题 3: 机器人不移动

**可能原因：**
1. 初始位姿不正确
2. 目标点超出地图范围
3. 路径被障碍物阻挡

**检查方法：**
```bash
# 1. 在 RViz 中检查机器人位置
# 2. 检查目标点是否在地图内
# 3. 查看 costmap
ros2 topic echo /local_costmap/costmap --once
```

### 问题 4: 交通灯控制不工作

**检查：**
```bash
# 1. 检查话题是否存在
ros2 topic list | grep traffic_light

# 2. 手动发布测试
ros2 topic pub /traffic_light/stop_control std_msgs/msg/Bool "data: true" --once

# 3. 检查导航脚本日志
# 应该看到 "STOP signal received"
```

## 📊 日志级别

修改日志级别：

```bash
# 详细日志
ros2 run planning navigation --ros-args --log-level debug

# 只显示警告和错误
ros2 run planning navigation --ros-args --log-level warn
```

## 🔍 监控和调试

### 查看当前位姿
```bash
ros2 topic echo /amcl_pose
```

### 查看导航状态
```bash
ros2 topic echo /navigate_to_pose/_action/status
```

### 查看交通灯状态
```bash
ros2 topic echo /traffic_light/stop_control
```

### 查看 TF 树
```bash
ros2 run tf2_tools view_frames
# 生成 frames.pdf
```

## 📝 参数调整

### 初始位姿协方差

在 `navigation.py` 的 `publish_init_pose()` 中：

```python
msg.pose.covariance = [
    0.25, 0.0, ...,  # x 方差 (增大 = 更不确定)
    0.0, 0.25, ...,  # y 方差
    ...,
    0.0, 0.0, ..., 0.068  # yaw 方差
]
```

**推荐值：**
- 高精度定位：`x/y = 0.1, yaw = 0.03`
- 中等精度：`x/y = 0.25, yaw = 0.068` （默认）
- 低精度：`x/y = 0.5, yaw = 0.15`

### 等待时间

```python
# AMCL 等待时间（秒）
max_wait = 30  # 默认 30 秒

# 定位稳定等待时间（秒）
# 在 publish_init_pose() 中修改循环次数
for i in range(6):  # 6 * 0.5 = 3 秒
    rclpy.spin_once(self, timeout_sec=0.5)
```

## 🎯 最佳实践

1. **总是先运行诊断**
   ```bash
   ros2 run planning diagnose_navigation
   ```

2. **确保地图已加载**
   - 在 RViz 中检查地图显示
   - 确认机器人在地图范围内

3. **验证初始位姿**
   - 使用 `test_init_pose` 测试
   - 在 RViz 中确认位置正确

4. **逐步测试**
   - 先测试单个路径点
   - 确认后再测试完整路径

5. **监控日志**
   - 观察 `[INFO]` 消息
   - 注意 `[WARN]` 和 `[ERROR]`

## 📚 相关文档

- [INIT_POSE_TROUBLESHOOTING.md](scripts/INIT_POSE_TROUBLESHOOTING.md) - 初始位姿详细故障排除
- [Nav2 Documentation](https://navigation.ros.org/)
- [TurtleBot4 Navigation](https://turtlebot.github.io/turtlebot4-user-manual/tutorials/navigation.html)

## 🆘 获取帮助

如果遇到问题，请提供：
1. `diagnose_navigation` 的完整输出
2. 导航脚本的日志（从启动到失败）
3. RViz 截图
4. 系统信息：
   ```bash
   ros2 doctor --report
   ```

---

**最后更新：** 2025-12-23  
**版本：** 2.0  
**维护者：** Tony

