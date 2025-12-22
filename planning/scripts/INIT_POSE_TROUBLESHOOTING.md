# Initial Pose 故障排除指南

## 问题描述
导航脚本中的 `publish_init_pose()` 有时会失败，但在 RViz 中手动设置初始位姿可以成功。

## 常见原因

### 1. **AMCL 未就绪**
- 脚本发布初始位姿时，AMCL 节点可能还没有完全启动
- 解决方案：脚本现在会等待 `/initialpose` 话题有订阅者

### 2. **协方差矩阵缺失**
- 原来的代码没有设置协方差矩阵
- AMCL 可能拒绝没有协方差的位姿
- 解决方案：已添加标准协方差矩阵（与 RViz 相同）

### 3. **发布次数不足**
- 网络延迟或消息丢失可能导致 AMCL 没收到消息
- 解决方案：增加发布次数到 5 次，每次间隔 0.5 秒

### 4. **时间戳问题**
- 使用过时的时间戳可能导致消息被忽略
- 解决方案：每次发布都更新时间戳

## 改进后的功能

### ✅ 新增功能

1. **订阅者检查**
   ```python
   # 等待 AMCL 订阅 /initialpose
   while self.init_pose_pub.get_subscription_count() == 0:
       # 等待最多 30 秒
   ```

2. **协方差矩阵**
   ```python
   msg.pose.covariance = [
       0.25, 0.0, ...,  # x 方差
       0.0, 0.25, ...,  # y 方差
       ...,
       0.0, 0.0, ..., 0.068  # yaw 方差
   ]
   ```

3. **多次发布**
   - 发布 5 次，每次更新时间戳
   - 每次间隔 0.5 秒

4. **详细日志**
   - 显示等待状态
   - 显示订阅者数量
   - 显示发布进度

## 诊断工具

### 1. 系统诊断脚本

检查导航系统是否就绪：

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run planning diagnose_navigation.py
```

**检查项目：**
- ✓ 必需的话题是否存在
- ✓ Nav2 节点是否运行
- ✓ 地图是否加载
- ✓ Action 服务器是否可用

### 2. 初始位姿测试脚本

独立测试初始位姿发布：

```bash
ros2 run planning test_init_pose.py
```

**功能：**
- 等待 AMCL 就绪
- 发布测试位姿
- 显示详细状态

### 3. 交通灯测试脚本

测试交通灯停止/恢复功能：

```bash
ros2 run planning test_traffic_light.py
```

**命令：**
- `s` + Enter：发送停止信号
- `g` + Enter：发送前进信号
- `q` + Enter：退出

## 使用流程

### 步骤 1：启动系统

```bash
# 启动机器人基础系统
ros2 launch planning full_system.launch.py

# 或者分别启动
ros2 launch turtlebot4_navigation localization.launch.py map:=<your_map>
ros2 launch turtlebot4_navigation nav2.launch.py
```

### 步骤 2：运行诊断

```bash
# 检查系统是否就绪
ros2 run planning diagnose_navigation.py
```

**期望输出：**
```
✓ /initialpose - Found
✓ /map - Found
✓ amcl - Running
✓ NavigateToPose action server - Available
✅ All checks passed! Navigation system is ready.
```

### 步骤 3：测试初始位姿

```bash
# 测试初始位姿发布
ros2 run planning test_init_pose.py
```

**期望输出：**
```
✓ Found 1 subscriber(s)
📍 Publishing initial pose:
   Position: x=6.713, y=-2.050
   Orientation: z=-0.483, w=0.875
   Published (1/5)
   ...
✅ Initial pose published successfully!
```

### 步骤 4：运行导航脚本

```bash
# 运行完整导航
ros2 run planning navigation.py
```

**期望日志：**
```
[INFO] Waiting for AMCL to be ready...
[INFO] Found 1 subscriber(s) on /initialpose
[INFO] Publishing initial pose: x=6.713, y=-2.050, z=-0.483, w=0.875
[INFO] Published /initialpose (1/5)
...
[INFO] Initial pose published successfully!
[INFO] Sending goal 1/6: (x=6.71, y=-2.05, z=-0.483, w=0.875)
```

## 故障排除

### 问题 1：No subscribers on /initialpose

**症状：**
```
[WARN] No subscribers on /initialpose topic! AMCL may not be running.
```

**解决方案：**
```bash
# 检查 AMCL 是否运行
ros2 node list | grep amcl

# 如果没有，启动定位
ros2 launch turtlebot4_navigation localization.launch.py map:=<your_map>
```

### 问题 2：NavigateToPose action server not available

**症状：**
```
[ERROR] NavigateToPose action server not available!
```

**解决方案：**
```bash
# 检查 Nav2 是否运行
ros2 node list | grep navigator

# 如果没有，启动 Nav2
ros2 launch turtlebot4_navigation nav2.launch.py
```

### 问题 3：Map not loaded

**症状：**
```
[WARN] No map received on /map topic
```

**解决方案：**
```bash
# 检查地图话题
ros2 topic echo /map --once

# 如果没有输出，加载地图
ros2 launch turtlebot4_navigation localization.launch.py map:=/path/to/your/map.yaml
```

### 问题 4：Initial pose not updating in RViz

**可能原因：**
1. 坐标超出地图范围
2. 地图坐标系不匹配
3. TF 树有问题

**检查方法：**
```bash
# 检查 TF 树
ros2 run tf2_tools view_frames

# 检查地图信息
ros2 topic echo /map/info --once

# 手动在 RViz 中设置位姿，看是否工作
```

## 参数调整

如果需要调整初始位姿的协方差（置信度）：

```python
# 在 navigation.py 中修改
msg.pose.covariance = [
    0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  # x 方差 (增大=更不确定)
    0.0, 0.25, 0.0, 0.0, 0.0, 0.0,  # y 方差
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.068  # yaw 方差
]
```

**建议值：**
- 高置信度：x/y = 0.1, yaw = 0.03
- 中等置信度：x/y = 0.25, yaw = 0.068（默认）
- 低置信度：x/y = 0.5, yaw = 0.15

## 最佳实践

1. **总是先运行诊断**
   ```bash
   ros2 run planning diagnose_navigation.py
   ```

2. **确保系统完全启动**
   - 等待所有节点启动（约 10-15 秒）
   - 检查 RViz 中是否显示地图

3. **验证初始位姿**
   - 使用 `test_init_pose.py` 先测试
   - 在 RViz 中确认机器人位置正确

4. **监控日志**
   - 观察 `[INFO]` 消息确认进度
   - 注意 `[WARN]` 和 `[ERROR]` 消息

5. **逐步测试**
   - 先测试单个路径点
   - 确认后再测试完整路径

## 联系支持

如果问题仍然存在，请提供：
1. `diagnose_navigation.py` 的完整输出
2. `navigation.py` 的日志（从启动到失败）
3. RViz 截图
4. 地图文件信息

---

**更新日期：** 2025-12-23  
**版本：** 2.0

