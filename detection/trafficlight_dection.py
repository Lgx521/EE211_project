#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 红绿灯检测节点（trafficlight_dection.py）

功能概述：
- 订阅摄像头图像话题（sensor_msgs/Image），使用 YOLO 最优模型 best.pt 进行实时推理
- 输出红绿灯检测结果：
  1) 标注后的图像（带检测框与类别）
  2) 文本状态（最高置信度类别，如 RED/GREEN/YELLOW/STOP/UNKNOWN）
  3) 检测框详细信息（JSON 数组，包含类别、置信度、像素坐标）

快速使用：
1) 确保依赖可用（建议在ROS2工作空间虚拟环境中安装）：
   pip install ultralytics opencv-python numpy
   sudo apt-get install ros-$ROS_DISTRO-cv-bridge

2) 将本脚本放置于 ROS2 工程: ros2_ws/src/EE211_project/detection/ 下（已完成）

3) 运行（两种方式其一）：
   - 直接运行脚本：
       cd ~/ros2_ws
       source install/setup.bash  # 或 source /opt/ros/$ROS_DISTRO/setup.bash
       python3 src/EE211_project/detection/trafficlight_dection.py --ros-args \
         -p model_path:=/home/tony/Yolo/runs/detect/traffic_detection/weights/best.pt \
         -p conf_threshold:=0.5 \
         -p input_image_topic:=/camera/image_raw

   - 若已打包为可执行节点，可使用 ros2 run（按你的包名/入口配置为准）：
       ros2 run EE211_project trafficlight_dection --ros-args ...

可调参数（ROS2 参数）：
- model_path (string): 模型权重路径，默认自动搜索 ~/Yolo/runs/detect/*/weights/best.pt
- conf_threshold (double): 置信度阈值，默认 0.5
- input_image_topic (string): 输入图像话题，默认 /camera/image_raw（可按需改为 /camera/color/image_raw）
- publish_annotated (bool): 是否发布标注图像，默认 True

发布话题：
- traffic_light/image_annotated (sensor_msgs/Image)
- traffic_light/status (std_msgs/String)
- traffic_light/boxes (std_msgs/String, JSON)

注意：
- 本脚本为 ROS2 (rclpy) 版本；若在 ROS1 环境请使用 roscpp/rospy 版本。
- 若 ultralytics 未安装，将在启动时报错提示，请先安装依赖。
"""

import os
import glob
import time
import json
from typing import List, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _ULTRALYTICS_OK = True
except Exception as e:
    _ULTRALYTICS_OK = False
    _ULTRALYTICS_ERR = str(e)


class TrafficLightDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('traffic_light_detector')

        # 声明参数并读取
        self.declare_parameter('model_path', self._auto_find_best())
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('input_image_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated', True)

        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold: float = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.input_image_topic: str = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.publish_annotated: bool = self.get_parameter('publish_annotated').get_parameter_value().bool_value

        self.get_logger().info('===============================================')
        self.get_logger().info('Traffic Light Detection (ROS2) 节点启动')
        self.get_logger().info(f'model_path: {self.model_path}')
        self.get_logger().info(f'conf_threshold: {self.conf_threshold}')
        self.get_logger().info(f'input_image_topic: {self.input_image_topic}')
        self.get_logger().info(f'publish_annotated: {self.publish_annotated}')
        self.get_logger().info('===============================================')

        # 加载模型
        if not _ULTRALYTICS_OK:
            self.get_logger().error('ultralytics 导入失败，请先安装: pip install ultralytics')
            self.get_logger().error(f'导入错误信息: {_ULTRALYTICS_ERR}')
            raise RuntimeError('ultralytics 未安装或导入失败')

        try:
            if not os.path.exists(self.model_path):
                self.get_logger().warn(f'指定模型不存在: {self.model_path}，将尝试自动搜索最新 best.pt')
                self.model_path = self._auto_find_best()
                self.get_logger().info(f'自动选择模型: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('YOLO 模型加载成功')
        except Exception as e:
            self.get_logger().error(f'YOLO 模型加载失败: {e}')
            raise

        # CvBridge 与话题
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.input_image_topic, self.image_cb, 10)
        self.ann_pub = self.create_publisher(Image, 'traffic_light/image_annotated', 10)
        self.status_pub = self.create_publisher(String, 'traffic_light/status', 10)
        self.boxes_pub = self.create_publisher(String, 'traffic_light/boxes', 10)

        # 统计
        self.frame_cnt = 0
        self.t0 = time.time()
        self.inf_hist: List[float] = []

        # 动态参数回调
        self.add_on_set_parameters_callback(self._on_params_update)

    def _auto_find_best(self) -> str:
        # 优先使用 ~/Yolo/runs/detect/*/weights/best.pt 中最新的
        home = os.path.expanduser('~')
        candidates = glob.glob(os.path.join(home, 'Yolo', 'runs', 'detect', '*', 'weights', 'best.pt'))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
        # 兜底：当前目录或相对路径
        return 'best.pt'

    def _on_params_update(self, params):
        for p in params:
            if p.name == 'conf_threshold' and p.type_ == p.Type.DOUBLE:
                self.conf_threshold = float(p.value)
                self.get_logger().info(f'更新 conf_threshold -> {self.conf_threshold}')
            elif p.name == 'publish_annotated' and p.type_ == p.Type.BOOL:
                self.publish_annotated = bool(p.value)
                self.get_logger().info(f'更新 publish_annotated -> {self.publish_annotated}')
            elif p.name == 'model_path' and p.type_ == p.Type.STRING:
                new_model = str(p.value)
                if os.path.exists(new_model):
                    try:
                        self.model = YOLO(new_model)
                        self.model_path = new_model
                        self.get_logger().info(f'模型已切换 -> {self.model_path}')
                    except Exception as e:
                        self.get_logger().error(f'切换模型失败: {e}')
                else:
                    self.get_logger().warn(f'模型文件不存在: {new_model}，保持原设置')
        from rclpy.parameter import SetParametersResult
        return SetParametersResult(successful=True)

    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 转换失败: {e}')
            return

        t_start = time.time()
        try:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            self.get_logger().error(f'YOLO 推理失败: {e}')
            return
        inf_ms = (time.time() - t_start) * 1000.0

        dets = self._extract(results[0])

        # 发布状态（取最高置信度）
        status = 'UNKNOWN'
        if dets:
            best = max(dets, key=lambda d: d['confidence'])
            status = best['class'].upper()
        self.status_pub.publish(String(data=status))

        # 发布检测框 JSON
        try:
            self.boxes_pub.publish(String(data=json.dumps(dets, ensure_ascii=False)))
        except Exception as e:
            self.get_logger().warn(f'发布 boxes JSON 失败: {e}')

        # 发布标注图像
        if self.publish_annotated:
            try:
                ann = results[0].plot()  # ndarray, BGR
                img_msg = self.bridge.cv2_to_imgmsg(ann, encoding='bgr8')
                img_msg.header = msg.header
                self.ann_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().warn(f'发布标注图像失败: {e}')

        # 性能统计
        self.frame_cnt += 1
        self.inf_hist.append(inf_ms)
        if len(self.inf_hist) > 30:
            self.inf_hist.pop(0)
        if self.frame_cnt % 30 == 0:
            dt = time.time() - self.t0
            fps = 30.0 / dt if dt > 0 else 0.0
            avg_ms = float(np.mean(self.inf_hist)) if self.inf_hist else inf_ms
            self.get_logger().info(f'FPS: {fps:.2f} | Avg inference: {avg_ms:.1f} ms | Status: {status} | Dets: {len(dets)}')
            self.t0 = time.time()

    def _extract(self, result) -> List[Dict]:
        dets: List[Dict] = []
        try:
            boxes = getattr(result, 'boxes', None)
            names = getattr(result, 'names', {})
            if boxes is None:
                return dets
            # 遍历每个检测框
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id))
                dets.append({
                    'class': cls_name,
                    'confidence': round(conf, 4),
                    'bbox': {
                        'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                        'x2': int(xyxy[2]), 'y2': int(xyxy[3])
                    }
                })
        except Exception as e:
            self.get_logger().warn(f'解析YOLO结果失败: {e}')
        return dets


def main():
    rclpy.init()
    node = None
    try:
        node = TrafficLightDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f'节点异常: {e}')
        else:
            print(f'节点初始化失败: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
