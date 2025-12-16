#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 红绿灯检测节点（trafficlight_dection2.py）

功能概要：
- 参考 aruco_detection_ros.py 的订阅方式，订阅相机图像并使用 YOLO 最优模型 best.pt 实时推理。
- 发布三个话题：
  - /traffic_light/status (std_msgs/String): RED/GREEN/YELLOW/UNKNOWN 等状态（取最高置信度结果）
  - /traffic_light/boxes (std_msgs/String): JSON 数组，包含每个检测框的类别、置信度与像素坐标
  - /traffic_light/image_annotated (sensor_msgs/Image): 叠加检测框的可视化图像（可通过参数关闭）

快速使用：
1) 安装依赖（在 ROS2 环境中）：
   pip install ultralytics opencv-python numpy
   sudo apt-get install ros-$ROS_DISTRO-cv-bridge

2) 运行（默认订阅 /camera/camera/color/image_raw）：
   cd ~/ros2_ws/src/EE211_project/detection
   python3 trafficlight_dection2.py --ros-args \
     -p model_path:=/home/tony/Yolo/runs/detect/traffic_detection3/weights/best.pt \
     -p conf_threshold:=0.5 \
     -p image_topic:=/camera/camera/color/image_raw

说明：
- 若不传 model_path，会自动在 ~/Yolo/runs/detect/*/weights/best.pt 中选择最新的 best.pt。
- 若你使用其它图像话题，请用 -p image_topic:=<你的话题> 指定。
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
    _UL_OK = True
except Exception as e:
    _UL_OK = False
    _UL_ERR = str(e)


class TrafficLightDetector2(Node):
    def __init__(self) -> None:
        super().__init__('traffic_light_detector2')

        # 参数（对齐 aruco_detection_ros.py 的命名风格）
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model_path', self._auto_find_best())
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)

        self.image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold: float = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.publish_annotated: bool = self.get_parameter('publish_annotated').get_parameter_value().bool_value

        self.get_logger().info('================ Traffic Light Detection v2 ================')
        self.get_logger().info(f'image_topic: {self.image_topic}')
        self.get_logger().info(f'model_path: {self.model_path}')
        self.get_logger().info(f'conf_threshold: {self.conf_threshold}')
        self.get_logger().info(f'publish_annotated: {self.publish_annotated}')
        self.get_logger().info('===========================================================')

        if not _UL_OK:
            self.get_logger().error('导入 ultralytics 失败，请先安装: pip install ultralytics')
            self.get_logger().error(f'错误信息: {_UL_ERR}')
            raise RuntimeError('ultralytics 不可用')

        try:
            if not os.path.exists(self.model_path):
                self.get_logger().warn(f'指定模型不存在: {self.model_path}，尝试自动搜索最新 best.pt')
                self.model_path = self._auto_find_best()
                self.get_logger().info(f'自动选择模型: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('YOLO 模型加载成功')
        except Exception as e:
            self.get_logger().error(f'YOLO 模型加载失败: {e}')
            raise

        # ROS 通信
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.ann_pub = self.create_publisher(Image, 'traffic_light/image_annotated', 10)
        self.status_pub = self.create_publisher(String, 'traffic_light/status', 10)
        self.boxes_pub = self.create_publisher(String, 'traffic_light/boxes', 10)

        # 统计
        self.frame_cnt = 0
        self.t0 = time.time()
        self.inf_hist: List[float] = []

    def _auto_find_best(self) -> str:
        home = os.path.expanduser('~')
        candidates = glob.glob(os.path.join(home, 'Yolo', 'runs', 'detect', '*', 'weights', 'best.pt'))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
        return 'best.pt'

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

        # 发布状态（最高置信度）
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

        # 性能日志
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
        node = TrafficLightDetector2()
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

