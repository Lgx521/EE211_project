#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 红绿灯检测节点 - OpenVINO 加速版本 (inference_nuc.py)

功能概要：
- 基于 trafficlight_dection2.py 修改，使用 OpenVINO 进行硬件加速推理
- 订阅相机图像并使用转换后的 OpenVINO 模型实时推理
- 发布三个话题：
  - /traffic_light/status (std_msgs/String): RED/GREEN/YELLOW/UNKNOWN 等状态
  - /traffic_light/boxes (std_msgs/String): JSON 数组，包含每个检测框的类别、置信度与像素坐标
  - /traffic_light/image_annotated (sensor_msgs/Image): 叠加检测框的可视化图像
  - /traffic_light/stop_control (std_msgs/Bool): 停止控制信号

快速使用：
1) 安装依赖（在 ROS2 环境中）：
   pip install openvino opencv-python numpy
   sudo apt-get install ros-$ROS_DISTRO-cv-bridge

2) 运行（默认订阅 /camera/camera/color/image_raw）：
   cd ~/ros2_ws/src/EE211_project/detection
   python3 inference_nuc.py --ros-args \
     -p model_path:=/home/tony/ros2_ws/src/EE211_project/detection/best_openvino_model \
     -p conf_threshold:=0.5 \
     -p image_topic:=/camera/camera/color/image_raw

说明：
- model_path 应指向包含 best.xml 和 best.bin 的文件夹
- 使用 OpenVINO 进行 CPU/GPU 加速推理
"""

import os
import time
import json
import yaml
from typing import List, Dict, Deque, Tuple
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

import numpy as np
import cv2

try:
    from openvino.runtime import Core
    _OV_OK = True
except Exception as e:
    _OV_OK = False
    _OV_ERR = str(e)


class TrafficLightDetectorOpenVINO(Node):
    def __init__(self) -> None:
        super().__init__('traffic_light_detector_openvino')

        # 参数声明
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model_path', '/home/tony/ros2_ws/src/EE211_project/detection/best_openvino_model')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('filter_window_size', 1)
        self.declare_parameter('stop_threshold', 70)
        self.declare_parameter('device', 'CPU')  # CPU, GPU, MYRIAD 等

        self.image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold: float = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.publish_annotated: bool = self.get_parameter('publish_annotated').get_parameter_value().bool_value
        self.filter_window_size: int = self.get_parameter('filter_window_size').get_parameter_value().integer_value
        self.stop_threshold: int = self.get_parameter('stop_threshold').get_parameter_value().integer_value
        self.device: str = self.get_parameter('device').get_parameter_value().string_value

        self.get_logger().info('================ Traffic Light Detection - OpenVINO ================')
        self.get_logger().info(f'image_topic: {self.image_topic}')
        self.get_logger().info(f'model_path: {self.model_path}')
        self.get_logger().info(f'conf_threshold: {self.conf_threshold}')
        self.get_logger().info(f'publish_annotated: {self.publish_annotated}')
        self.get_logger().info(f'filter_window_size: {self.filter_window_size}')
        self.get_logger().info(f'stop_threshold: {self.stop_threshold}')
        self.get_logger().info(f'device: {self.device}')
        self.get_logger().info('====================================================================')

        if not _OV_OK:
            self.get_logger().error('导入 openvino 失败，请先安装: pip install openvino')
            self.get_logger().error(f'错误信息: {_OV_ERR}')
            raise RuntimeError('openvino 不可用')

        # 加载 OpenVINO 模型
        try:
            self._load_openvino_model()
            self.get_logger().info('OpenVINO 模型加载成功')
        except Exception as e:
            self.get_logger().error(f'OpenVINO 模型加载失败: {e}')
            raise

        # ROS 通信
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.ann_pub = self.create_publisher(Image, 'traffic_light/image_annotated', 10)
        self.status_pub = self.create_publisher(String, 'traffic_light/status', 10)
        self.boxes_pub = self.create_publisher(String, 'traffic_light/boxes', 10)
        self.stop_control_pub = self.create_publisher(Bool, 'traffic_light/stop_control', 10)

        # 统计
        self.frame_cnt = 0
        self.t0 = time.time()
        self.inf_hist: List[float] = []
        
        # 滤波相关变量
        self.detection_history: Deque[bool] = deque(maxlen=self.filter_window_size)
        self.current_stop_signal = False
        
        # 帧跳过机制 - 降低相机订阅频率，避免与SLAM冲突
        self.frame_skip_counter = 0
        self.frame_skip_rate = 3  # 只处理每第N帧 (1=处理所有帧, 2=跳过一半, 3=处理1/3)

    def _load_openvino_model(self) -> None:
        """加载 OpenVINO 模型"""
        # 查找模型文件
        xml_path = os.path.join(self.model_path, 'best.xml')
        bin_path = os.path.join(self.model_path, 'best.bin')
        metadata_path = os.path.join(self.model_path, 'metadata.yaml')

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f'模型文件不存在: {xml_path}')
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f'模型文件不存在: {bin_path}')

        # 加载类别名称
        self.class_names = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    if 'names' in metadata:
                        self.class_names = metadata['names']
                        self.get_logger().info(f'加载类别名称: {self.class_names}')
            except Exception as e:
                self.get_logger().warn(f'加载 metadata.yaml 失败: {e}，将使用数字作为类别名')

        # 初始化 OpenVINO
        self.ie = Core()
        
        # 读取模型
        self.get_logger().info(f'正在加载模型: {xml_path}')
        model = self.ie.read_model(model=xml_path)
        
        # 编译模型
        self.get_logger().info(f'正在编译模型到设备: {self.device}')
        self.compiled_model = self.ie.compile_model(model=model, device_name=self.device)
        
        # 获取输入输出信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 获取输入形状 (通常是 [1, 3, 640, 640])
        self.input_shape = self.input_layer.shape
        self.get_logger().info(f'模型输入形状: {self.input_shape}')
        self.get_logger().info(f'模型输出形状: {self.output_layer.shape}')
        
        # 提取输入尺寸
        if len(self.input_shape) == 4:
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
        else:
            self.get_logger().warn(f'意外的输入形状: {self.input_shape}，使用默认 640x640')
            self.input_height = 640
            self.input_width = 640

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            preprocessed: 预处理后的图像 [1, 3, H, W]
            scale: 缩放比例
            pad: 填充大小 (pad_w, pad_h)
        """
        # 保存原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 计算缩放比例（保持宽高比）
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充后的图像（灰色填充）
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # 计算填充位置（居中）
        pad_w = (self.input_width - new_w) // 2
        pad_h = (self.input_height - new_h) // 2
        
        # 将缩放后的图像放入填充图像中
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # 转换为 RGB 并归一化
        padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = padded_rgb.astype(np.float32) / 255.0
        
        # 转换为 CHW 格式并添加 batch 维度
        transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # CHW -> NCHW
        
        return batched, scale, (pad_w, pad_h)

    def _postprocess(self, output: np.ndarray, scale: float, pad: Tuple[int, int], 
                     orig_shape: Tuple[int, int]) -> List[Dict]:
        """
        后处理模型输出
        
        Args:
            output: 模型输出 [1, 84, 8400] 或 [1, num_classes+4, num_boxes]
            scale: 预处理时的缩放比例
            pad: 预处理时的填充 (pad_w, pad_h)
            orig_shape: 原始图像尺寸 (height, width)
            
        Returns:
            检测结果列表
        """
        dets = []
        
        try:
            # YOLOv8 输出格式: [1, 84, 8400] -> [1, num_classes+4, num_boxes]
            # 前4个是 [cx, cy, w, h]，后面是类别概率
            if len(output.shape) == 3:
                output = output[0]  # 移除 batch 维度 -> [84, 8400]
                output = output.T   # 转置 -> [8400, 84]
            
            # 分离坐标和类别分数
            boxes = output[:, :4]      # [num_boxes, 4] - cx, cy, w, h
            scores = output[:, 4:]     # [num_boxes, num_classes]
            
            # 获取每个框的最大类别分数和类别索引
            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # 过滤低置信度的检测
            mask = class_scores >= self.conf_threshold
            boxes = boxes[mask]
            class_scores = class_scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                return dets
            
            # 转换坐标格式：从 (cx, cy, w, h) 到 (x1, y1, x2, y2)
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            
            # 调整坐标到原始图像尺寸
            pad_w, pad_h = pad
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # 裁剪到图像边界
            orig_h, orig_w = orig_shape
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            # NMS (非极大值抑制)
            indices = self._nms(x1, y1, x2, y2, class_scores, iou_threshold=0.45)
            
            # 构建检测结果
            for idx in indices:
                cls_id = int(class_ids[idx])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                
                dets.append({
                    'class': cls_name,
                    'confidence': round(float(class_scores[idx]), 4),
                    'bbox': {
                        'x1': int(x1[idx]), 'y1': int(y1[idx]),
                        'x2': int(x2[idx]), 'y2': int(y2[idx])
                    }
                })
                
        except Exception as e:
            self.get_logger().warn(f'后处理失败: {e}')
            
        return dets

    def _nms(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, 
             scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """
        非极大值抑制 (NMS)
        
        Args:
            x1, y1, x2, y2: 边界框坐标
            scores: 置信度分数
            iou_threshold: IoU 阈值
            
        Returns:
            保留的索引列表
        """
        # 计算面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按分数降序排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算 IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留 IoU 小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def _apply_filter(self, should_stop: bool) -> bool:
        """应用滤波函数，防止偶发的错误识别导致小车停止"""
        self.detection_history.append(should_stop)
        
        if not self.detection_history:
            return False

        if len(self.detection_history) < self.filter_window_size:
            return False
            
        stop_count = sum(self.detection_history)
        total_count = len(self.detection_history)
        stop_ratio = stop_count / total_count
        
        filtered_result = stop_ratio >= (self.stop_threshold / 100.0)
        
        return filtered_result

    def _should_stop_traffic(self, status: str, dets: List[Dict]) -> bool:
        """根据检测结果判断是否应该停止小车"""
        if status == 'UNKNOWN':
            return False
        
        if 'RED' in status.upper():
            return True
            
        for det in dets:
            class_name = det['class'].upper()
            if 'STOP' in class_name:
                return True
                
        return False

    def _draw_detections(self, image: np.ndarray, dets: List[Dict]) -> np.ndarray:
        """在图像上绘制检测框"""
        annotated = image.copy()
        
        # 定义颜色映射
        color_map = {
            'RED': (0, 0, 255),      # 红色
            'GREEN': (0, 255, 0),    # 绿色
            'YELLOW': (0, 255, 255), # 黄色
            'STOP': (255, 0, 0),     # 蓝色
        }
        
        for det in dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # 选择颜色
            class_upper = det['class'].upper()
            color = (0, 255, 0)  # 默认绿色
            for key, val in color_map.items():
                if key in class_upper:
                    color = val
                    break
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1] + 10)
            
            cv2.rectangle(annotated, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0], label_y), color, -1)
            cv2.putText(annotated, label, (x1, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated

    def image_cb(self, msg: Image) -> None:
        """图像回调函数"""
        # 帧跳过机制 - 减少处理频率，避免与SLAM争夺相机资源
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_rate != 0:
            return  # 跳过这一帧
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 转换失败: {e}')
            return

        orig_shape = frame.shape[:2]
        
        # 预处理
        t_start = time.time()
        input_tensor, scale, pad = self._preprocess(frame)
        
        # OpenVINO 推理
        try:
            output = self.compiled_model([input_tensor])[self.output_layer]
        except Exception as e:
            self.get_logger().error(f'OpenVINO 推理失败: {e}')
            return
        
        # 后处理
        dets = self._postprocess(output, scale, pad, orig_shape)
        inf_ms = (time.time() - t_start) * 1000.0

        # 发布状态（最高置信度）
        status = 'UNKNOWN'
        if dets:
            best = max(dets, key=lambda d: d['confidence'])
            status = best['class'].upper()
        self.status_pub.publish(String(data=status))
        
        # 发布停止控制信号
        should_stop = self._should_stop_traffic(status, dets)
        filtered_stop_signal = self._apply_filter(should_stop)
        self.stop_control_pub.publish(Bool(data=filtered_stop_signal))
        self.current_stop_signal = filtered_stop_signal

        # 发布检测框 JSON
        try:
            self.boxes_pub.publish(String(data=json.dumps(dets, ensure_ascii=False)))
        except Exception as e:
            self.get_logger().warn(f'发布 boxes JSON 失败: {e}')

        # 发布标注图像
        if self.publish_annotated:
            try:
                ann = self._draw_detections(frame, dets)
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
            stop_signal_str = "STOP" if self.current_stop_signal else "GO"
            self.get_logger().info(
                f'FPS: {fps:.2f} | Avg inference: {avg_ms:.1f} ms | '
                f'Status: {status} | Dets: {len(dets)} | Control: {stop_signal_str}'
            )
            self.t0 = time.time()


def main():
    rclpy.init()
    node = None
    try:
        node = TrafficLightDetectorOpenVINO()
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

