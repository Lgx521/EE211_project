#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 红绿灯检测节点（trafficlight_detection_debug.py）

功能概要：
- 【调试版】包含详细的性能分析和调试输出，用于定位性能瓶颈
- 在推理前缩小图像尺寸，并使用半精度(FP16)推理，以大幅提升帧率。
- 将检测框坐标重新缩放回原始图像尺寸，确保坐标准确性。
- 订阅相机图像并使用 YOLO 最优模型 best.pt 实时推理。

快速使用：
   cd ~/ros2_ws/src/EE211_project/detection
   python3 trafficlight_detection_debug.py --ros-args \
     -p model_path:=/home/tony/Yolo/runs/detect/traffic_detection3/weights/best.pt \
     -p conf_threshold:=0.5 \
     -p image_topic:=/camera/camera/color/image_raw \
     -p inference_size:=320
"""

import os
import glob
import time
import json
from typing import List, Dict, Deque
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _UL_OK = True
except Exception as e:
    _UL_OK = False
    _UL_ERR = str(e)

try:
    import torch
    _TORCH_OK = True
except Exception as e:
    _TORCH_OK = False
    _TORCH_ERR = str(e)


class TrafficLightDetectorDebug(Node):
    def __init__(self) -> None:
        super().__init__('traffic_light_detector_debug')

        # 参数
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('model_path', '/home/tony/ros2_ws/src/EE211_project/detection/best1218.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('filter_window_size', 5)
        self.declare_parameter('stop_threshold', 60)
        self.declare_parameter('inference_size', 320)  # 新增：推理图像尺寸（更小=更快）
        self.declare_parameter('use_half', True)  # 新增：是否使用半精度
        self.declare_parameter('detailed_log_interval', 10)  # 详细日志间隔（帧数）

        self.image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold: float = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.publish_annotated: bool = self.get_parameter('publish_annotated').get_parameter_value().bool_value
        self.filter_window_size: int = self.get_parameter('filter_window_size').get_parameter_value().integer_value
        self.stop_threshold: int = self.get_parameter('stop_threshold').get_parameter_value().integer_value
        self.inference_size: int = self.get_parameter('inference_size').get_parameter_value().integer_value
        self.use_half: bool = self.get_parameter('use_half').get_parameter_value().bool_value
        self.detailed_log_interval: int = self.get_parameter('detailed_log_interval').get_parameter_value().integer_value

        self.get_logger().info('================ Traffic Light Detection (DEBUG) ================')
        self.get_logger().info(f'image_topic: {self.image_topic}')
        self.get_logger().info(f'model_path: {self.model_path}')
        self.get_logger().info(f'conf_threshold: {self.conf_threshold}')
        self.get_logger().info(f'publish_annotated: {self.publish_annotated}')
        self.get_logger().info(f'filter_window_size: {self.filter_window_size}')
        self.get_logger().info(f'stop_threshold: {self.stop_threshold}')
        self.get_logger().info(f'inference_size: {self.inference_size}')
        self.get_logger().info(f'use_half: {self.use_half}')
        self.get_logger().info('==================================================================')

        # 检查 PyTorch 和 CUDA
        if _TORCH_OK:
            self.get_logger().info(f'PyTorch 版本: {torch.__version__}')
            self.get_logger().info(f'CUDA 可用: {torch.cuda.is_available()}')
            if torch.cuda.is_available():
                self.get_logger().info(f'CUDA 版本: {torch.version.cuda}')
                self.get_logger().info(f'GPU 设备: {torch.cuda.get_device_name(0)}')
                self.get_logger().info(f'GPU 数量: {torch.cuda.device_count()}')
            else:
                self.get_logger().warn('⚠️ CUDA 不可用，将使用 CPU 进行推理（速度会很慢）')
        else:
            self.get_logger().warn(f'PyTorch 不可用: {_TORCH_ERR}')

        if not _UL_OK:
            self.get_logger().error('导入 ultralytics 失败，请先安装: pip install ultralytics')
            self.get_logger().error(f'错误信息: {_UL_ERR}')
            raise RuntimeError('ultralytics 不可用')

        # 加载模型
        t_model_start = time.time()
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().warn(f'指定模型不存在: {self.model_path}，尝试自动搜索最新 best.pt')
                self.model_path = self._auto_find_best()
                self.get_logger().info(f'自动选择模型: {self.model_path}')
            
            self.get_logger().info(f'正在加载模型: {self.model_path}')
            self.model = YOLO(self.model_path)
            
            # 获取模型信息
            model_info = self.model.info(verbose=False)
            self.get_logger().info(f'模型加载耗时: {(time.time() - t_model_start):.2f} 秒')
            self.get_logger().info(f'模型类型: {type(self.model.model).__name__}')
            
            # 检查模型设备
            if _TORCH_OK and hasattr(self.model, 'device'):
                self.get_logger().info(f'模型设备: {self.model.device}')
            
        except Exception as e:
            self.get_logger().error(f'YOLO 模型加载失败: {e}')
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
        
        # 详细计时统计
        self.timing_stats = {
            'bridge_convert': [],
            'resize': [],
            'inference': [],
            'extract': [],
            'publish_status': [],
            'publish_boxes': [],
            'publish_annotated': [],
            'total_callback': []
        }
        
        # 滤波
        self.detection_history: Deque[bool] = deque(maxlen=self.filter_window_size)
        self.current_stop_signal = False
        
        self.get_logger().info('✅ 节点初始化完成，等待图像数据...')

    def _auto_find_best(self) -> str:
        home = os.path.expanduser('~')
        candidates = glob.glob(os.path.join(home, 'Yolo', 'runs', 'detect', '*', 'weights', 'best.pt'))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
        return 'best.pt'

    def _apply_filter(self, should_stop: bool) -> bool:
        self.detection_history.append(should_stop)
        if not self.detection_history or len(self.detection_history) < self.filter_window_size:
            return should_stop
        stop_ratio = sum(self.detection_history) / len(self.detection_history)
        filtered_result = stop_ratio >= (self.stop_threshold / 100.0)
        return filtered_result

    def _should_stop_traffic(self, status: str, dets: List[Dict]) -> bool:
        if 'RED' in status or any('STOP' in det['class'].upper() for det in dets):
            return True
        return False

    def image_cb(self, msg: Image) -> None:
        callback_start = time.time()
        
        # 步骤1: CvBridge 转换
        t1 = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 转换失败: {e}')
            return
        t_bridge = (time.time() - t1) * 1000.0
        
        # 记录原始图像尺寸（仅第一帧）
        if self.frame_cnt == 0:
            self.get_logger().info(f'📷 原始图像尺寸: {frame.shape[1]}x{frame.shape[0]} (WxH)')
            self.get_logger().info(f'📷 推理图像尺寸: {self.inference_size}x{self.inference_size}')

        # 步骤2: 图像缩放
        t2 = time.time()
        resized_frame = cv2.resize(frame, (self.inference_size, self.inference_size), interpolation=cv2.INTER_LINEAR)
        t_resize = (time.time() - t2) * 1000.0

        # 步骤3: YOLO 推理
        t3 = time.time()
        try:
            results = self.model.predict(
                resized_frame, 
                conf=self.conf_threshold, 
                verbose=False, 
                half=self.use_half
            )
        except Exception as e:
            self.get_logger().error(f'YOLO 推理失败: {e}')
            if 'half' in str(e).lower() or 'fp16' in str(e).lower():
                self.get_logger().error('⚠️ 半精度推理失败，可能是 CPU 不支持或模型问题')
                self.get_logger().error('建议: 设置参数 -p use_half:=False')
            return
        t_inference = (time.time() - t3) * 1000.0

        # 步骤4: 提取检测结果
        t4 = time.time()
        dets = self._extract(results[0], frame.shape, resized_frame.shape)
        t_extract = (time.time() - t4) * 1000.0

        # 步骤5: 发布状态
        t5 = time.time()
        status = 'UNKNOWN'
        if dets:
            best = max(dets, key=lambda d: d['confidence'])
            status = best['class'].upper()
        self.status_pub.publish(String(data=status))
        
        should_stop = self._should_stop_traffic(status, dets)
        filtered_stop_signal = self._apply_filter(should_stop)
        self.stop_control_pub.publish(Bool(data=filtered_stop_signal))
        self.current_stop_signal = filtered_stop_signal
        t_publish_status = (time.time() - t5) * 1000.0

        # 步骤6: 发布检测框
        t6 = time.time()
        try:
            self.boxes_pub.publish(String(data=json.dumps(dets, ensure_ascii=False)))
        except Exception as e:
            self.get_logger().warn(f'发布 boxes JSON 失败: {e}')
        t_publish_boxes = (time.time() - t6) * 1000.0

        # 步骤7: 发布标注图像
        t7 = time.time()
        if self.publish_annotated:
            try:
                ann = results[0].plot()
                img_msg = self.bridge.cv2_to_imgmsg(ann, encoding='bgr8')
                img_msg.header = msg.header
                self.ann_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().warn(f'发布标注图像失败: {e}')
        t_publish_annotated = (time.time() - t7) * 1000.0

        # 总回调时间
        t_total_callback = (time.time() - callback_start) * 1000.0

        # 记录统计
        self.timing_stats['bridge_convert'].append(t_bridge)
        self.timing_stats['resize'].append(t_resize)
        self.timing_stats['inference'].append(t_inference)
        self.timing_stats['extract'].append(t_extract)
        self.timing_stats['publish_status'].append(t_publish_status)
        self.timing_stats['publish_boxes'].append(t_publish_boxes)
        self.timing_stats['publish_annotated'].append(t_publish_annotated)
        self.timing_stats['total_callback'].append(t_total_callback)
        
        # 保持统计队列长度
        for key in self.timing_stats:
            if len(self.timing_stats[key]) > 30:
                self.timing_stats[key].pop(0)

        self.frame_cnt += 1
        self.inf_hist.append(t_inference)
        if len(self.inf_hist) > 30:
            self.inf_hist.pop(0)

        # 详细日志输出
        if self.frame_cnt % self.detailed_log_interval == 0:
            dt = time.time() - self.t0
            fps = self.detailed_log_interval / dt if dt > 0 else 0.0
            
            self.get_logger().info('=' * 80)
            self.get_logger().info(f'📊 性能统计 (第 {self.frame_cnt} 帧)')
            self.get_logger().info(f'   FPS: {fps:.2f}')
            self.get_logger().info(f'   检测数量: {len(dets)} | 状态: {status} | 控制: {"STOP" if self.current_stop_signal else "GO"}')
            self.get_logger().info('-' * 80)
            self.get_logger().info('⏱️  各步骤平均耗时 (ms):')
            self.get_logger().info(f'   1. CvBridge 转换:    {np.mean(self.timing_stats["bridge_convert"]):.2f} ms')
            self.get_logger().info(f'   2. 图像缩放:         {np.mean(self.timing_stats["resize"]):.2f} ms')
            self.get_logger().info(f'   3. YOLO 推理:        {np.mean(self.timing_stats["inference"]):.2f} ms  ⚠️ 主要瓶颈')
            self.get_logger().info(f'   4. 结果提取:         {np.mean(self.timing_stats["extract"]):.2f} ms')
            self.get_logger().info(f'   5. 发布状态:         {np.mean(self.timing_stats["publish_status"]):.2f} ms')
            self.get_logger().info(f'   6. 发布检测框:       {np.mean(self.timing_stats["publish_boxes"]):.2f} ms')
            self.get_logger().info(f'   7. 发布标注图像:     {np.mean(self.timing_stats["publish_annotated"]):.2f} ms')
            self.get_logger().info(f'   总回调时间:          {np.mean(self.timing_stats["total_callback"]):.2f} ms')
            self.get_logger().info('-' * 80)
            
            # 计算各部分占比
            total_avg = np.mean(self.timing_stats["total_callback"])
            if total_avg > 0:
                inference_pct = (np.mean(self.timing_stats["inference"]) / total_avg) * 100
                annotated_pct = (np.mean(self.timing_stats["publish_annotated"]) / total_avg) * 100
                self.get_logger().info(f'📈 时间占比:')
                self.get_logger().info(f'   推理占比: {inference_pct:.1f}%')
                self.get_logger().info(f'   标注图像占比: {annotated_pct:.1f}%')
            
            # 优化建议
            if np.mean(self.timing_stats["inference"]) > 100:
                self.get_logger().warn('⚠️ 推理时间过长 (>100ms)，优化建议:')
                self.get_logger().warn('   1. 减小推理尺寸: -p inference_size:=224 或 160')
                self.get_logger().warn('   2. 确保使用 GPU (检查上方 CUDA 信息)')
                self.get_logger().warn('   3. 使用更小的模型 (yolov8n.pt 而不是 yolov8x.pt)')
                
            if np.mean(self.timing_stats["publish_annotated"]) > 20 and self.publish_annotated:
                self.get_logger().warn('⚠️ 标注图像发布耗时较长，可以关闭: -p publish_annotated:=False')
            
            self.get_logger().info('=' * 80)
            self.t0 = time.time()

    def _extract(self, result, original_shape, resized_shape) -> List[Dict]:
        dets: List[Dict] = []
        try:
            boxes = getattr(result, 'boxes', None)
            names = getattr(result, 'names', {})
            if boxes is None:
                return dets

            # 计算缩放比例
            gain_w = original_shape[1] / resized_shape[1]
            gain_h = original_shape[0] / resized_shape[0]

            for b in boxes:
                xyxy_resized = b.xyxy[0].cpu().numpy()
                
                # 重新缩放坐标到原始图像尺寸
                xyxy_original = [
                    xyxy_resized[0] * gain_w,
                    xyxy_resized[1] * gain_h,
                    xyxy_resized[2] * gain_w,
                    xyxy_resized[3] * gain_h
                ]

                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id))
                
                dets.append({
                    'class': cls_name,
                    'confidence': round(conf, 4),
                    'bbox': {
                        'x1': int(xyxy_original[0]), 'y1': int(xyxy_original[1]),
                        'x2': int(xyxy_original[2]), 'y2': int(xyxy_original[3])
                    }
                })
        except Exception as e:
            self.get_logger().warn(f'解析YOLO结果失败: {e}')
        return dets


def main():
    rclpy.init()
    node = None
    try:
        node = TrafficLightDetectorDebug()
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

