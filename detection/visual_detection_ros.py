#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import time
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point

# 导入 SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("请先安装 segment_anything: pip install segment-anything")
    raise

class SamBoxDetectorCPU(Node):
    def __init__(self):
        super().__init__('sam_box_detector_node')

        # --- 参数设置 ---
        self.declare_parameter("box_side_length", 0.05)   # 盒子边长 5cm
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        
        # 模型配置 (针对 NUC CPU 优化)
        # 请修改为实际路径
        self.declare_parameter("sam_checkpoint", "/home/tony/ros2_ws/src/EE211_project/detection/sam_vit_b_01ec64.pth") 
        self.declare_parameter("model_type", "vit_b")     # 使用 vit_b
        self.declare_parameter("device", "cpu")           # 强制使用 cpu

        # HSV 白色阈值
        self.declare_parameter("hsv_h_min", 0)
        self.declare_parameter("hsv_h_max", 180)
        self.declare_parameter("hsv_s_min", 0)
        self.declare_parameter("hsv_s_max", 40)  # 白色饱和度低
        self.declare_parameter("hsv_v_min", 180) # 白色亮度高
        self.declare_parameter("hsv_v_max", 255)

        # 获取参数
        self.box_side_length = self.get_parameter("box_side_length").value
        self.image_topic = self.get_parameter("image_topic").value
        self.info_topic = self.get_parameter("info_topic").value
        checkpoint_path = self.get_parameter("sam_checkpoint").value
        model_type = self.get_parameter("model_type").value
        device = self.get_parameter("device").value

        # HSV 范围数组
        self.lower_white = np.array([
            self.get_parameter("hsv_h_min").value,
            self.get_parameter("hsv_s_min").value,
            self.get_parameter("hsv_v_min").value
        ])
        self.upper_white = np.array([
            self.get_parameter("hsv_h_max").value,
            self.get_parameter("hsv_s_max").value,
            self.get_parameter("hsv_v_max").value
        ])

        # --- 初始化 SAM ---
        self.get_logger().info(f"正在 CPU 上加载 SAM ({model_type})... 这可能需要一点时间")
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)
            self.get_logger().info("SAM 模型加载完成")
        except Exception as e:
            self.get_logger().error(f"加载 SAM 失败: {e}")
            exit(1)

        # --- 相机内参变量 ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics_received = False

        # --- PnP 3D 点 (以盒子中心为原点) ---
        s = self.box_side_length / 2.0
        # 顺序: 左上, 右上, 右下, 左下 (Z=0)
        self.object_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ], dtype=np.float32)

        # --- ROS 工具 ---
        self.bridge = CvBridge()
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseArray, '/aruco_detector/marker_poses', 10)
        self.mask_pub = self.create_publisher(Image, 'box_detector/debug_image', 10)

        # Subscribers
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 10)
        
        # 订阅图像 (QoS 设置为 1，尽量只拿最新的)
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 1)

        # 状态标志：防止 CPU 处理不过来导致延迟堆积
        self.is_processing = False

        self.get_logger().info("节点已启动，等待图像...")

    def camera_info_callback(self, msg):
        if not self.intrinsics_received:
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            self.intrinsics_received = True
            self.get_logger().info("收到相机内参")

    def rotation_matrix_to_quaternion(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        return r.as_quat()

    # def image_callback(self, msg):
    #     # 1. 如果正在处理上一帧，或者没收到内参，直接丢弃这一帧
    #     if self.is_processing:
    #         return
    #     if not self.intrinsics_received:
    #         return

    #     self.is_processing = True # 上锁
    #     start_time = time.time()

    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
    #         # --- 核心逻辑开始 ---
            
    #         # 缩小图像以加快 HSV 处理 (可选，SAM 内部反正会缩放到 1024)
    #         # scale = 0.5
    #         # input_image = cv2.resize(cv_image, (0,0), fx=scale, fy=scale)
    #         input_image = cv_image # 这里保持原图，保证 PnP 精度

    #         # A. HSV 寻找 Prompt
    #         hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    #         mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
            
    #         # 简单的形态学去噪
    #         kernel = np.ones((5,5), np.uint8)
    #         mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
            
    #         contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
    #         target_box = None
    #         max_area = 0
            
    #         # 找到最大的白色区域
    #         for cnt in contours:
    #             area = cv2.contourArea(cnt)
    #             if area > 1000: # 过滤小噪点
    #                 if area > max_area:
    #                     max_area = area
    #                     x, y, w, h = cv2.boundingRect(cnt)
    #                     # Box Prompt: [x1, y1, x2, y2]
    #                     target_box = np.array([x, y, x+w, y+h])

    #         display_image = input_image.copy()

    #         if target_box is not None:
    #             # B. SAM 推理 (这是最耗时的一步)
    #             # set_image 在 CPU 上可能会耗时 0.5s - 2s
    #             self.predictor.set_image(input_image)
                
    #             masks, _, _ = self.predictor.predict(
    #                 point_coords=None,
    #                 point_labels=None,
    #                 box=target_box[None, :], # 增加 batch 维度
    #                 multimask_output=False,
    #             )
                
    #             best_mask = masks[0]

    #             # C. 提取角点与 PnP
    #             mask_uint8 = (best_mask * 255).astype(np.uint8)
    #             mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
    #             if len(mask_contours) > 0:
    #                 largest_mask_cnt = max(mask_contours, key=cv2.contourArea)
                    
    #                 # 获取旋转矩形
    #                 rect = cv2.minAreaRect(largest_mask_cnt) 
    #                 box_points = cv2.boxPoints(rect)
    #                 box_points = np.float32(box_points)

    #                 # 排序角点 (逆时针/位置排序)
    #                 # 简单逻辑：计算重心，按角度排序
    #                 center = np.mean(box_points, axis=0)
    #                 sorted_inds = np.argsort(np.arctan2(box_points[:,1]-center[1], box_points[:,0]-center[0]))
    #                 sorted_box_points = box_points[sorted_inds]

    #                 # 绘制
    #                 display_image[best_mask] = display_image[best_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
    #                 for i, p in enumerate(sorted_box_points):
    #                     cv2.circle(display_image, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    #                     cv2.putText(display_image, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    #                 # ----------------- 修改开始 -----------------
                    
    #                 # 1. 保存 Mask 到本地用于 Debug
    #                 # 每次覆盖保存，避免硬盘填满。也可以加上时间戳。
    #                 debug_mask_path = '/tmp/sam_last_mask.png'
    #                 cv2.imwrite(debug_mask_path, mask_uint8)
    #                 # self.get_logger().info(f"Mask saved to {debug_mask_path}") # 可选：打印日志

    #                 # 2. 绘制可视化内容
    #                 # 绘制半透明绿色 Mask
    #                 display_image[best_mask] = display_image[best_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
                    
    #                 # 绘制角点和边长信息
    #                 for i in range(4):
    #                     p_start = sorted_box_points[i]
    #                     p_end = sorted_box_points[(i + 1) % 4] # 连接下一个点，形成闭环

    #                     # A. 绘制边框线
    #                     pt1 = (int(p_start[0]), int(p_start[1]))
    #                     pt2 = (int(p_end[0]), int(p_end[1]))
    #                     cv2.line(display_image, pt1, pt2, (0, 255, 255), 2) # 黄色边框

    #                     # B. 计算像素边长
    #                     dist_px = np.linalg.norm(p_start - p_end)

    #                     # C. 计算显示文字的位置 (边的中点)
    #                     mid_x = int((p_start[0] + p_end[0]) / 2)
    #                     mid_y = int((p_start[1] + p_end[1]) / 2)

    #                     # D. 在图像上显示像素长度 (例如 "120px")
    #                     # 如果需要显示物理长度，需要依赖 PnP 的结果反推，但这里显示像素长度更能反映视觉检测的原始质量
    #                     cv2.putText(display_image, f"{dist_px:.0f}px", (mid_x, mid_y), 
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
    #                     # 绘制角点序号
    #                     cv2.circle(display_image, pt1, 5, (0, 0, 255), -1)
    #                     cv2.putText(display_image, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    #                 # ----------------- 修改结束 -----------------

    #                 # SolvePnP (保持原有逻辑不变)
    #                 success, rvec, tvec = cv2.solvePnP(
    #                     self.object_points, 
    #                     sorted_box_points, 
    #                     self.camera_matrix, 
    #                     self.dist_coeffs,
    #                     flags=cv2.SOLVEPNP_ITERATIVE
    #                 )



    #                 # SolvePnP
    #                 success, rvec, tvec = cv2.solvePnP(
    #                     self.object_points, 
    #                     sorted_box_points, 
    #                     self.camera_matrix, 
    #                     self.dist_coeffs,
    #                     flags=cv2.SOLVEPNP_ITERATIVE
    #                 )

    #                 if success:
    #                     # 绘制坐标轴
    #                     cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

    #                     # 发布位姿
    #                     # pose_msg = PoseStamped()
    #                     # pose_msg.header = msg.header
    #                     # pose_msg.pose.position.x = float(tvec[0])
    #                     # pose_msg.pose.position.y = float(tvec[1])
    #                     # pose_msg.pose.position.z = float(tvec[2])

    #                     # quat = self.rotation_matrix_to_quaternion(rvec)
    #                     # pose_msg.pose.orientation.x = quat[0]
    #                     # pose_msg.pose.orientation.y = quat[1]
    #                     # pose_msg.pose.orientation.z = quat[2]
    #                     # pose_msg.pose.orientation.w = quat[3]
                        
    #                     # 修改为 (插入新逻辑)
    #                     pose_array = PoseArray()
    #                     pose_array.header = msg.header  # 保持时间戳同步

    #                     # 构建单个 Pose
    #                     pose = Pose()
    #                     pose.position.x = float(tvec[0])
    #                     pose.position.y = float(tvec[1])
    #                     pose.position.z = float(tvec[2])

    #                     quat = self.rotation_matrix_to_quaternion(rvec)
    #                     pose.orientation.x = quat[0]
    #                     pose.orientation.y = quat[1]
    #                     pose.orientation.z = quat[2]
    #                     pose.orientation.w = quat[3]

    #                     # 将 Pose 加入数组 (虽然只有一个盒子，但格式需要是列表)
    #                     pose_array.poses.append(pose)

    #                     self.pose_pub.publish(pose_array)
                        
    #                     # 打印帧率调试
    #                     process_time = time.time() - start_time
    #                     self.get_logger().info(f"Detected! Z={tvec[2][0]:.2f}m. Time: {process_time:.3f}s (FPS: {1/process_time:.1f})")

    #         # 发布 Debug 图像
    #         out_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
    #         out_msg.header = msg.header
    #         self.mask_pub.publish(out_msg)

    #     except Exception as e:
    #         self.get_logger().error(f"处理错误: {e}")
    #     finally:
    #         # 释放锁，允许处理下一帧
    #         self.is_processing = False
    
    def image_callback(self, msg):
        # 1. 性能保护：如果上一帧还在处理，或者没收到内参，直接丢弃
        if self.is_processing:
            return
        if not self.intrinsics_received:
            return

        self.is_processing = True # 上锁
        start_time = time.time()

        try:
            # 转换 ROS 图像 -> OpenCV 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            full_h, full_w = cv_image.shape[:2]
            
            # 用于发布的调试图 (在原图基础上绘制)
            display_image = cv_image.copy()

            # ---------------------------------------------------------
            # 2. 定义中心裁剪区域 (ROI) - 仅在中心区域搜索
            # ---------------------------------------------------------
            # 比例系数: 0.6 表示只取中心 60% 的区域
            roi_scale = 0.4
            
            roi_w = int(full_w * roi_scale)
            roi_h = int(full_h * roi_scale)
            roi_x = int((full_w - roi_w) / 2) # ROI 左上角 X
            roi_y = int((full_h - roi_h) / 2) # ROI 左上角 Y

            # 绘制 ROI 蓝色框，指示探测范围
            cv2.rectangle(display_image, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)

            # 裁剪出中心图像 (输入给 SAM，像素少，速度快)
            roi_image = cv_image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            # ---------------------------------------------------------
            # 3. HSV 预处理 (在 ROI 内寻找白色 Prompt)
            # ---------------------------------------------------------
            hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
            mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
            
            # 形态学去噪
            kernel = np.ones((5,5), np.uint8)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_box_roi = None
            max_area = 0
            
            # 寻找 ROI 中最大的白色块
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500: # 面积阈值 (基于 ROI 大小)
                    if area > max_area:
                        max_area = area
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 这里得到的是相对于 roi_image 的坐标
                        target_box_roi = np.array([x, y, x+w, y+h])

            if target_box_roi is not None:
                # -----------------------------------------------------
                # 4. SAM 推理 (仅对 ROI 图像)
                # -----------------------------------------------------
                self.predictor.set_image(roi_image)
                
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=target_box_roi[None, :], 
                    multimask_output=False,
                )
                
                # 这里的 mask 尺寸是 ROI 的尺寸
                best_mask = masks[0] 

                # Debug: 保存 Mask 到本地 (每次覆盖)
                debug_mask_path = '/tmp/sam_roi_mask.png'
                # 将 mask 转为 0-255 图片保存
                cv2.imwrite(debug_mask_path, (best_mask * 255).astype(np.uint8))

                # -----------------------------------------------------
                # 5. 提取轮廓与坐标还原
                # -----------------------------------------------------
                mask_uint8 = (best_mask * 255).astype(np.uint8)
                mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(mask_contours) > 0:
                    largest_mask_cnt = max(mask_contours, key=cv2.contourArea)
                    
                    # A. 获取最小外接矩形 (ROI 坐标系)
                    rect = cv2.minAreaRect(largest_mask_cnt) 
                    box_points_roi = cv2.boxPoints(rect)
                    box_points_roi = np.float32(box_points_roi)

                    # B. *** 坐标还原 ***: ROI 坐标 -> 全图坐标
                    # 加上 ROI 左上角的偏移量
                    box_points_full = box_points_roi + np.array([roi_x, roi_y])

                    # C. 角点排序 (逆时针，确保与 3D 点对应)
                    center = np.mean(box_points_full, axis=0)
                    sorted_inds = np.argsort(np.arctan2(box_points_full[:,1]-center[1], box_points_full[:,0]-center[0]))
                    sorted_box_points = box_points_full[sorted_inds]

                    # -------------------------------------------------
                    # 6. 可视化绘制 (在全图上)
                    # -------------------------------------------------
                    for i in range(4):
                        p_start = sorted_box_points[i]
                        p_end = sorted_box_points[(i + 1) % 4]
                        
                        pt1 = (int(p_start[0]), int(p_start[1]))
                        pt2 = (int(p_end[0]), int(p_end[1]))
                        
                        # 绘制黄色边框
                        cv2.line(display_image, pt1, pt2, (0, 255, 255), 2)
                        
                        # 计算并显示像素边长
                        dist_px = np.linalg.norm(p_start - p_end)
                        mid_pt = (int((p_start[0]+p_end[0])/2), int((p_start[1]+p_end[1])/2))
                        cv2.putText(display_image, f"{dist_px:.0f}px", mid_pt, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # 标出角点序号
                        cv2.putText(display_image, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                    # -------------------------------------------------
                    # 7. Pose 估计 (PnP)
                    # -------------------------------------------------
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points, 
                        sorted_box_points,  # 必须使用全图坐标
                        self.camera_matrix, 
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        # 绘制坐标轴
                        cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                        
                        # 构建 PoseArray 消息
                        pose_array = PoseArray()
                        pose_array.header = msg.header # 继承时间戳
                        
                        pose = Pose()
                        pose.position.x = float(tvec[0])
                        pose.position.y = float(tvec[1])
                        pose.position.z = float(tvec[2])
                        
                        quat = self.rotation_matrix_to_quaternion(rvec)
                        pose.orientation.x = quat[0]
                        pose.orientation.y = quat[1]
                        pose.orientation.z = quat[2]
                        pose.orientation.w = quat[3]
                        
                        pose_array.poses.append(pose)
                        
                        # 发布 Pose
                        self.pose_pub.publish(pose_array)
                        
                        # 打印日志
                        process_time = time.time() - start_time
                        self.get_logger().info(f"Box Found! Z={tvec[2][0]:.2f}m. Time: {process_time:.3f}s")

            # 发布处理后的 Debug 图像
            out_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            out_msg.header = msg.header
            self.mask_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 释放锁，允许处理下一帧
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = SamBoxDetectorCPU()
    try:
        # 使用 MultiThreadedExecutor 可能更好，但这里用 spin 足以
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()