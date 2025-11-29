#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header, Int32MultiArray

class ArucoDetectorROS2(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # --- Parameters ---
        # ROS 2 需要先声明参数
        self.declare_parameter("marker_size", 0.035)
        self.declare_parameter("aruco_dictionary_name", "DICT_5X5_250")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        self.declare_parameter("show_cv_window", False)
        self.declare_parameter("color_width", 1280)
        self.declare_parameter("color_height", 720)
        self.declare_parameter("color_fps", 30)

        # 获取参数
        self.marker_real_size_meters = self.get_parameter("marker_size").value
        aruco_dict_name_param = self.get_parameter("aruco_dictionary_name").value
        self.camera_frame_id = self.get_parameter("camera_frame_id").value
        self.show_cv_window = self.get_parameter("show_cv_window").value
        
        self.color_width = self.get_parameter("color_width").value
        self.color_height = self.get_parameter("color_height").value
        self.color_fps = self.get_parameter("color_fps").value

        # --- ArUco Dictionary Setup ---
        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None:
                raise AttributeError
        except AttributeError:
            self.get_logger().error(f"Invalid ArUco dictionary name: {aruco_dict_name_param}. Using DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        # 兼容不同版本的 OpenCV ArUco 参数初始化
        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create()

        # --- RealSense Initialization ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
        
        try:
            self.profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started.")
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            raise e

        # 获取内参
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics_color = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [self.intrinsics_color.fx, 0, self.intrinsics_color.ppx],
            [0, self.intrinsics_color.fy, self.intrinsics_color.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array(self.intrinsics_color.coeffs, dtype=np.float32)
        # 处理畸变系数
        if self.dist_coeffs is None or len(self.dist_coeffs) == 0:
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)
        elif not (len(self.dist_coeffs) in [4, 5, 8, 12, 14]):
            self.get_logger().warn(f"Unexpected number of distortion coefficients ({len(self.dist_coeffs)}). Using zero distortion.")
            self.dist_coeffs = np.zeros((5,1), dtype=np.float32)

        self.get_logger().info(f"Camera Matrix: \n{self.camera_matrix}")

        # --- ROS Publishers ---
        self.bridge = CvBridge()
        
        # 发布包含画框的处理后图像
        self.image_pub = self.create_publisher(Image, 'aruco_detector/image_processed', 10)
        
        # 发布位姿数组 (标准消息，包含所有检测到的 marker 位姿)
        self.pose_array_pub = self.create_publisher(PoseArray, 'aruco_detector/marker_poses', 10)
        
        # 发布 ID 数组 (与 pose_array 一一对应)
        self.id_array_pub = self.create_publisher(Int32MultiArray, 'aruco_detector/marker_ids', 10)

        # --- Timer ---
        # 使用定时器替代 while 循环，周期为 1/fps
        timer_period = 1.0 / self.color_fps
        self.timer = self.create_timer(timer_period, self.process_frame)

    def rotation_matrix_to_quaternion(self, rvec):
        """
        使用 scipy 将旋转向量 (Rodrigues) 转换为四元数 (x, y, z, w)
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        # scipy 返回的是 [x, y, z, w]
        return r.as_quat()

    def process_frame(self):
        # 1. 获取 RealSense 帧
        try:
            # non-blocking wait usually better in ROS2 loop, but strictly following user logic
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            self.get_logger().warn(f"Timeout waiting for frames: {e}")
            return

        color_frame_rs = frames.get_color_frame()
        if not color_frame_rs:
            return

        # 2. 图像转换
        color_image = np.asanyarray(color_frame_rs.get_data())
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 3. ArUco 检测
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_frame, self.dictionary, parameters=self.parameters
        )

        # 准备 ROS 消息头
        current_time = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = current_time
        header.frame_id = self.camera_frame_id

        # 用于绘制的副本
        display_image = color_image.copy()

        pose_array_msg = PoseArray()
        pose_array_msg.header = header
        
        id_array_msg = Int32MultiArray()

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            # 姿态估计
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            detected_ids = []

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]

                # 绘制坐标轴
                try:
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                except AttributeError:
                    aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                
                # 构建 Pose
                pose = Pose()
                pose.position.x = float(tvec[0][0])
                pose.position.y = float(tvec[0][1])
                pose.position.z = float(tvec[0][2])

                # 旋转向量 -> 四元数
                quat = self.rotation_matrix_to_quaternion(rvec)
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array_msg.poses.append(pose)
                detected_ids.append(int(marker_id))
            
            id_array_msg.data = detected_ids
            
            # 发布 Pose 和 ID
            self.pose_array_pub.publish(pose_array_msg)
            self.id_array_pub.publish(id_array_msg)

        # 4. 发布处理后的图像
        try:
            image_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            image_msg.header = header
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")

        # 5. 可视化窗口 (可选)
        if self.show_cv_window:
            cv2.imshow("ArUco Detection (ROS 2)", display_image)
            cv2.waitKey(1)

    def destroy_node(self):
        # 重写 destroy_node 以关闭 RealSense
        self.get_logger().info("Stopping RealSense pipeline...")
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.show_cv_window:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = ArucoDetectorROS2()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 这里的 detector 可能会因为初始化失败而未定义，稍微做个保护
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()