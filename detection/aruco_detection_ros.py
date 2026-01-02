#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header, Int32MultiArray

class ArucoDetectorROS2(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # --- Parameters ---
        self.declare_parameter("marker_size", 0.03)
        self.declare_parameter("aruco_dictionary_name", "DICT_5X5_250")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame") # 通常从消息header读取，这里作为备用或强制覆盖
        self.declare_parameter("show_cv_window", False)

        self.marker_real_size_meters = self.get_parameter("marker_size").value
        aruco_dict_name_param = self.get_parameter("aruco_dictionary_name").value
        self.image_topic = self.get_parameter("image_topic").value
        self.info_topic = self.get_parameter("info_topic").value
        self.target_frame_id = self.get_parameter("camera_frame_id").value
        self.show_cv_window = self.get_parameter("show_cv_window").value

        # --- ArUco Dictionary Setup ---
        try:
            self.aruco_dictionary_name = getattr(aruco, aruco_dict_name_param)
            if self.aruco_dictionary_name is None:
                raise AttributeError
        except AttributeError:
            self.get_logger().error(f"Invalid ArUco dictionary name: {aruco_dict_name_param}. Using DICT_6X6_250.")
            self.aruco_dictionary_name = aruco.DICT_6X6_250

        self.dictionary = aruco.getPredefinedDictionary(self.aruco_dictionary_name)
        
        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create()

        try:
            self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        except AttributeError:
            self.detector = None

        # --- Variables for Intrinsics ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics_received = False

        # --- ROS Utilities ---
        self.bridge = CvBridge()

        # --- ROS Publishers ---
        self.image_pub = self.create_publisher(Image, 'aruco_detector/image_processed', 10)
        self.pose_array_pub = self.create_publisher(PoseArray, 'aruco_detector/marker_poses', 10)
        self.id_array_pub = self.create_publisher(Int32MultiArray, 'aruco_detector/marker_ids', 10)

        # --- ROS Subscribers ---
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.camera_info_callback,
            10
        )

        # 2. 订阅图像数据
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f"Waiting for camera info on {self.info_topic}...")

    def camera_info_callback(self, msg):
        """
        回调函数：接收相机内参
        """
        if not self.intrinsics_received:
            # 将 9个元素的 K 列表转换为 3x3 矩阵
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            
            self.get_logger().info(f"Received Camera Matrix:\n{self.camera_matrix}")
            self.get_logger().info(f"Received Distortion Coeffs: {self.dist_coeffs}")
            self.intrinsics_received = True
        else:
            pass

    def rotation_matrix_to_quaternion(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        return r.as_quat()

    def image_callback(self, msg):
        """
        回调函数：主处理逻辑，当收到图像时触发
        """
        if not self.intrinsics_received:
            self.get_logger().warn_throttle(2.0, "Waiting for camera calibration info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        display_image = cv_image.copy()

        if self.detector is not None:
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray_frame)
        else:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray_frame, self.dictionary, parameters=self.parameters
            )

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id if msg.header.frame_id else self.target_frame_id

        pose_array_msg = PoseArray()
        pose_array_msg.header = header
        
        id_array_msg = Int32MultiArray()

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_image, corners, ids)
            
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_real_size_meters, self.camera_matrix, self.dist_coeffs
            )

            detected_ids = []

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]

                try:
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                except AttributeError:
                    aruco.drawAxis(display_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_real_size_meters / 2)
                
                pose = Pose()
                pose.position.x = float(tvec[0][0])
                pose.position.y = float(tvec[0][1])
                pose.position.z = float(tvec[0][2])

                quat = self.rotation_matrix_to_quaternion(rvec)
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                pose_array_msg.poses.append(pose)
                detected_ids.append(int(marker_id))
            
            id_array_msg.data = detected_ids
            
            self.pose_array_pub.publish(pose_array_msg)
            self.id_array_pub.publish(id_array_msg)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            out_msg.header = header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish Image Error: {e}")

        if self.show_cv_window:
            cv2.imshow("ArUco Detection (ROS 2)", display_image)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.show_cv_window:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoDetectorROS2()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
