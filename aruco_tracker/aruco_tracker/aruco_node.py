import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        
        # --- 参数配置 ---
        # 你的 Aruco 标记的边长 (单位: 米)
        self.marker_length = 0.05  
        # 使用的字典，需与生成标记时一致
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # --- 这里的相机内参矩阵 (Intrinsics) 非常重要 ---
        # 如果你有真实的内参，请替换这里。如果没有，这里是一个大概的估算值。
        # 格式: np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.camera_matrix = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
        self.dist_coeffs = np.zeros(5) # 假设无畸变

        # --- 初始化工具 ---
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- 订阅相机 ---
        # 注意：这里的话题名 '/camera/color/image_raw' 可能需要根据你的实际情况修改
        self.subscription = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        self.get_logger().info("Aruco 检测节点已启动...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # 1. 检测标记
        corners, ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            # 2. 估计位姿
            # rvecs: 旋转向量, tvecs: 平移向量
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            # 3. 遍历检测到的每一个标记
            for i in range(len(ids)):
                # 在画面上画框（可选，用于调试）
                cv2.aruco.drawDetectedMarkers(cv_image, corners)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # 4. 发布 TF
                self.publish_tf(rvecs[i], tvecs[i], ids[i][0], msg.header)

        # 显示图像（调试用，正式运行时可注释掉）
        cv2.imshow("Aruco View", cv_image)
        cv2.waitKey(1)

    def publish_tf(self, rvec, tvec, marker_id, header):
        t = TransformStamped()
        
        # 时间戳必须和图像一致
        t.header.stamp = header.stamp
        # 父坐标系：相机的光学坐标系
        t.header.frame_id = header.frame_id 
        # 子坐标系：生成的标记坐标系
        t.child_frame_id = f"aruco_marker_{marker_id}"

        # 平移
        t.transform.translation.x = tvec[0][0]
        t.transform.translation.y = tvec[0][1]
        t.transform.translation.z = tvec[0][2]

        # 旋转 (Rodrigues向量 转 四元数)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat() # [x, y, z, w]

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()