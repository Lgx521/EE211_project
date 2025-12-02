import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import threading
import sys

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.topic_name = '/camera/camera/color/image_raw'
        self.base_dir = os.path.expanduser('dataset')
        
        # 实时存储最新的一帧图像，供主线程调用
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.listener_callback,
            10)
        self.br = CvBridge()
        
        self.init_directories()

    def init_directories(self):
        categories = ['red_light', 'green_light', 'stop_sign', 'negative']
        for cat in categories:
            path = os.path.join(self.base_dir, cat)
            os.makedirs(path, exist_ok=True)

    def listener_callback(self, data):
        try:
            # 只要把图转好存起来，不做任何显示
            frame = self.br.imgmsg_to_cv2(data, "bgr8")
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def save_image(self, category):
        with self.frame_lock:
            if self.latest_frame is None:
                print("⚠️ 还没收到图像，请检查相机话题是否正确！")
                return

            timestamp = int(time.time() * 1000)
            filename = f"{category}_{timestamp}.jpg"
            save_path = os.path.join(self.base_dir, category, filename)
            cv2.imwrite(save_path, self.latest_frame)
            print(f"✅ 已保存到 [{category}]: {filename}")

def main():
    rclpy.init()
    node = DataCollector()
    
    # 1. 启动一个后台线程来处理 ROS 回调（接收图像）
    # 这样主线程就可以卡在 input() 等待你输入，而不会阻塞图像接收
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    print("-" * 40)
    print(" 🤖 终端控制模式启动 (无需点击图像窗口)")
    print(" 输入以下字符并回车保存当前画面：")
    print(" r : 红灯 (Red)")
    print(" g : 绿灯 (Green)")
    print(" s : Stop标志")
    print(" n : 负样本 (Negative)")
    print(" q : 退出")
    print("-" * 40)

    try:
        while rclpy.ok():
            # 2. 在主线程等待用户输入
            cmd = input("请输入指令 (r/g/s/n) > ").strip().lower()
            
            if cmd == 'r':
                node.save_image('red_light')
            elif cmd == 'g':
                node.save_image('green_light')
            elif cmd == 's':
                node.save_image('stop_sign')
            elif cmd == 'n':
                node.save_image('negative')
            elif cmd == 'q':
                print("退出程序...")
                break
            else:
                pass # 忽略无效输入

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()