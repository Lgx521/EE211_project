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
