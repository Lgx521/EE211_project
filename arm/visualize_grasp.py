#!/usr/bin/env python3
"""
抓取过程可视化工具
使用matplotlib绘制机械臂运动轨迹
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class ArmVisualizer:
    def __init__(self, L1=0.08945, L2=0.100, L3=0.100, L4=0.12915):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        
    def forward_kinematics(self, waist, shoulder, elbow, wrist):
        """计算各关节位置"""
        # 基座
        p0 = np.array([0, 0, 0])
        
        # 肩关节
        p1 = np.array([0, 0, self.L1])
        
        # 肘关节（在肩坐标系）
        r1 = self.L2 * math.cos(shoulder)
        z1 = self.L2 * math.sin(shoulder)
        p2_local = np.array([r1, 0, z1])
        
        # 腕关节
        r2 = self.L3 * math.cos(shoulder + elbow)
        z2 = self.L3 * math.sin(shoulder + elbow)
        p3_local = np.array([r1 + r2, 0, z1 + z2])
        
        # 末端
        pitch = shoulder + elbow + wrist
        r3 = self.L4 * math.cos(pitch)
        z3 = self.L4 * math.sin(pitch)
        p4_local = np.array([r1 + r2 + r3, 0, z1 + z2 + z3])
        
        # 旋转到世界坐标系（腰部旋转）
        cos_w = math.cos(waist)
        sin_w = math.sin(waist)
        
        def rotate_waist(p):
            return np.array([
                p[0] * cos_w - p[1] * sin_w,
                p[0] * sin_w + p[1] * cos_w,
                p[2] + self.L1
            ])
        
        p2 = rotate_waist(p2_local)
        p3 = rotate_waist(p3_local)
        p4 = rotate_waist(p4_local)
        
        return [p0, p1, p2, p3, p4]
    
    def plot_arm(self, ax, joints, color='b', label='', alpha=1.0):
        """绘制机械臂"""
        positions = self.forward_kinematics(*joints)
        
        # 绘制连杆
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   color=color, linewidth=2, alpha=alpha)
        
        # 绘制关节
        for i, p in enumerate(positions):
            ax.scatter(p[0], p[1], p[2], color=color, s=50, alpha=alpha)
        
        # 绘制末端标记
        end = positions[-1]
        ax.scatter(end[0], end[1], end[2], color=color, s=100, 
                  marker='*', label=label, alpha=alpha)
        
        return positions[-1]


def visualize_grasp_sequence(aruco_x=0.20, aruco_y=0.0, aruco_z=0.03):
    """可视化完整抓取序列"""
    from test_kinematics import PX100Kinematics
    
    ik = PX100Kinematics()
    vis = ArmVisualizer()
    
    # 计算各阶段的关节角度
    HOVER_OFFSET = 0.08
    GRASP_OFFSET = 0.02
    LIFT_OFFSET = 0.10
    
    z_hover = aruco_z + HOVER_OFFSET
    z_grasp = aruco_z + GRASP_OFFSET
    z_lift = aruco_z + LIFT_OFFSET
    
    joints_hover = ik.solve_ik(aruco_x, aruco_y, z_hover, pitch_goal=0.0)
    joints_grasp = ik.solve_ik(aruco_x, aruco_y, z_grasp, pitch_goal=0.0)
    joints_lift = ik.solve_ik(aruco_x, aruco_y, z_lift, pitch_goal=0.0)
    joints_default = [0.0, -0.5, 0.8, 0.0]
    
    if not (joints_hover and joints_grasp and joints_lift):
        print("错误: 某些位置IK无解")
        return
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 侧视图（XZ平面）
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title('侧视图 (XZ平面)', fontsize=12, fontweight='bold')
    
    # 绘制各阶段
    vis.plot_arm(ax1, joints_default, color='gray', label='初始位置', alpha=0.3)
    vis.plot_arm(ax1, joints_hover, color='blue', label='悬停', alpha=0.6)
    vis.plot_arm(ax1, joints_grasp, color='red', label='抓取', alpha=0.8)
    vis.plot_arm(ax1, joints_lift, color='green', label='提升', alpha=0.6)
    
    # 绘制物块
    ax1.scatter(aruco_x, aruco_y, aruco_z, color='orange', s=200, 
               marker='s', label='ArUco物块')
    
    # 设置视角和范围
    ax1.view_init(elev=10, azim=0)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_xlim([-0.1, 0.35])
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_zlim([0, 0.3])
    
    # 子图2: 俯视图（XY平面）
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_title('俯视图 (XY平面)', fontsize=12, fontweight='bold')
    
    vis.plot_arm(ax2, joints_default, color='gray', alpha=0.3)
    vis.plot_arm(ax2, joints_hover, color='blue', alpha=0.6)
    vis.plot_arm(ax2, joints_grasp, color='red', alpha=0.8)
    vis.plot_arm(ax2, joints_lift, color='green', alpha=0.6)
    ax2.scatter(aruco_x, aruco_y, aruco_z, color='orange', s=200, marker='s')
    
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim([-0.1, 0.35])
    ax2.set_ylim([-0.2, 0.2])
    ax2.set_zlim([0, 0.3])
    
    # 子图3: 3D透视图
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title('3D透视图', fontsize=12, fontweight='bold')
    
    vis.plot_arm(ax3, joints_default, color='gray', alpha=0.3)
    vis.plot_arm(ax3, joints_hover, color='blue', alpha=0.6)
    vis.plot_arm(ax3, joints_grasp, color='red', alpha=0.8)
    vis.plot_arm(ax3, joints_lift, color='green', alpha=0.6)
    ax3.scatter(aruco_x, aruco_y, aruco_z, color='orange', s=200, marker='s')
    
    ax3.view_init(elev=20, azim=45)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend()
    ax3.set_xlim([-0.1, 0.35])
    ax3.set_ylim([-0.2, 0.2])
    ax3.set_zlim([0, 0.3])
    
    # 子图4: 关节角度变化
    ax4 = fig.add_subplot(224)
    ax4.set_title('关节角度变化', fontsize=12, fontweight='bold')
    
    stages = ['初始', '悬停', '抓取', '提升']
    joints_list = [joints_default, joints_hover, joints_grasp, joints_lift]
    
    joint_names = ['腰部', '肩部', '肘部', '腕部']
    colors = ['red', 'green', 'blue', 'orange']
    
    for i in range(4):
        angles = [math.degrees(joints[i]) for joints in joints_list]
        ax4.plot(stages, angles, marker='o', label=joint_names[i], 
                color=colors[i], linewidth=2)
    
    ax4.set_ylabel('角度 (度)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 打印详细信息
    print("\n" + "=" * 60)
    print("抓取序列详细信息")
    print("=" * 60)
    print(f"\nArUco位置: ({aruco_x:.3f}, {aruco_y:.3f}, {aruco_z:.3f})")
    
    for stage, joints in zip(stages, joints_list):
        print(f"\n{stage}:")
        print(f"  关节角度(弧度): [{joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}]")
        print(f"  关节角度(度):   [{math.degrees(joints[0]):.1f}°, {math.degrees(joints[1]):.1f}°, "
              f"{math.degrees(joints[2]):.1f}°, {math.degrees(joints[3]):.1f}°]")
        
        # 计算末端位置
        positions = vis.forward_kinematics(*joints)
        end = positions[-1]
        print(f"  末端位置: ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f})")
    
    plt.show()


def plot_workspace():
    """绘制工作空间"""
    from test_kinematics import PX100Kinematics
    
    ik = PX100Kinematics()
    
    fig = plt.figure(figsize=(12, 5))
    
    # XZ平面工作空间
    ax1 = fig.add_subplot(121)
    ax1.set_title('工作空间 - XZ平面 (Y=0)', fontsize=12, fontweight='bold')
    
    x_range = np.linspace(0.05, 0.35, 30)
    z_range = np.linspace(0.0, 0.25, 25)
    
    reachable_x = []
    reachable_z = []
    unreachable_x = []
    unreachable_z = []
    
    for x in x_range:
        for z in z_range:
            joints = ik.solve_ik(x, 0.0, z, pitch_goal=0.0)
            if joints:
                reachable_x.append(x)
                reachable_z.append(z)
            else:
                unreachable_x.append(x)
                unreachable_z.append(z)
    
    ax1.scatter(reachable_x, reachable_z, c='green', s=10, alpha=0.5, label='可达')
    ax1.scatter(unreachable_x, unreachable_z, c='red', s=10, alpha=0.3, label='不可达')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # XY平面工作空间
    ax2 = fig.add_subplot(122)
    ax2.set_title('工作空间 - XY平面 (Z=0.05m)', fontsize=12, fontweight='bold')
    
    x_range = np.linspace(-0.3, 0.3, 30)
    y_range = np.linspace(-0.3, 0.3, 30)
    
    reachable_x = []
    reachable_y = []
    unreachable_x = []
    unreachable_y = []
    
    for x in x_range:
        for y in y_range:
            joints = ik.solve_ik(x, y, 0.05, pitch_goal=0.0)
            if joints:
                reachable_x.append(x)
                reachable_y.append(y)
            else:
                unreachable_x.append(x)
                unreachable_y.append(y)
    
    ax2.scatter(reachable_x, reachable_y, c='green', s=10, alpha=0.5, label='可达')
    ax2.scatter(unreachable_x, unreachable_y, c='red', s=10, alpha=0.3, label='不可达')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("机械臂抓取可视化工具")
    print("=" * 60)
    print("\n选项:")
    print("  1) 可视化抓取序列")
    print("  2) 绘制工作空间")
    print("  3) 自定义位置测试")
    
    choice = input("\n请选择 [1-3]: ")
    
    if choice == "1":
        print("\n使用默认ArUco位置: (0.20, 0.0, 0.03)")
        visualize_grasp_sequence()
    elif choice == "2":
        print("\n绘制工作空间...")
        plot_workspace()
    elif choice == "3":
        try:
            x = float(input("输入X坐标 (m): "))
            y = float(input("输入Y坐标 (m): "))
            z = float(input("输入Z坐标 (m): "))
            visualize_grasp_sequence(x, y, z)
        except ValueError:
            print("输入错误")
    else:
        print("无效选项")

