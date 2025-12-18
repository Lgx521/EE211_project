#!/usr/bin/env python3
"""
运动学测试脚本
用于验证IK求解器的正确性和工作空间范围
"""

import math
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入运动学类（需要修改导入方式）
class PX100Kinematics:
    def __init__(self,
                 L1=0.08945,   # 基座到肩关节高度 (m)
                 L2=0.100,     # 肩到肘 (m)
                 L3=0.100,     # 肘到腕 (m)
                 L4=0.12915,   # 腕到指尖/抓取TCP (m)
                 grasp_x_offset=0.035):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.GRASP_X_OFFSET = grasp_x_offset

    def solve_ik(self, x, y, z, pitch_goal=0.0):
        waist = math.atan2(y, x)
        r_target_raw = math.sqrt(x**2 + y**2)
        r_target = max(0.0, r_target_raw - self.GRASP_X_OFFSET)
        z_w = z - self.L1

        # 策略A: 水平抓取
        r_wrist = r_target - self.L4 * math.cos(pitch_goal)
        z_wrist = z_w - self.L4 * math.sin(pitch_goal)
        joints = self._compute_2link_ik(r_wrist, z_wrist, pitch_goal=pitch_goal)
        if joints:
            return [waist] + joints

        # 策略B: 指向抓取
        angle_to_origin = math.atan2(z_w, r_target)
        r_wrist_p = r_target - (self.L4 * math.cos(angle_to_origin))
        z_wrist_p = z_w - (self.L4 * math.sin(angle_to_origin))
        joints = self._compute_2link_ik(r_wrist_p, z_wrist_p, pitch_goal=angle_to_origin)
        if joints:
            return [waist] + joints

        return None

    def _compute_2link_ik(self, r, z, pitch_goal):
        d = math.sqrt(r**2 + z**2)
        if d > (self.L2 + self.L3) or d < abs(self.L2 - self.L3):
            return None

        alpha = math.atan2(z, r)
        try:
            cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
            if abs(cos_beta) > 1.0:
                return None
            beta = math.acos(cos_beta)
            theta_shoulder = alpha + beta

            cos_gamma = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
            if abs(cos_gamma) > 1.0:
                return None
            gamma = math.acos(cos_gamma)
            theta_elbow = -(math.pi - gamma)

            theta_wrist = pitch_goal - theta_shoulder - theta_elbow

            if not (-2.0 < theta_shoulder < 2.0):
                return None
            if not (-2.4 < theta_elbow < 2.4):
                return None
            if not (-2.0 < theta_wrist < 2.0):
                return None

            return [theta_shoulder, theta_elbow, theta_wrist]
        except ValueError:
            return None

    def forward_kinematics(self, waist, shoulder, elbow, wrist):
        """正运动学：从关节角度计算末端位置"""
        # 计算在肩坐标系中的位置
        r1 = self.L2 * math.cos(shoulder)
        z1 = self.L2 * math.sin(shoulder)
        
        r2 = self.L3 * math.cos(shoulder + elbow)
        z2 = self.L3 * math.sin(shoulder + elbow)
        
        pitch = shoulder + elbow + wrist
        r3 = self.L4 * math.cos(pitch)
        z3 = self.L4 * math.sin(pitch)
        
        r_total = r1 + r2 + r3
        z_total = z1 + z2 + z3
        
        # 转换到基座坐标系
        x = (r_total + self.GRASP_X_OFFSET) * math.cos(waist)
        y = (r_total + self.GRASP_X_OFFSET) * math.sin(waist)
        z = z_total + self.L1
        
        return x, y, z, pitch


def test_workspace():
    """测试工作空间范围"""
    print("=" * 60)
    print("机械臂工作空间测试")
    print("=" * 60)
    
    ik = PX100Kinematics()
    
    print(f"\n运动学参数:")
    print(f"  L1 (基座高度): {ik.L1*1000:.2f} mm")
    print(f"  L2 (肩到肘):   {ik.L2*1000:.2f} mm")
    print(f"  L3 (肘到腕):   {ik.L3*1000:.2f} mm")
    print(f"  L4 (腕到TCP):  {ik.L4*1000:.2f} mm")
    print(f"  最大水平伸展:  {(ik.L2 + ik.L3 + ik.L4)*1000:.2f} mm")
    print(f"  最小水平伸展:  {abs(ik.L2 - ik.L3) + ik.L4:.2f} mm")
    
    # 测试典型位置
    test_positions = [
        # (x, y, z, 描述)
        (0.15, 0.0, 0.05, "前方近距离"),
        (0.20, 0.0, 0.05, "前方中距离"),
        (0.25, 0.0, 0.05, "前方远距离"),
        (0.15, 0.05, 0.05, "右前方"),
        (0.15, -0.05, 0.05, "左前方"),
        (0.20, 0.0, 0.10, "前方高位"),
        (0.20, 0.0, 0.02, "前方低位"),
    ]
    
    print("\n" + "=" * 60)
    print("典型位置IK测试")
    print("=" * 60)
    
    for x, y, z, desc in test_positions:
        joints = ik.solve_ik(x, y, z, pitch_goal=0.0)
        if joints:
            # 验证正运动学
            x_fk, y_fk, z_fk, pitch_fk = ik.forward_kinematics(*joints)
            error = math.sqrt((x-x_fk)**2 + (y-y_fk)**2 + (z-z_fk)**2)
            
            print(f"\n{desc}: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  ✓ IK解: [{joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}] (弧度)")
            print(f"    关节角度(度): [{math.degrees(joints[0]):.1f}°, {math.degrees(joints[1]):.1f}°, "
                  f"{math.degrees(joints[2]):.1f}°, {math.degrees(joints[3]):.1f}°]")
            print(f"  FK验证: ({x_fk:.3f}, {y_fk:.3f}, {z_fk:.3f}), 误差: {error*1000:.2f} mm")
        else:
            print(f"\n{desc}: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  ✗ IK无解（超出工作空间）")


def test_vertical_grasp():
    """测试垂直抓取场景"""
    print("\n" + "=" * 60)
    print("垂直抓取场景测试")
    print("=" * 60)
    
    ik = PX100Kinematics()
    
    # 模拟ArUco在物块顶部的位置
    aruco_positions = [
        (0.18, 0.0, 0.03, "正前方物块"),
        (0.20, 0.05, 0.03, "右前方物块"),
        (0.22, 0.0, 0.05, "较高物块"),
    ]
    
    HOVER_OFFSET = 0.08
    GRASP_OFFSET = 0.02
    
    for x, y, z_aruco, desc in aruco_positions:
        print(f"\n{desc} - ArUco位置: ({x:.3f}, {y:.3f}, {z_aruco:.3f})")
        
        # 测试悬停位置
        z_hover = z_aruco + HOVER_OFFSET
        joints_hover = ik.solve_ik(x, y, z_hover, pitch_goal=0.0)
        
        # 测试抓取位置
        z_grasp = z_aruco + GRASP_OFFSET
        joints_grasp = ik.solve_ik(x, y, z_grasp, pitch_goal=0.0)
        
        if joints_hover and joints_grasp:
            print(f"  ✓ 悬停位置 (z={z_hover:.3f}): 可达")
            print(f"    关节: [{math.degrees(joints_hover[0]):.1f}°, {math.degrees(joints_hover[1]):.1f}°, "
                  f"{math.degrees(joints_hover[2]):.1f}°, {math.degrees(joints_hover[3]):.1f}°]")
            print(f"  ✓ 抓取位置 (z={z_grasp:.3f}): 可达")
            print(f"    关节: [{math.degrees(joints_grasp[0]):.1f}°, {math.degrees(joints_grasp[1]):.1f}°, "
                  f"{math.degrees(joints_grasp[2]):.1f}°, {math.degrees(joints_grasp[3]):.1f}°]")
            
            # 检查是否为垂直下降
            if abs(joints_hover[0] - joints_grasp[0]) < 0.01:
                print(f"  ✓ 垂直下降: 腰部角度保持不变")
            else:
                print(f"  ⚠ 非垂直下降: 腰部角度变化 {math.degrees(abs(joints_hover[0] - joints_grasp[0])):.2f}°")
        else:
            if not joints_hover:
                print(f"  ✗ 悬停位置 (z={z_hover:.3f}): 超出工作空间")
            if not joints_grasp:
                print(f"  ✗ 抓取位置 (z={z_grasp:.3f}): 超出工作空间")


def test_workspace_map():
    """生成工作空间地图"""
    print("\n" + "=" * 60)
    print("工作空间地图 (XZ平面, Y=0)")
    print("=" * 60)
    
    ik = PX100Kinematics()
    
    # 扫描XZ平面
    x_range = [i * 0.02 for i in range(5, 16)]  # 0.10 ~ 0.30m
    z_range = [i * 0.02 for i in range(0, 11)]   # 0.00 ~ 0.20m
    
    print("\n     ", end="")
    for z in z_range:
        print(f"{z:.2f}  ", end="")
    print()
    
    for x in x_range:
        print(f"{x:.2f} ", end="")
        for z in z_range:
            joints = ik.solve_ik(x, 0.0, z, pitch_goal=0.0)
            if joints:
                print("  ✓  ", end="")
            else:
                print("  ✗  ", end="")
        print()
    
    print("\n图例: ✓ = 可达, ✗ = 不可达")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PX100 机械臂运动学测试工具")
    print("=" * 60)
    
    test_workspace()
    test_vertical_grasp()
    test_workspace_map()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

