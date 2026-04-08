"""
全局物理参数配置
================
微型防空导弹6DOF仿真平台参数定义

坐标系约定：NED（North-East-Down）
  X: 前（北）
  Y: 右（东）
  Z: 下
"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================
#  物理常数
# ============================================================
G = 9.81          # 重力加速度 m/s²
RHO = 1.225       # 海平面空气密度 kg/m³


# ============================================================
#  导弹参数
# ============================================================
@dataclass
class MissileParams:
    """导弹物理参数（弹径40mm, 全长400mm）"""
    # 几何
    diameter: float = 0.04          # 弹径 m
    length: float = 0.4             # 全长 m
    S_ref: float = field(init=False)  # 参考面积 m²
    d_ref: float = field(init=False)  # 参考长度 m（= 弹径）

    # 质量
    m0: float = 0.5                 # 初始质量 kg
    m_propellant: float = 0.051     # 主脉冲推进剂质量 kg（Isp=150s估算）

    # 转动惯量（均匀圆柱近似，随质量变化在dynamics中更新）
    Ix0: float = 1.0e-4             # 滚转惯量 kg·m²
    Iy0: float = 7.0e-3             # 俯仰惯量 kg·m²
    Iz0: float = 7.0e-3             # 偏航惯量 kg·m²

    # 气动系数（经验占位，等风洞数据替换）
    CD0: float = 0.35               # 零升阻力系数
    CNa: float = 15.0               # 法向力系数斜率 /rad（40mm带翼弹体典型值8-15）
    # 压心位置（从弹头量起，占全长比例）
    x_cp_ratio: float = 0.6         # 压心在60%弹长处
    # 质心位置
    x_cg_ratio: float = 0.5         # 质心在50%弹长处（初始，燃烧后前移）
    # 静稳定裕度 = (x_cp - x_cg) / d > 0 为静稳定
    # 初始: (0.6-0.5)*0.4 / 0.04 = 1.0 cal，合理

    # 舵面效率
    Cmd: float = 8.0                # 舵面力矩系数 /rad（Cm_delta，尾翼控制典型值5-10）
    delta_max: float = np.radians(20)  # 最大舵偏角 rad
    delta_rate_max: float = np.radians(300)  # 最大舵偏角速率 rad/s

    # 阻尼力矩系数（无量纲化：Cmq * d / (2V)）
    Cmq_coeff: float = -8.0         # 俯仰阻尼系数

    def __post_init__(self):
        self.S_ref = np.pi * (self.diameter / 2) ** 2
        self.d_ref = self.diameter


# ============================================================
#  推力参数
# ============================================================
@dataclass
class PropulsionParams:
    """双脉冲固体火箭参数"""
    # 主脉冲
    thrust_main: float = 150.0      # 主脉冲推力 N
    t_burn_main: float = 0.5        # 主脉冲燃烧时间 s
    # 第二脉冲（回收用，暂占位）
    thrust_recovery: float = 50.0   # 回收脉冲推力 N
    t_burn_recovery: float = 0.3    # 回收脉冲燃烧时间 s
    t_recovery_start: float = 10.0  # 回收脉冲启动时间（暂设大值，不影响拦截仿真）

    def get_thrust(self, t: float) -> float:
        """获取t时刻推力（沿体轴X正方向）"""
        if t < self.t_burn_main:
            return self.thrust_main
        elif t >= self.t_recovery_start and t < self.t_recovery_start + self.t_burn_recovery:
            return self.thrust_recovery
        return 0.0

    def get_mass_flow(self, t: float, Isp: float = 150.0) -> float:
        """获取t时刻质量流率 kg/s"""
        T = self.get_thrust(t)
        if T > 0:
            return T / (Isp * G)
        return 0.0


# ============================================================
#  目标参数
# ============================================================
@dataclass
class TargetParams:
    """目标参数（FPV穿越机）"""
    V0: float = 30.0                # 初始速度 m/s
    a_max: float = 10 * G           # 最大机动过载 10g
    maneuver_type: str = "random"   # "none", "step", "sine", "spiral", "random"
    has_gravity: bool = True        # 目标受重力影响


# ============================================================
#  仿真配置
# ============================================================
@dataclass
class SimConfig:
    """仿真配置"""
    dt: float = 0.0005              # 仿真步长 s（2kHz，6DOF需要小步长）
    t_max: float = 5.0              # 最大仿真时间 s
    r_hit: float = 0.05             # 命中判定距离 m（厘米级动能撞击）
    r_init_min: float = 30.0        # 初始距离范围 m
    r_init_max: float = 300.0
    alt_max: float = 500.0          # 最大拦截高度 m
    decision_dt: float = 0.01       # 制导决策周期 s（100Hz）
