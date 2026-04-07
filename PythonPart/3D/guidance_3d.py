"""
3D 制导律
=========
3D PNG (比例导引) 和 APN (增广比例导引)
两通道独立制导：俯仰通道 + 偏航通道

接口：compute(env) -> (a_el_cmd, a_az_cmd)
"""

import numpy as np
from missile_env_3d import MissileEngagement3D


class GuidanceLaw3D:
    """制导律基类"""
    def compute(self, env: MissileEngagement3D):
        raise NotImplementedError


class ProportionalNavigation3D(GuidanceLaw3D):
    """
    3D比例导引律 (PNG)

    两通道独立：
        a_el = N * V * lam_el_dot
        a_az = N * V * lam_az_dot
    """
    def __init__(self, N: float = 4.0):
        self.N = N

    def compute(self, env: MissileEngagement3D):
        s = env.state
        V = env.mp.V
        a_el = self.N * V * s.lam_el_dot
        a_az = self.N * V * s.lam_az_dot
        return a_el, a_az


class AugmentedPN3D(GuidanceLaw3D):
    """
    3D增广比例导引 (APN)

    两通道独立，补偿目标法向加速度：
        a_el = N * V * lam_el_dot + 0.5 * N * at_el_normal
        a_az = N * V * lam_az_dot + 0.5 * N * at_az_normal
    """
    def __init__(self, N: float = 4.0):
        self.N = N

    def compute(self, env: MissileEngagement3D):
        s = env.state
        V = env.mp.V
        N = self.N

        # 目标加速度在LOS法线方向的分量（简化：直接用目标加速度）
        a_el = N * V * s.lam_el_dot + 0.5 * N * s.at_el
        a_az = N * V * s.lam_az_dot + 0.5 * N * s.at_az
        return a_el, a_az
