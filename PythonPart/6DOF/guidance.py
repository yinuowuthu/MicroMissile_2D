"""
比例导引律 (PNG)
================
a_cmd = N * V_c * dλ/dt

含接近速度估计和基于弹体能力的指令限幅。
"""

import numpy as np
from config import G, RHO, MissileParams


class ProNavGuidance:
    """比例导引律"""

    def __init__(self, N: float = 4.0, mp: MissileParams = None):
        self.N = N
        self.mp = mp or MissileParams()

        self._r_prev = None
        self._Vc_filt = 0.0

    def reset(self):
        self._r_prev = None
        self._Vc_filt = 0.0

    def _max_accel(self, V: float, mass: float) -> float:
        """当前飞行条件下的最大可用加速度"""
        mp = self.mp
        qbar = 0.5 * RHO * V * V
        K_aero = qbar * mp.S_ref * mp.CNa / max(mass, 0.1)
        alpha_max = np.radians(12)  # 实际可用攻角（留余量）
        return K_aero * alpha_max

    def compute(self, seeker_data: dict, V_missile: float,
                dt: float, mass: float = None, t: float = None) -> tuple:
        """
        计算制导加速度指令

        返回:
            (a_el_cmd, a_az_cmd) m/s²
        """
        if not seeker_data['locked']:
            return 0.0, 0.0

        # 低速时不制导（boost阶段气动力不足）
        if V_missile < 50.0:
            self._r_prev = seeker_data['r_est']
            return 0.0, 0.0

        r = seeker_data['r_est']
        los_rate_el = seeker_data['los_rate_el']
        los_rate_az = seeker_data['los_rate_az']

        # 接近速度估计
        if self._r_prev is not None and dt > 1e-6:
            Vc_raw = -(r - self._r_prev) / dt
            k = min(dt / (0.05 + dt), 0.5)
            self._Vc_filt += k * (Vc_raw - self._Vc_filt)
        self._r_prev = r

        Vc = max(self._Vc_filt, V_missile * 0.3)

        # PNG指令
        a_el = self.N * Vc * los_rate_el
        a_az = self.N * Vc * los_rate_az

        # 限幅到弹体实际能力的80%（留余量给耦合通道）
        if mass is None:
            mass = self.mp.m0
        a_lim = self._max_accel(V_missile, mass) * 0.8
        a_lim = max(a_lim, 1.0)  # 最小1 m/s²

        # 总加速度矢量限幅（不是分通道限幅）
        a_total = np.sqrt(a_el**2 + a_az**2)
        if a_total > a_lim:
            scale = a_lim / a_total
            a_el *= scale
            a_az *= scale

        return a_el, a_az
