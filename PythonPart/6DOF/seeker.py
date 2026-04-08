"""
导引头模型
==========
红外/视觉导引头，提供视线角测量和视线角速率估计。

噪声模型：
- 闪烁噪声（glint）：σ_glint = target_size / range
- 热噪声：σ_thermal = 常数
- 总噪声：σ = sqrt(σ_glint² + σ_thermal²)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SeekerParams:
    """导引头参数"""
    gimbal_max: float = np.radians(60)     # 万向节最大偏转角 rad

    # 噪声参数
    sigma_thermal: float = np.radians(0.3) # 热噪声 rad (~0.3°)
    target_size: float = 0.3               # 目标特征尺寸 m
    sigma_glint_min: float = np.radians(0.05)

    # LOS速率滤波
    lpf_tau: float = 0.10                  # 一阶低通时间常数 s


class Seeker:
    """
    导引头模型

    输入：视线向量（惯性系）+ 导弹姿态
    输出：视线角、视线角速率、锁定状态
    """

    def __init__(self, params: SeekerParams = None):
        self.sp = params or SeekerParams()

        self._los_az_prev = 0.0
        self._los_el_prev = 0.0
        self._los_rate_az = 0.0
        self._los_rate_el = 0.0
        self._t_last = -1.0
        self._first = True
        self._locked = False
        self._rng = np.random.RandomState(42)

    def reset(self, seed: int = None):
        self._los_az_prev = 0.0
        self._los_el_prev = 0.0
        self._los_rate_az = 0.0
        self._los_rate_el = 0.0
        self._t_last = -1.0
        self._first = True
        self._locked = False
        if seed is not None:
            self._rng = np.random.RandomState(seed)

    def update(self, r_los: np.ndarray, t: float,
               m_psi: float = 0.0, m_theta: float = 0.0) -> dict:
        """
        更新导引头测量

        参数:
            r_los: 视线向量（惯性系，弹→目标）[3]
            t: 当前时间 s
            m_psi: 导弹偏航角 rad
            m_theta: 导弹俯仰角 rad

        返回:
            dict with los_az, los_el, los_rate_az, los_rate_el, r_est, locked
        """
        sp = self.sp
        r = np.linalg.norm(r_los)

        if r < 0.01:
            return {
                'locked': self._locked,
                'los_az': self._los_az_prev,
                'los_el': self._los_el_prev,
                'los_rate_az': self._los_rate_az,
                'los_rate_el': self._los_rate_el,
                'r_est': r,
            }

        # 惯性系视线角
        los_az_true = np.arctan2(r_los[1], r_los[0])
        r_xy = np.sqrt(r_los[0]**2 + r_los[1]**2)
        los_el_true = np.arctan2(-r_los[2], max(r_xy, 1e-6))

        # 离轴角（相对弹体轴线）
        off_az = (los_az_true - m_psi + np.pi) % (2 * np.pi) - np.pi
        off_el = (los_el_true - m_theta + np.pi) % (2 * np.pi) - np.pi
        off_boresight = np.sqrt(off_az**2 + off_el**2)

        # FOV检查（近程时跳过，物理上不可能在几米内丢目标）
        if off_boresight > sp.gimbal_max and r > 20.0:
            self._locked = False
            return {
                'locked': False,
                'los_az': self._los_az_prev,
                'los_el': self._los_el_prev,
                'los_rate_az': 0.0,
                'los_rate_el': 0.0,
                'r_est': r,
            }

        self._locked = True

        # 噪声
        sigma_glint = max(sp.target_size / max(r, 0.1), sp.sigma_glint_min)
        sigma = np.sqrt(sigma_glint**2 + sp.sigma_thermal**2)

        los_az_meas = los_az_true + self._rng.normal(0, sigma)
        los_el_meas = los_el_true + self._rng.normal(0, sigma)

        # LOS速率估计
        dt = t - self._t_last if self._t_last >= 0 else 0.01

        if self._first or dt < 1e-6:
            self._los_az_prev = los_az_meas
            self._los_el_prev = los_el_meas
            self._t_last = t
            self._first = False
            return {
                'locked': True,
                'los_az': los_az_meas,
                'los_el': los_el_meas,
                'los_rate_az': 0.0,
                'los_rate_el': 0.0,
                'r_est': r,
            }

        # 后向差分 + 低通滤波
        raw_az = (los_az_meas - self._los_az_prev) / dt
        raw_el = (los_el_meas - self._los_el_prev) / dt

        k = dt / (sp.lpf_tau + dt)
        self._los_rate_az += k * (raw_az - self._los_rate_az)
        self._los_rate_el += k * (raw_el - self._los_rate_el)

        self._los_az_prev = los_az_meas
        self._los_el_prev = los_el_meas
        self._t_last = t

        return {
            'locked': True,
            'los_az': los_az_meas,
            'los_el': los_el_meas,
            'los_rate_az': self._los_rate_az,
            'los_rate_el': self._los_rate_el,
            'r_est': r,
        }
