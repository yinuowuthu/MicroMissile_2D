"""
三回路自驾仪
============
结构：外环(加速度跟踪) → 中环(攻角稳定) → 内环(角速率阻尼) → 舵偏角

外环用前馈+PI反馈（含抗积分饱和）：
  alpha_cmd = a_cmd / K_aero  [前馈]
            + Kp * e_a + Ki * ∫e_a dt  [PI修正]

中环用P控制：q_cmd = Kp_alpha * (alpha_cmd - alpha)
内环用P控制：delta_e = Kp_q * (q_cmd - q)
"""

import numpy as np
from config import MissileParams, G, RHO


class ThreeLoopAutopilot:
    """三回路自驾仪：前馈+PI外环 → P中环 → P内环"""

    def __init__(self, mp: MissileParams = None):
        self.mp = mp or MissileParams()

        # ---- 外环：加速度→攻角 (前馈 + PI) ----
        self.Kp_acc = 0.002     # PI的P项 (rad per m/s²)
        self.Ki_acc = 0.05      # PI的I项

        # ---- 中环：攻角→角速率 (P) ----
        self.Kp_alpha = 60.0    # rad/s per rad

        # ---- 内环：角速率→舵偏角 (P) ----
        self.Kp_q = 1.0         # rad per rad/s

        # ---- 滚转通道 ----
        self.Kp_p = 0.05

        # ---- 积分器状态 ----
        self._int_el = 0.0
        self._int_az = 0.0

        # ---- 指令滤波（一阶滞后） ----
        self._alpha_cmd_filt_el = 0.0
        self._alpha_cmd_filt_az = 0.0
        self.cmd_tau = 0.05     # 指令滤波时间常数 s

        # ---- 限幅 ----
        self.alpha_max = np.radians(15)
        self.rate_max = np.radians(300)
        self.int_max_acc = 20.0   # 积分器饱和限

    def reset(self):
        """重置内部状态"""
        self._int_el = 0.0
        self._int_az = 0.0
        self._alpha_cmd_filt_el = 0.0
        self._alpha_cmd_filt_az = 0.0

    def _compute_channel(self, a_cmd, angle, rate, K_aero, dt, is_pitch=True):
        """单通道计算（俯仰或偏航共用逻辑）"""
        dm = self.mp.delta_max

        # 前馈
        alpha_ff = a_cmd / max(K_aero, 1.0)

        # 当前加速度估计
        a_actual = K_aero * angle

        # PI反馈
        e = a_cmd - a_actual

        if is_pitch:
            int_ref = self._int_el
        else:
            int_ref = self._int_az

        alpha_fb = self.Kp_acc * e + self.Ki_acc * int_ref
        alpha_cmd_raw = alpha_ff + alpha_fb
        alpha_cmd_raw = np.clip(alpha_cmd_raw, -self.alpha_max, self.alpha_max)

        # 一阶滤波平滑指令
        k = dt / (self.cmd_tau + dt)
        if is_pitch:
            self._alpha_cmd_filt_el += k * (alpha_cmd_raw - self._alpha_cmd_filt_el)
            alpha_cmd = self._alpha_cmd_filt_el
        else:
            self._alpha_cmd_filt_az += k * (alpha_cmd_raw - self._alpha_cmd_filt_az)
            alpha_cmd = self._alpha_cmd_filt_az

        # 中环
        q_cmd = self.Kp_alpha * (alpha_cmd - angle)
        q_cmd = np.clip(q_cmd, -self.rate_max, self.rate_max)

        # 内环
        delta = self.Kp_q * (q_cmd - rate)

        # 限幅
        delta_clipped = np.clip(delta, -dm, dm)

        # 抗积分饱和：舵面饱和时，只允许减小积分值方向的积分
        saturated = abs(delta) > dm
        if is_pitch:
            if not saturated or (e * self._int_el < 0):
                self._int_el += e * dt
                self._int_el = np.clip(self._int_el, -self.int_max_acc, self.int_max_acc)
        else:
            if not saturated or (e * self._int_az < 0):
                self._int_az += e * dt
                self._int_az = np.clip(self._int_az, -self.int_max_acc, self.int_max_acc)

        return delta_clipped

    def compute(self, a_el_cmd: float, a_az_cmd: float,
                alpha: float, beta: float,
                p: float, q: float, r: float,
                V: float, mass: float,
                dt: float) -> tuple:
        """
        计算舵偏角指令

        参数:
            a_el_cmd: 俯仰加速度指令 m/s²
            a_az_cmd: 偏航加速度指令 m/s²
            alpha, beta: 攻角/侧滑角 rad
            p, q, r: 角速率 rad/s
            V: 速度 m/s
            mass: 质量 kg
            dt: 时间步长 s

        返回:
            (delta_e, delta_r, delta_a) 舵偏角 rad
        """
        mp = self.mp
        qbar = 0.5 * RHO * V * V

        if qbar < 1.0 or V < 1.0:
            return 0.0, 0.0, 0.0

        # 气动增益
        K_aero = qbar * mp.S_ref * mp.CNa / mass

        # 俯仰通道
        delta_e = self._compute_channel(a_el_cmd, alpha, q, K_aero, dt, is_pitch=True)

        # 偏航通道：beta符号与pitch相反（鼻右偏时beta<0），取反对齐
        delta_r = self._compute_channel(a_az_cmd, -beta, r, K_aero, dt, is_pitch=False)

        # 滚转通道
        delta_a = -self.Kp_p * p
        delta_a = np.clip(delta_a, -mp.delta_max, mp.delta_max)

        return delta_e, delta_r, delta_a
