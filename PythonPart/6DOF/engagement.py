"""
交战仿真器
==========================
组装6DOF导弹 + 目标质点 + 导引头 + 自驾仪 + 制导律。
支持PNG基线和RL制导的统一接口。
"""

import numpy as np
from typing import Optional
from config import (MissileParams, PropulsionParams, TargetParams,
                    SimConfig, G, RHO)
from dynamics import Missile6DOF, body_to_inertial
from seeker import Seeker, SeekerParams
from autopilot import ThreeLoopAutopilot
from guidance import ProNavGuidance


# ============================================================
#  目标模型（变速有重力质点）
# ============================================================

class Target:
    """
    FPV穿越机目标模型
    6维状态: [x, y, z, vx, vy, vz] 惯性系
    含重力、多种机动模式
    """

    def __init__(self, tp: TargetParams = None):
        self.tp = tp or TargetParams()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.t = 0.0

        # 机动参数（每次reset随机）
        self._man_freq = 1.0
        self._man_start = 0.5
        self._man_dir = np.array([0.0, 1.0, 0.0])

    def init_state(self, pos: np.ndarray, vel: np.ndarray):
        """初始化目标状态"""
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.t = 0.0

    def randomize_maneuver(self, rng: np.random.RandomState = None):
        """随机化机动参数"""
        rng = rng or np.random
        self._man_freq = rng.uniform(0.5, 3.0)
        self._man_start = rng.uniform(0.2, 1.0)
        rand_dir = rng.randn(3)
        speed = np.linalg.norm(self.vel)
        if speed > 0.1:
            v_hat = self.vel / speed
            rand_dir -= np.dot(rand_dir, v_hat) * v_hat
        norm = np.linalg.norm(rand_dir)
        self._man_dir = rand_dir / norm if norm > 1e-6 else np.array([0, 0, 1.0])

    def get_accel(self, t: float) -> np.ndarray:
        """计算目标机动加速度（惯性系）"""
        tp = self.tp
        a = np.zeros(3)

        if tp.maneuver_type == "none" or t < self._man_start:
            pass
        else:
            dt_man = t - self._man_start
            amp = tp.a_max
            freq = self._man_freq
            phase = 2 * np.pi * freq * dt_man

            if tp.maneuver_type == "step":
                a = self._man_dir * amp
            elif tp.maneuver_type == "sine":
                a = self._man_dir * amp * np.sin(phase)
            elif tp.maneuver_type == "spiral":
                d2 = np.cross(self._man_dir, self.vel / max(np.linalg.norm(self.vel), 0.1))
                d2_norm = np.linalg.norm(d2)
                if d2_norm > 1e-6:
                    d2 /= d2_norm
                else:
                    d2 = np.array([0, 0, 1.0])
                a = self._man_dir * amp * np.sin(phase) + d2 * amp * np.cos(phase)
            elif tp.maneuver_type == "random":
                if int(dt_man * 1000) % 2 == 0:
                    a = self._man_dir * amp * np.sign(np.sin(phase))
                else:
                    a = self._man_dir * amp * np.sin(phase)

        a_mag = np.linalg.norm(a)
        if a_mag > tp.a_max:
            a = a * (tp.a_max / a_mag)

        return a

    def step(self, dt: float):
        """推进一步（欧拉积分）"""
        a = self.get_accel(self.t)
        if self.tp.has_gravity:
            a = a + np.array([0.0, 0.0, G])
        self.vel += a * dt
        self.pos += self.vel * dt
        self.t += dt


# ============================================================
#  交战仿真器
# ============================================================

class Engagement:
    """
    弹目交战仿真器

    集成：6DOF导弹 + 目标 + 导引头 + 自驾仪 + 制导律
    支持PNG和外部制导（RL）两种模式
    """

    def __init__(self, mp: MissileParams = None, prop: PropulsionParams = None,
                 tp: TargetParams = None, cfg: SimConfig = None,
                 seeker_params: SeekerParams = None):
        self.mp = mp or MissileParams()
        self.prop = prop or PropulsionParams()
        self.tp = tp or TargetParams()
        self.cfg = cfg or SimConfig()

        self.missile = Missile6DOF(self.mp, self.prop)
        self.target = Target(self.tp)
        self.seeker = Seeker(seeker_params)
        self.autopilot = ThreeLoopAutopilot(self.mp)
        self.guidance = ProNavGuidance(N=4.0, mp=self.mp)

        self.t = 0.0
        self.done = False
        self.hit = False
        self.reason = ""
        self.r_min = float('inf')

        # 制导决策计时
        self._t_last_guidance = -1.0
        self._a_el_cmd = 0.0
        self._a_az_cmd = 0.0
        self._seeker_data = {}

    def reset(self, r0: float = None, seed: int = None):
        """初始化交战场景"""
        rng = np.random.RandomState(seed)

        if r0 is None:
            r0 = rng.uniform(self.cfg.r_init_min, self.cfg.r_init_max)

        az0 = rng.uniform(np.radians(-20), np.radians(20))
        el0 = rng.uniform(np.radians(-10), np.radians(10))

        m_pos = np.array([0.0, 0.0, 0.0])
        m_euler = np.array([0.0, el0, az0])

        self.missile.init_state(m_pos, m_euler, speed=0.0)

        r_xy = r0 * np.cos(el0)
        t_pos = np.array([
            r_xy * np.cos(az0),
            r_xy * np.sin(az0),
            -r0 * np.sin(el0),
        ])

        t_heading = az0 + np.pi / 2 + rng.uniform(-0.5, 0.5)
        t_climb = rng.uniform(-0.1, 0.1)
        t_speed = self.tp.V0
        t_vel = np.array([
            t_speed * np.cos(t_climb) * np.cos(t_heading),
            t_speed * np.cos(t_climb) * np.sin(t_heading),
            -t_speed * np.sin(t_climb),
        ])

        self.target.init_state(t_pos, t_vel)
        self.target.randomize_maneuver(rng)

        self.seeker.reset(seed=rng.randint(0, 2**31))
        self.autopilot.reset()
        self.guidance.reset()

        self.t = 0.0
        self.done = False
        self.hit = False
        self.reason = ""
        self.r_min = float('inf')
        self._t_last_guidance = -1.0
        self._a_el_cmd = 0.0
        self._a_az_cmd = 0.0
        self._seeker_data = {}

    def get_range(self) -> float:
        return np.linalg.norm(self.target.pos - self.missile.pos)

    def get_los(self) -> np.ndarray:
        return self.target.pos - self.missile.pos

    def step_guided(self, a_el_override=None, a_az_override=None):
        """
        推进一个仿真步长（含制导回路）

        参数:
            a_el_override: 外部俯仰加速度指令（RL模式）
            a_az_override: 外部偏航加速度指令（RL模式）
        """
        dt = self.cfg.dt
        m = self.missile

        # ---- 制导决策（按decision_dt周期） ----
        do_guidance = (self.t - self._t_last_guidance >= self.cfg.decision_dt - 1e-6
                       or self._t_last_guidance < 0)

        if do_guidance:
            self._t_last_guidance = self.t

            # 导引头更新（传入导弹姿态用于FOV检查）
            r_los = self.get_los()
            m_psi = m.state[8]    # 偏航角
            m_theta = m.state[7]  # 俯仰角
            self._seeker_data = self.seeker.update(r_los, self.t, m_psi, m_theta)

            if a_el_override is not None:
                # RL模式：外部提供加速度指令
                self._a_el_cmd = a_el_override
                self._a_az_cmd = a_az_override if a_az_override is not None else 0.0
            else:
                # PNG模式
                self._a_el_cmd, self._a_az_cmd = self.guidance.compute(
                    self._seeker_data, m.speed, self.cfg.decision_dt, m.mass)

        # ---- 自驾仪（每个仿真步都运行） ----
        alpha = m.alpha
        beta = m.beta
        p, q, r = m.state[9:12]
        V = m.speed

        de, dr_fin, da = self.autopilot.compute(
            self._a_el_cmd, self._a_az_cmd,
            alpha, beta, p, q, r, V, m.mass, dt)
        m.set_fins(de, dr_fin, da)

        # ---- 积分 ----
        m.step(dt)
        self.target.step(dt)
        self.t += dt

        # ---- 终止判断 ----
        r_range = self.get_range()
        if r_range < self.r_min:
            self.r_min = r_range

        if r_range <= self.cfg.r_hit:
            self.done = True
            self.hit = True
            self.reason = "HIT"
        elif self.t >= self.cfg.t_max:
            self.done = True
            self.reason = "TIMEOUT"
        elif m.pos[2] > 10.0:
            self.done = True
            self.reason = "GROUND"
        elif r_range > 1000.0:
            self.done = True
            self.reason = "OUT_OF_RANGE"

    def step(self):
        """无制导步进（向后兼容）"""
        dt = self.cfg.dt
        self.missile.step(dt)
        self.target.step(dt)
        self.t += dt

        r = self.get_range()
        if r < self.r_min:
            self.r_min = r

        if r <= self.cfg.r_hit:
            self.done = True
            self.hit = True
            self.reason = "HIT"
        elif self.t >= self.cfg.t_max:
            self.done = True
            self.reason = "TIMEOUT"
        elif self.missile.pos[2] > 10.0:
            self.done = True
            self.reason = "GROUND"
        elif r > 1000.0:
            self.done = True
            self.reason = "OUT_OF_RANGE"

    def step_n(self, n: int):
        for _ in range(n):
            self.step()
            if self.done:
                break


# ============================================================
#  轨迹记录器
# ============================================================

def run_ballistic(eng: Engagement, t_end: float = None) -> dict:
    """运行无制导弹道仿真"""
    if t_end is None:
        t_end = eng.cfg.t_max

    traj = {
        't': [], 'x': [], 'y': [], 'z': [],
        'u': [], 'v': [], 'w': [], 'V': [],
        'phi': [], 'theta': [], 'psi': [],
        'p': [], 'q': [], 'r_rate': [],
        'alpha': [], 'beta': [],
        'mass': [], 'thrust': [],
        'n_load': [],
        'xt': [], 'yt': [], 'zt': [],
        'range': [],
    }

    while eng.t < t_end and not eng.done:
        m = eng.missile
        s = m.state

        traj['t'].append(eng.t)
        traj['x'].append(s[0])
        traj['y'].append(s[1])
        traj['z'].append(s[2])
        traj['u'].append(s[3])
        traj['v'].append(s[4])
        traj['w'].append(s[5])
        traj['V'].append(m.speed)
        traj['phi'].append(np.degrees(s[6]))
        traj['theta'].append(np.degrees(s[7]))
        traj['psi'].append(np.degrees(s[8]))
        traj['p'].append(np.degrees(s[9]))
        traj['q'].append(np.degrees(s[10]))
        traj['r_rate'].append(np.degrees(s[11]))
        traj['alpha'].append(np.degrees(m.alpha))
        traj['beta'].append(np.degrees(m.beta))
        traj['mass'].append(m.mass)
        traj['thrust'].append(eng.prop.get_thrust(eng.t))
        traj['n_load'].append(m.normal_accel)

        traj['xt'].append(eng.target.pos[0])
        traj['yt'].append(eng.target.pos[1])
        traj['zt'].append(eng.target.pos[2])
        traj['range'].append(eng.get_range())

        eng.step()

    for k in traj:
        traj[k] = np.array(traj[k])

    traj['miss_distance'] = eng.r_min
    traj['hit'] = eng.hit
    traj['reason'] = eng.reason
    traj['flight_time'] = eng.t

    return traj


def run_guided(eng: Engagement, t_end: float = None) -> dict:
    """运行PNG制导仿真，记录完整轨迹"""
    if t_end is None:
        t_end = eng.cfg.t_max

    traj = {
        't': [], 'x': [], 'y': [], 'z': [], 'V': [],
        'alpha': [], 'beta': [],
        'delta_e': [], 'delta_r': [],
        'a_el_cmd': [], 'a_az_cmd': [],
        'xt': [], 'yt': [], 'zt': [],
        'range': [],
        'los_rate_el': [], 'los_rate_az': [],
        'locked': [],
    }

    while eng.t < t_end and not eng.done:
        m = eng.missile

        traj['t'].append(eng.t)
        traj['x'].append(m.pos[0])
        traj['y'].append(m.pos[1])
        traj['z'].append(m.pos[2])
        traj['V'].append(m.speed)
        traj['alpha'].append(np.degrees(m.alpha))
        traj['beta'].append(np.degrees(m.beta))
        traj['delta_e'].append(np.degrees(m.delta_e))
        traj['delta_r'].append(np.degrees(m.delta_r))
        traj['a_el_cmd'].append(eng._a_el_cmd / G)
        traj['a_az_cmd'].append(eng._a_az_cmd / G)
        traj['xt'].append(eng.target.pos[0])
        traj['yt'].append(eng.target.pos[1])
        traj['zt'].append(eng.target.pos[2])
        traj['range'].append(eng.get_range())
        sd = eng._seeker_data
        traj['los_rate_el'].append(np.degrees(sd.get('los_rate_el', 0)))
        traj['los_rate_az'].append(np.degrees(sd.get('los_rate_az', 0)))
        traj['locked'].append(sd.get('locked', False))

        eng.step_guided()

    for k in traj:
        traj[k] = np.array(traj[k])

    traj['miss_distance'] = eng.r_min
    traj['hit'] = eng.hit
    traj['reason'] = eng.reason
    traj['flight_time'] = eng.t

    return traj
