"""
3D 质点弹目交战仿真环境
========================
从2D (missile_env.py) 扩展到3D空间。
质点模型，匀速飞行，三通道独立一阶惯性自驾仪（滚转假设完美稳定）。

坐标系：惯性系 OXYZ，X前 Y右 Z上（NED可后续切换）
弹目相对运动用球坐标描述（距离r, 方位角lam_az, 高低角lam_el）

制导律接口：输入两通道加速度指令 (a_el_cmd, a_az_cmd)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MissileParams:
    """导弹参数（微型防空弹）"""
    V: float = 300.0            # 弹速 m/s
    a_max: float = 20 * 9.81   # 最大可用过载 20g
    tau: float = 0.2            # 自驾仪时间常数 s（俯仰/偏航通道相同）
    fov: float = np.radians(60) # 导引头半锥角 60°


@dataclass
class TargetParams:
    """目标参数（小型无人机）"""
    V: float = 50.0             # 目标速度 m/s
    a_max: float = 5 * 9.81    # 最大机动过载 5g
    maneuver_type: str = "random"  # "none", "step", "sine", "spiral", "random"


@dataclass
class SimConfig:
    """仿真配置"""
    dt: float = 0.001           # 仿真步长 s (1kHz)
    t_max: float = 20.0         # 最大仿真时间 s
    r_hit: float = 0.5          # 命中判定距离 m
    r_init_min: float = 1500.0  # 初始距离范围 m
    r_init_max: float = 3000.0
    decision_dt: float = 0.02   # 制导决策周期 s (50Hz)


class EngagementState3D:
    """3D交战状态"""
    __slots__ = [
        'xm', 'ym', 'zm', 'gamma_m', 'chi_m', 'am_el', 'am_az',
        'xt', 'yt', 'zt', 'gamma_t', 'chi_t', 'at_el', 'at_az',
        'r', 'lam_el', 'lam_az', 'r_dot', 'lam_el_dot', 'lam_az_dot',
        'look_el', 'look_az', 'look_total',
        't', 'done', 'hit', 'reason', 'r_min',
    ]

    def __init__(self):
        # 导弹（惯性系）
        self.xm = self.ym = self.zm = 0.0
        self.gamma_m = 0.0      # 弹道倾角（俯仰）
        self.chi_m = 0.0        # 弹道偏航角
        self.am_el = 0.0        # 俯仰通道实际加速度
        self.am_az = 0.0        # 偏航通道实际加速度

        # 目标（惯性系）
        self.xt = self.yt = self.zt = 0.0
        self.gamma_t = 0.0
        self.chi_t = 0.0
        self.at_el = 0.0
        self.at_az = 0.0

        # 相对量（球坐标）
        self.r = 0.0
        self.lam_el = 0.0       # 高低视线角
        self.lam_az = 0.0       # 方位视线角
        self.r_dot = 0.0
        self.lam_el_dot = 0.0
        self.lam_az_dot = 0.0

        # 导引头视线偏差
        self.look_el = 0.0
        self.look_az = 0.0
        self.look_total = 0.0

        self.t = 0.0
        self.done = False
        self.hit = False
        self.reason = ""
        self.r_min = float('inf')


def _wrap_angle(a: float) -> float:
    """归一化角度到 [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _vel_components(V: float, gamma: float, chi: float) -> Tuple[float, float, float]:
    """速度矢量分量：(Vx, Vy, Vz) from (V, gamma, chi)"""
    cg = np.cos(gamma)
    return V * cg * np.cos(chi), V * cg * np.sin(chi), V * np.sin(gamma)


class MissileEngagement3D:
    """
    3D质点弹目交战仿真器

    动力学：
    - 弹目均为质点，匀速飞行
    - 自驾仪：俯仰/偏航通道独立一阶惯性，滚转假设完美稳定
    - 目标：多种3D机动模式
    """

    def __init__(self, missile: MissileParams = None,
                 target: TargetParams = None,
                 config: SimConfig = None):
        self.mp = missile or MissileParams()
        self.tp = target or TargetParams()
        self.cfg = config or SimConfig()
        self.state = EngagementState3D()
        self._step_counter = 0
        self._decision_steps = max(1, int(self.cfg.decision_dt / self.cfg.dt))

        # 目标机动参数（每次reset随机）
        self._maneuver_amp = 0.0
        self._maneuver_freq = 0.0
        self._maneuver_start = 0.0
        self._maneuver_dir_el = 1.0
        self._maneuver_dir_az = 1.0

    def reset(self, seed: int = None) -> EngagementState3D:
        """初始化3D交战场景"""
        if seed is not None:
            np.random.seed(seed)

        s = self.state = EngagementState3D()

        # 初始弹目距离
        r0 = np.random.uniform(self.cfg.r_init_min, self.cfg.r_init_max)

        # 初始视线角（方位 ±20°，高低 ±15°）
        lam_az0 = np.random.uniform(np.radians(-20), np.radians(20))
        lam_el0 = np.random.uniform(np.radians(-15), np.radians(15))

        # 导弹从原点出发
        s.xm = s.ym = s.zm = 0.0

        # 目标位置（球坐标 → 笛卡尔）
        r_xy = r0 * np.cos(lam_el0)
        s.xt = r_xy * np.cos(lam_az0)
        s.yt = r_xy * np.sin(lam_az0)
        s.zt = r0 * np.sin(lam_el0)

        # 导弹初始航迹角：大致指向目标 ± 小扰动
        s.gamma_m = lam_el0 + np.random.uniform(np.radians(-5), np.radians(5))
        s.chi_m = lam_az0 + np.random.uniform(np.radians(-5), np.radians(5))

        # 目标航迹角：大致背向导弹 ± 扰动
        s.gamma_t = -lam_el0 + np.random.uniform(np.radians(-15), np.radians(15))
        s.chi_t = np.pi + lam_az0 + np.random.uniform(np.radians(-30), np.radians(30))

        # 自驾仪初始状态
        s.am_el = s.am_az = 0.0
        s.at_el = s.at_az = 0.0

        # 计算初始相对量
        self._update_relative_state()

        # 目标机动参数
        self._maneuver_amp = self.tp.a_max
        self._maneuver_freq = np.random.uniform(0.5, 2.0)
        self._maneuver_start = np.random.uniform(1.0, 3.0)
        self._maneuver_dir_el = np.random.choice([-1.0, 1.0])
        self._maneuver_dir_az = np.random.choice([-1.0, 1.0])

        s.t = 0.0
        s.done = False
        s.hit = False
        s.r_min = float('inf')
        self._step_counter = 0

        return s

    def _update_relative_state(self):
        """计算3D弹目相对运动量（球坐标）"""
        s = self.state
        dx = s.xt - s.xm
        dy = s.yt - s.ym
        dz = s.zt - s.zm

        r_sq = dx*dx + dy*dy + dz*dz
        s.r = np.sqrt(r_sq) + 1e-10
        r_xy = np.sqrt(dx*dx + dy*dy) + 1e-10

        s.lam_az = np.arctan2(dy, dx)
        s.lam_el = np.arctan2(dz, r_xy)

        # 速度分量
        vxm, vym, vzm = _vel_components(self.mp.V, s.gamma_m, s.chi_m)
        vxt, vyt, vzt = _vel_components(self.tp.V, s.gamma_t, s.chi_t)
        dvx, dvy, dvz = vxt - vxm, vyt - vym, vzt - vzm

        # 距离变化率
        s.r_dot = (dx * dvx + dy * dvy + dz * dvz) / s.r

        # 方位视线角速率: d(lam_az)/dt = (dy_dot*dx - dx_dot*dy) / r_xy^2
        s.lam_az_dot = (dvy * dx - dvx * dy) / (r_xy * r_xy)

        # 高低视线角速率: d(lam_el)/dt
        r_xy_dot = (dx * dvx + dy * dvy) / r_xy
        cos_el = np.cos(s.lam_el)
        cos_el = max(abs(cos_el), 1e-6) * np.sign(cos_el) if cos_el != 0 else 1e-6
        s.lam_el_dot = (dvz * s.r - dz * s.r_dot) / (s.r * s.r * cos_el)

        # 导引头视线偏差（弹体系下）
        s.look_el = _wrap_angle(s.lam_el - s.gamma_m)
        s.look_az = _wrap_angle(s.lam_az - s.chi_m)
        s.look_total = np.sqrt(s.look_el**2 + s.look_az**2)

    def _get_target_accel(self, t: float) -> Tuple[float, float]:
        """
        计算目标3D机动加速度 (a_el, a_az)
        返回俯仰和偏航两通道加速度
        """
        tp = self.tp
        if tp.maneuver_type == "none":
            return 0.0, 0.0

        if t < self._maneuver_start:
            return 0.0, 0.0

        dt_man = t - self._maneuver_start
        amp = self._maneuver_amp
        freq = self._maneuver_freq

        if tp.maneuver_type == "step":
            return (self._maneuver_dir_el * amp,
                    self._maneuver_dir_az * amp * 0.5)

        elif tp.maneuver_type == "sine":
            phase = 2 * np.pi * freq * dt_man
            return amp * np.sin(phase), amp * 0.5 * np.cos(phase)

        elif tp.maneuver_type == "spiral":
            # 螺旋机动：俯仰/偏航正交正弦
            phase = 2 * np.pi * freq * dt_man
            return amp * np.sin(phase), amp * np.cos(phase)

        elif tp.maneuver_type == "random":
            phase = 2 * np.pi * freq * dt_man
            if int(dt_man * 1000) % 3 == 0:
                # bang-bang
                a_el = self._maneuver_dir_el * amp * np.sign(np.sin(phase))
                a_az = self._maneuver_dir_az * amp * 0.5 * np.sign(np.cos(phase))
            elif int(dt_man * 1000) % 3 == 1:
                # 正弦
                a_el = amp * np.sin(phase)
                a_az = amp * 0.5 * np.cos(phase)
            else:
                # 螺旋
                a_el = amp * np.sin(phase)
                a_az = amp * np.cos(phase)
            return a_el, a_az

        return 0.0, 0.0

    def step_sim(self, a_el_cmd: float, a_az_cmd: float) -> EngagementState3D:
        """
        推进一个仿真步长

        参数:
            a_el_cmd: 俯仰通道加速度指令 (m/s²)
            a_az_cmd: 偏航通道加速度指令 (m/s²)
        """
        s = self.state
        dt = self.cfg.dt
        mp = self.mp
        tp = self.tp

        # 限幅
        a_el_cmd = np.clip(a_el_cmd, -mp.a_max, mp.a_max)
        a_az_cmd = np.clip(a_az_cmd, -mp.a_max, mp.a_max)

        # 1. 自驾仪动力学（俯仰/偏航独立一阶惯性）
        s.am_el += (a_el_cmd - s.am_el) / mp.tau * dt
        s.am_az += (a_az_cmd - s.am_az) / mp.tau * dt

        # 2. 目标机动
        at_el, at_az = self._get_target_accel(s.t)
        at_el = np.clip(at_el, -tp.a_max, tp.a_max)
        at_az = np.clip(at_az, -tp.a_max, tp.a_max)
        s.at_el = at_el
        s.at_az = at_az

        # 3. 导弹运动学积分
        s.gamma_m += s.am_el / mp.V * dt
        cos_gm = np.cos(s.gamma_m)
        if abs(cos_gm) > 1e-6:
            s.chi_m += s.am_az / (mp.V * cos_gm) * dt

        s.xm += mp.V * cos_gm * np.cos(s.chi_m) * dt
        s.ym += mp.V * cos_gm * np.sin(s.chi_m) * dt
        s.zm += mp.V * np.sin(s.gamma_m) * dt

        # 4. 目标运动学积分
        s.gamma_t += at_el / tp.V * dt
        cos_gt = np.cos(s.gamma_t)
        if abs(cos_gt) > 1e-6:
            s.chi_t += at_az / (tp.V * cos_gt) * dt

        s.xt += tp.V * cos_gt * np.cos(s.chi_t) * dt
        s.yt += tp.V * cos_gt * np.sin(s.chi_t) * dt
        s.zt += tp.V * np.sin(s.gamma_t) * dt

        # 5. 更新相对量
        self._update_relative_state()

        # 6. 更新最小距离
        if s.r < s.r_min:
            s.r_min = s.r

        # 7. 时间推进
        s.t += dt
        self._step_counter += 1

        # 8. 终止判断
        if s.r <= self.cfg.r_hit:
            s.done = True
            s.hit = True
            s.reason = "HIT"
        elif s.t >= self.cfg.t_max:
            s.done = True
            s.reason = "TIMEOUT"
        elif s.look_total > mp.fov and s.r > 100:
            # 近距离（<100m）不判FOV丢失：末端LOS角速率大是正常的
            s.done = True
            s.reason = "FOV_LOST"
        elif s.r_dot > 0 and s.r > 50:
            s.done = True
            s.reason = "MISSED"

        return s

    def step_guidance(self, a_el_cmd: float, a_az_cmd: float) -> EngagementState3D:
        """推进一个制导决策周期（内含多个仿真步长）"""
        for _ in range(self._decision_steps):
            self.step_sim(a_el_cmd, a_az_cmd)
            if self.state.done:
                break
        return self.state

    def get_obs(self, noise_std: float = 0.0) -> np.ndarray:
        """
        观测向量（10维，供制导律/RL使用）

        [0] lam_el_dot   高低视线角速率（归一化）
        [1] lam_az_dot   方位视线角速率（归一化）
        [2] look_el      高低视线偏差（归一化）
        [3] look_az      方位视线偏差（归一化）
        [4] am_el        俯仰加速度（归一化）
        [5] am_az        偏航加速度（归一化）
        [6] r_dot        距离变化率（归一化）
        [7] r            弹目距离（归一化）
        """
        s = self.state
        obs = np.array([
            s.lam_el_dot / 0.5,
            s.lam_az_dot / 0.5,
            s.look_el / self.mp.fov,
            s.look_az / self.mp.fov,
            s.am_el / self.mp.a_max,
            s.am_az / self.mp.a_max,
            s.r_dot / self.mp.V,
            s.r / self.cfg.r_init_max,
        ], dtype=np.float32)

        if noise_std > 0:
            obs += np.random.normal(0, noise_std, size=obs.shape)
        return obs

    def compute_zem(self) -> float:
        """
        3D零控脱靶量 ZEM（叉积法）
        ZEM = |r_vec × v_rel| / |v_rel|
        """
        s = self.state
        rx, ry, rz = s.xt - s.xm, s.yt - s.ym, s.zt - s.zm
        vxm, vym, vzm = _vel_components(self.mp.V, s.gamma_m, s.chi_m)
        vxt, vyt, vzt = _vel_components(self.tp.V, s.gamma_t, s.chi_t)
        dvx, dvy, dvz = vxt - vxm, vyt - vym, vzt - vzm

        # 叉积 r × v_rel
        cx = ry * dvz - rz * dvy
        cy = rz * dvx - rx * dvz
        cz = rx * dvy - ry * dvx

        v_rel_mag = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz) + 1e-10
        return np.sqrt(cx*cx + cy*cy + cz*cz) / v_rel_mag


# ============================================================
#  仿真运行器
# ============================================================

def run_episode(env: MissileEngagement3D, guidance, seed: int = None) -> dict:
    """
    运行一个完整的3D交战episode

    guidance: 需要有 compute(env) -> (a_el, a_az) 方法
    """
    env.reset(seed=seed)

    traj = {
        't': [], 'xm': [], 'ym': [], 'zm': [],
        'xt': [], 'yt': [], 'zt': [],
        'r': [], 'lam_el_dot': [], 'lam_az_dot': [],
        'am_el': [], 'am_az': [], 'ac_el': [], 'ac_az': [],
        'zem': [], 'look_total': [],
    }

    while not env.state.done:
        s = env.state
        traj['t'].append(s.t)
        traj['xm'].append(s.xm)
        traj['ym'].append(s.ym)
        traj['zm'].append(s.zm)
        traj['xt'].append(s.xt)
        traj['yt'].append(s.yt)
        traj['zt'].append(s.zt)
        traj['r'].append(s.r)
        traj['lam_el_dot'].append(s.lam_el_dot)
        traj['lam_az_dot'].append(s.lam_az_dot)
        traj['am_el'].append(s.am_el)
        traj['am_az'].append(s.am_az)
        traj['zem'].append(env.compute_zem())
        traj['look_total'].append(np.degrees(s.look_total))

        a_el, a_az = guidance.compute(env)
        traj['ac_el'].append(a_el)
        traj['ac_az'].append(a_az)

        env.step_guidance(a_el, a_az)

    for k in traj:
        traj[k] = np.array(traj[k])

    s = env.state
    traj['miss_distance'] = s.r_min
    traj['hit'] = s.hit
    traj['reason'] = s.reason
    traj['flight_time'] = s.t
    traj['energy'] = np.sum(traj['am_el']**2 + traj['am_az']**2) * env.cfg.decision_dt

    return traj
