"""
2D平面弹目交战仿真环境
=======================
基于He et al. (2021) 和 Hong et al. (2020) 的运动学模型
适用于：微型红外制导导弹拦截小型无人机

坐标系：惯性系 XOY，X水平，Y竖直向上
弹目相对运动用极坐标描述（距离r, 视线角λ）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class MissileParams:
    """导弹参数（微型防空弹）"""
    V: float = 300.0          # 弹速 m/s（微型弹典型值）
    a_max: float = 40 * 9.81  # 最大可用过载 40g（对齐PNG.py）
    tau: float = 0.2          # 自驾仪时间常数 s（一阶惯性）
    fov: float = np.radians(60)  # 导引头半视场角 60°（对齐PNG.py）


@dataclass
class TargetParams:
    """目标参数（小型无人机）"""
    V: float = 50.0           # 目标速度 m/s
    a_max: float = 10 * 9.81  # 最大机动过载 10g
    maneuver_type: str = "random"  # "none", "step", "sine", "random"


@dataclass
class SimConfig:
    """仿真配置"""
    dt: float = 0.001         # 仿真步长 s（1ms, 1000Hz，对齐PNG.py）
    t_max: float = 20.0       # 最大仿真时间 s（对齐PNG.py）
    r_hit: float = 0.5        # 命中判定距离 m（对齐PNG.py）
    r_init_min: float = 1500  # 初始距离范围 m
    r_init_max: float = 3000
    decision_dt: float = 0.02 # 制导律决策周期 s（50Hz）


class EngagementState:
    """交战状态"""
    def __init__(self):
        # 导弹状态（惯性系）
        self.xm = 0.0
        self.ym = 0.0
        self.gamma_m = 0.0    # 弹道倾角
        self.am = 0.0         # 当前实际法向加速度（经过自驾仪延迟后）

        # 目标状态（惯性系）
        self.xt = 0.0
        self.yt = 0.0
        self.gamma_t = 0.0    # 目标航迹角
        self.at = 0.0         # 目标当前加速度

        # 相对运动量（极坐标）
        self.r = 0.0          # 弹目距离
        self.lam = 0.0        # 视线角(LOS angle)
        self.r_dot = 0.0      # 距离变化率
        self.lam_dot = 0.0    # 视线角速率

        # 导引头视线角（弹体坐标系下的视线偏差）
        self.look_angle = 0.0

        self.t = 0.0
        self.done = False
        self.hit = False
        self.reason = ""

        # 性能指标
        self.r_min = float('inf')  # 最小距离（真实脱靶量）


class MissileEngagement2D:
    """
    2D平面弹目交战仿真器
    
    动力学模型：
    - 弹目均为质点
    - 导弹匀速飞行（气动减速后续扩展）
    - 自驾仪建模为一阶惯性延迟
    - 目标可执行多种机动模式
    """

    def __init__(self, missile: MissileParams = None,
                 target: TargetParams = None,
                 config: SimConfig = None):
        self.mp = missile or MissileParams()
        self.tp = target or TargetParams()
        self.cfg = config or SimConfig()
        self.state = EngagementState()
        self._step_counter = 0
        self._decision_steps = max(1, int(self.cfg.decision_dt / self.cfg.dt))

        # 目标机动参数（每次reset随机生成）
        self._maneuver_amp = 0.0
        self._maneuver_freq = 0.0
        self._maneuver_start = 0.0
        self._maneuver_dir = 1.0

    def reset(self, seed: int = None) -> EngagementState:
        """
        初始化交战场景
        随机生成弹目初始几何关系（对齐PNG.py的初始化逻辑）
        """
        if seed is not None:
            np.random.seed(seed)

        s = self.state = EngagementState()

        # 初始弹目距离（加小扰动）
        r0 = np.random.uniform(self.cfg.r_init_min, self.cfg.r_init_max)

        # 初始视线角（±20°，对齐PNG.py）
        lam0 = np.random.uniform(np.radians(-20), np.radians(20))

        # 导弹从原点出发
        s.xm = 0.0
        s.ym = 0.0

        # 目标位置（极坐标 -> 笛卡尔）
        s.xt = r0 * np.cos(lam0)
        s.yt = r0 * np.sin(lam0)

        # 导弹初始航迹角：视线角 ± 5°（对齐PNG.py）
        s.gamma_m = lam0 + np.random.uniform(np.radians(-5), np.radians(5))

        # 目标航迹角：大致背向导弹 ± 30°（对齐PNG.py）
        s.gamma_t = np.pi + lam0 + np.random.uniform(np.radians(-30), np.radians(30))

        # 初始化自驾仪状态
        s.am = 0.0
        s.at = 0.0

        # 计算初始相对运动量
        self._update_relative_state()

        # 随机生成目标机动参数（对齐PNG.py）
        self._maneuver_amp = self.tp.a_max  # 使用目标的最大过载
        self._maneuver_freq = np.random.uniform(0.5, 2.0)  # Hz
        self._maneuver_start = np.random.uniform(1.0, 3.0)  # 延迟1-3s开始机动
        self._maneuver_dir = np.random.choice([-1, 1])

        s.t = 0.0
        s.done = False
        s.hit = False
        s.r_min = float('inf')  # 重置最小距离
        self._step_counter = 0

        return s

    def _update_relative_state(self):
        """计算弹目相对运动量"""
        s = self.state
        dx = s.xt - s.xm
        dy = s.yt - s.ym

        s.r = np.sqrt(dx**2 + dy**2) + 1e-10  # 防除零
        s.lam = np.arctan2(dy, dx)

        # 相对速度分量（沿LOS和垂直LOS）
        Vr_m = self.mp.V * np.cos(s.gamma_m - s.lam)
        Vr_t = self.tp.V * np.cos(s.gamma_t - s.lam)
        Vl_m = self.mp.V * np.sin(s.gamma_m - s.lam)
        Vl_t = self.tp.V * np.sin(s.gamma_t - s.lam)

        s.r_dot = Vr_t - Vr_m        # 距离变化率（负=逼近）
        s.lam_dot = (Vl_t - Vl_m) / s.r   # 视线角速率

        # 导引头视线角（弹体系下的偏差角）
        s.look_angle = s.lam - s.gamma_m
        # 归一化到 [-pi, pi]
        s.look_angle = (s.look_angle + np.pi) % (2 * np.pi) - np.pi

    def _get_target_accel(self, t: float) -> float:
        """
        计算目标机动加速度
        模拟无人机的规避机动
        """
        tp = self.tp
        if tp.maneuver_type == "none":
            return 0.0
        elif tp.maneuver_type == "step":
            if t > self._maneuver_start:
                return self._maneuver_dir * self._maneuver_amp
            return 0.0
        elif tp.maneuver_type == "sine":
            if t > self._maneuver_start:
                return self._maneuver_amp * np.sin(
                    2 * np.pi * self._maneuver_freq * (t - self._maneuver_start))
            return 0.0
        elif tp.maneuver_type == "random":
            # 随机选择：50%概率bang-bang, 50%概率正弦
            if t < self._maneuver_start:
                return 0.0
            phase = (t - self._maneuver_start) * self._maneuver_freq
            if int(phase * 1000) % 2 == 0:  # 简单伪随机切换
                return self._maneuver_dir * self._maneuver_amp * np.sign(
                    np.sin(2 * np.pi * self._maneuver_freq * (t - self._maneuver_start)))
            else:
                return self._maneuver_amp * np.sin(
                    2 * np.pi * self._maneuver_freq * (t - self._maneuver_start))
        return 0.0

    def step_sim(self, ac: float) -> EngagementState:
        """
        推进一个仿真步长
        
        参数:
            ac: 制导律输出的加速度指令 (m/s²)
        返回:
            更新后的状态
        """
        s = self.state
        dt = self.cfg.dt
        mp = self.mp
        tp = self.tp

        # 限幅
        ac = np.clip(ac, -mp.a_max, mp.a_max)

        # 1. 自驾仪动力学（一阶惯性）
        #    τ·ȧm + am = ac  →  ȧm = (ac - am) / τ
        s.am += (ac - s.am) / mp.tau * dt

        # 2. 目标机动
        at_cmd = self._get_target_accel(s.t)
        at_cmd = np.clip(at_cmd, -tp.a_max, tp.a_max)
        s.at = at_cmd  # 目标假设无延迟（无人机响应快）

        # 3. 运动学积分（欧拉法）
        # 导弹
        s.gamma_m += s.am / mp.V * dt
        s.xm += mp.V * np.cos(s.gamma_m) * dt
        s.ym += mp.V * np.sin(s.gamma_m) * dt

        # 目标
        s.gamma_t += s.at / tp.V * dt
        s.xt += tp.V * np.cos(s.gamma_t) * dt
        s.yt += tp.V * np.sin(s.gamma_t) * dt

        # 4. 更新相对量
        self._update_relative_state()

        # 5. 更新最小距离
        if s.r < s.r_min:
            s.r_min = s.r

        # 6. 时间推进
        s.t += dt
        self._step_counter += 1

        # 7. 终止判断
        if s.r <= self.cfg.r_hit:
            s.done = True
            s.hit = True
            s.reason = "HIT"
        elif s.t >= self.cfg.t_max:
            s.done = True
            s.reason = "TIMEOUT"
        elif abs(s.look_angle) > mp.fov:
            s.done = True
            s.reason = "FOV_LOST"
        elif s.r_dot > 0 and s.r > 50:
            # 弹目距离在增大且已经较远 → 已经飞过头了
            s.done = True
            s.reason = "MISSED"

        return s

    def step_guidance(self, ac: float) -> EngagementState:
        """
        推进一个制导决策周期（内部包含多个仿真步长）
        这是制导律调用的主接口
        """
        for _ in range(self._decision_steps):
            self.step_sim(ac)
            if self.state.done:
                break
        return self.state

    def get_obs(self, noise_std: float = 0.0) -> np.ndarray:
        """
        获取观测向量（供制导律/RL智能体使用）
        
        返回归一化的观测向量：
        [0] lam_dot_norm  : 视线角速率（归一化）   ← 红外导引头可测
        [1] look_angle_norm: 视线偏差角（归一化）   ← 红外导引头可测
        [2] am_norm       : 当前加速度（归一化）    ← 加速度计可测
        [3] r_dot_norm    : 距离变化率（归一化）    ← 可选，纯红外无此项
        [4] r_norm        : 弹目距离（归一化）      ← 可选，纯红外无此项
        """
        s = self.state
        obs = np.array([
            s.lam_dot / 0.5,                     # 视线角速率，99%值在±0.5内
            s.look_angle / self.mp.fov,          # 视线偏差/最大视场角
            s.am / self.mp.a_max,                # 当前加速度/最大加速度
            s.r_dot / self.mp.V,                 # 接近速度/弹速
            s.r / self.cfg.r_init_max,           # 距离/初始最大距离
        ], dtype=np.float32)

        if noise_std > 0:
            obs += np.random.normal(0, noise_std, size=obs.shape)

        return obs

    def get_obs_ir_only(self, noise_std: float = 0.0) -> np.ndarray:
        """
        纯红外导引头观测（无距离信息）
        仅3维：[视线角速率, 视线偏差角, 当前加速度]
        """
        s = self.state
        obs = np.array([
            s.lam_dot / 0.5,
            s.look_angle / self.mp.fov,
            s.am / self.mp.a_max,
        ], dtype=np.float32)

        if noise_std > 0:
            obs += np.random.normal(0, noise_std, size=obs.shape)

        return obs

    def compute_zem(self) -> float:
        """
        计算零控脱靶量 ZEM (Zero Effort Miss)
        ZEM = r * Vλ / sqrt(Vr² + Vλ²)
        """
        s = self.state
        Vl = s.r * s.lam_dot
        Vr = s.r_dot
        denom = np.sqrt(Vr**2 + Vl**2) + 1e-10
        return abs(s.r * Vl / denom)


# ============================================================
#  制导律接口
# ============================================================

class GuidanceLaw:
    """制导律基类"""
    def compute(self, env: MissileEngagement2D) -> float:
        raise NotImplementedError


class ProportionalNavigation(GuidanceLaw):
    """
    比例导引律 (PNG)
    ac = N * Vm * λ̇
    
    N: 导航比（通常取3~5，N=3为能量最优）
    """
    def __init__(self, N: float = 4.0):
        self.N = N

    def compute(self, env: MissileEngagement2D) -> float:
        s = env.state
        # 经典PNG：ac = N * V_missile * LOS_rate
        ac = self.N * env.mp.V * s.lam_dot
        return ac


class AugmentedPN(GuidanceLaw):
    """
    增广比例导引 (APN)
    ac = N * Vm * λ̇ + 0.5 * N * aT_normal
    需要估计目标法向加速度（实际中难以获取）
    """
    def __init__(self, N: float = 4.0):
        self.N = N

    def compute(self, env: MissileEngagement2D) -> float:
        s = env.state
        # 目标法向加速度在LOS法线方向的分量
        at_normal = s.at * np.cos(s.lam - s.gamma_t)
        ac = self.N * env.mp.V * s.lam_dot + 0.5 * self.N * at_normal
        return ac


# ============================================================
#  仿真运行器
# ============================================================

def run_episode(env: MissileEngagement2D,
                guidance: GuidanceLaw,
                seed: int = None,
                noise_std: float = 0.0) -> dict:
    """
    运行一个完整的交战episode
    
    返回:
        字典包含完整飞行轨迹和性能指标
    """
    env.reset(seed=seed)

    # 记录轨迹
    traj = {
        't': [], 'xm': [], 'ym': [], 'xt': [], 'yt': [],
        'r': [], 'lam_dot': [], 'am': [], 'ac': [],
        'zem': [], 'look_angle': [],
    }

    while not env.state.done:
        s = env.state

        # 记录
        traj['t'].append(s.t)
        traj['xm'].append(s.xm)
        traj['ym'].append(s.ym)
        traj['xt'].append(s.xt)
        traj['yt'].append(s.yt)
        traj['r'].append(s.r)
        traj['lam_dot'].append(s.lam_dot)
        traj['am'].append(s.am)
        traj['zem'].append(env.compute_zem())
        traj['look_angle'].append(np.degrees(s.look_angle))

        # 计算制导指令
        ac = guidance.compute(env)
        traj['ac'].append(ac)

        # 推进一个决策周期
        env.step_guidance(ac)

    # 转为numpy数组
    for k in traj:
        traj[k] = np.array(traj[k])

    # 性能指标
    s = env.state
    traj['miss_distance'] = s.r_min  # 使用真实的最小距离
    traj['final_zem'] = traj['zem'][-1] if len(traj['zem']) > 0 else float('inf')
    traj['hit'] = s.hit
    traj['reason'] = s.reason
    traj['flight_time'] = s.t
    traj['energy'] = np.sum(traj['am']**2) * env.cfg.decision_dt  # ∫a²dt

    return traj
