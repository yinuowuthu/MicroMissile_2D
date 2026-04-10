"""
6DOF导弹制导 Gymnasium环境 v4
================================
修复：
  - 观测加入 LOS 角度（los_el, los_az），智能体现在知道目标在哪
  - 奖励重新平衡：接近奖励 + 小miss惩罚，命中奖励主导
  - 观测维度：10
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import TargetParams, SimConfig, MissileParams
from engagement import Engagement
from seeker import SeekerParams


class MissileGym6DOF(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tp: TargetParams = None,
        cfg: SimConfig = None,
        seeker_params: SeekerParams = None,
        a_max_cmd: float = 150.0,
        hit_reward: float = 100.0,
        miss_penalty: float = 10.0,
        seed: int = None,
    ):
        super().__init__()

        self.tp = tp or TargetParams(maneuver_type='random', V0=30.0)
        self.cfg = cfg or SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5)
        self.sp = seeker_params or SeekerParams()
        self.a_max_cmd = a_max_cmd
        self.hit_reward = hit_reward
        self.miss_penalty = miss_penalty

        self.eng = Engagement(tp=self.tp, cfg=self.cfg, seeker_params=self.sp)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.RandomState(seed)
        self._r_prev = 100.0
        self._r0 = 100.0  # 初始距离，用于奖励归一化
        self._steps_per_decision = max(1, round(self.cfg.decision_dt / self.cfg.dt))

    def _get_obs(self) -> np.ndarray:
        eng = self.eng
        sd = eng._seeker_data
        m = eng.missile
        r = eng.get_range()
        r_dot = (r - self._r_prev) / self.cfg.decision_dt if self._r_prev > 0 else 0.0

        los_rate_el = sd.get('los_rate_el', 0.0)
        los_rate_az = sd.get('los_rate_az', 0.0)
        # LOS角度：智能体需要知道目标在哪个方向
        los_el = sd.get('los_el', 0.0)
        los_az = sd.get('los_az', 0.0)

        return np.array([
            np.clip(los_el / np.radians(45),    -10, 10),  # LOS仰角（归一化到±45°）
            np.clip(los_az / np.radians(45),    -10, 10),  # LOS方位角
            np.clip(los_rate_el / 1.0,          -10, 10),  # LOS仰角速率 rad/s
            np.clip(los_rate_az / 1.0,          -10, 10),  # LOS方位角速率 rad/s
            np.clip(m.alpha / np.radians(20),   -10, 10),
            np.clip(m.beta  / np.radians(20),   -10, 10),
            np.clip(r / 100.0,                    0, 10),
            np.clip(r_dot / 100.0,              -10, 10),
            np.clip(m.speed / 200.0,              0, 10),
            np.clip(eng.t / self.cfg.t_max,       0,  1),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        ep_seed = int(self._rng.randint(0, 2**31))
        self.eng.reset(r0=None, seed=ep_seed)

        r_los = self.eng.get_los()
        m = self.eng.missile
        self.eng._seeker_data = self.eng.seeker.update(
            r_los, self.eng.t, m.state[8], m.state[7])

        self._r0 = self.eng.get_range()
        self._r_prev = self._r0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        a_el = float(action[0]) * self.a_max_cmd
        a_az = float(action[1]) * self.a_max_cmd

        r_before = self._r_prev

        for _ in range(self._steps_per_decision):
            if self.eng.done:
                break
            self.eng.step_guided(a_el_override=a_el, a_az_override=a_az)

        r_after = self.eng.get_range()

        # 接近奖励：放大5倍，增强过程信号
        dr = r_before - r_after
        reward = 5.0 * dr / max(self._r0, 1.0)

        terminated = False
        if self.eng.done:
            terminated = True
            if self.eng.reason == 'HIT':
                reward += self.hit_reward
            else:
                # 平滑脱靶惩罚：r_min越小惩罚越小，鼓励"接近命中"
                r_min = self.eng.r_min
                reward -= self.miss_penalty * (1.0 - np.exp(-r_min / 3.0))

        self._r_prev = r_after

        info = {
            'r': r_after,
            'r_min': self.eng.r_min,
            'hit': self.eng.hit,
            'reason': self.eng.reason,
            't': self.eng.t,
        }
        return self._get_obs(), reward, terminated, False, info
