"""
3D Gymnasium环境包装器 - 微型导弹制导
==========================================
将MissileEngagement3D包装成标准Gymnasium环境，用于PPO训练

观测空间（8维）：
    [0] lam_el_dot_norm  : 高低视线角速率（归一化）
    [1] lam_az_dot_norm  : 方位视线角速率（归一化）
    [2] look_el_norm     : 高低视线偏差（归一化）
    [3] look_az_norm     : 方位视线偏差（归一化）
    [4] am_el_norm       : 俯仰加速度（归一化）
    [5] am_az_norm       : 偏航加速度（归一化）
    [6] r_dot_norm       : 距离变化率（归一化）
    [7] r_norm           : 弹目距离（归一化）

动作空间（2维）：
    [0] a_el_cmd : 俯仰加速度指令 [-1, 1] → [-a_max, a_max]
    [1] a_az_cmd : 偏航加速度指令 [-1, 1] → [-a_max, a_max]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from missile_env_3d import (
    MissileEngagement3D, MissileParams, TargetParams, SimConfig
)


class MissileGymEnv3D(gym.Env):
    """3D导弹制导Gymnasium环境"""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        missile_params: Optional[Dict] = None,
        target_params: Optional[Dict] = None,
        sim_config: Optional[Dict] = None,
        reward_weights: Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        mp = MissileParams(**missile_params) if missile_params else MissileParams()
        tp = TargetParams(**target_params) if target_params else TargetParams()
        cfg = SimConfig(**sim_config) if sim_config else SimConfig()

        self.env = MissileEngagement3D(missile=mp, target=tp, config=cfg)
        self.render_mode = render_mode

        # 奖励权重（调优版：归一化ZEM + FOV保持）
        default_weights = {
            'k_energy': 0.002,       # 轻微能量惩罚
            'k_zem': 0.5,            # ZEM惩罚（归一化后）
            'k_approach': 1.0,       # 逼近奖励（主要正向信号）
            'k_fov': 0.5,            # FOV保持奖励
            'terminal_hit': 500.0,   # 命中大奖励
            'terminal_miss': -50.0,  # 未命中惩罚（不要太大）
        }
        self.reward_weights = reward_weights or default_weights

        # 观测空间8维，动作空间2维
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.env.reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0

        obs = self.env.get_obs()
        return obs, self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 动作映射：[-1, 1] → [-a_max, a_max]
        a_el = float(action[0]) * self.env.mp.a_max
        a_az = float(action[1]) * self.env.mp.a_max

        self.env.step_guidance(a_el, a_az)
        self._episode_length += 1

        obs = self.env.get_obs()
        reward = self._compute_reward()
        self._episode_reward += reward

        terminated = self.env.state.done
        truncated = False

        info = self._get_info()
        if terminated:
            info['episode'] = {
                'r': self._episode_reward,
                'l': self._episode_length,
                'hit': self.env.state.hit,
                'reason': self.env.state.reason,
                'miss_distance': self.env.state.r_min,
            }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """
        改进奖励函数（3D）

        核心改进：
        - ZEM归一化到[0,1]，避免量级爆炸
        - 加入FOV保持奖励（FOV丢失是主要失败模式）
        - 逼近奖励加大权重
        - 每步奖励量级控制在[-2, +2]，终端奖励主导
        """
        s = self.env.state
        mp = self.env.mp
        cfg = self.env.cfg

        # 1. 能量惩罚（轻微，两通道）
        a_norm_sq = (s.am_el / mp.a_max) ** 2 + (s.am_az / mp.a_max) ** 2
        r_energy = -0.005 * a_norm_sq

        # 2. ZEM惩罚（归一化：ZEM/r，比值越小越好）
        zem = self.env.compute_zem()
        zem_ratio = min(zem / max(s.r, 1.0), 1.0)  # [0, 1]
        r_zem = -0.5 * zem_ratio

        # 3. 逼近奖励（距离在缩短时为正）
        r_approach = 1.0 * max(0, -s.r_dot / mp.V)  # [0, ~1]

        # 4. FOV保持奖励（视线偏差小→奖励）
        fov_ratio = s.look_total / mp.fov  # [0, 1+]
        if fov_ratio < 0.5:
            r_fov = 0.2  # FOV良好
        elif fov_ratio < 0.8:
            r_fov = 0.0
        else:
            r_fov = -1.0 * fov_ratio  # 接近FOV边界，强惩罚

        # 5. 终端奖励
        r_terminal = 0.0
        if s.done:
            if s.hit:
                r_terminal = 300.0
                r_terminal += max(0, 10.0 - s.t) * 5.0  # 快速命中额外奖励
            else:
                # 按脱靶量分级惩罚
                miss = s.r_min
                if miss < 5.0:
                    r_terminal = -20.0   # 差一点，轻惩罚
                elif miss < 20.0:
                    r_terminal = -80.0
                else:
                    r_terminal = -150.0  # 完全脱靶

        return r_energy + r_zem + r_approach + r_fov + r_terminal

    def _get_info(self) -> Dict[str, Any]:
        s = self.env.state
        return {
            'r': s.r,
            'zem': self.env.compute_zem(),
            'look_total': s.look_total,
            'am_el': s.am_el,
            'am_az': s.am_az,
            't': s.t,
        }

    def render(self):
        pass

    def close(self):
        pass


def make_env(rank: int, seed: int = 0, **env_kwargs):
    """环境工厂函数（用于多进程向量化）"""
    def _init():
        env = MissileGymEnv3D(**env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init
