"""
Gymnasium环境包装器 - 微型导弹制导
==========================================
将MissileEngagement2D包装成标准Gymnasium环境，用于RL训练

参考：He et al. (2021) - Computational Missile Guidance: A Deep RL Approach
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from missile_env import (
    MissileEngagement2D, MissileParams, TargetParams, SimConfig
)


class MissileGymEnv(gym.Env):
    """
    Gymnasium环境：2D导弹制导

    观测空间（5维）：
        [0] lam_dot_norm   : 视线角速率（归一化）
        [1] look_angle_norm: 视线偏差角（归一化）
        [2] am_norm        : 当前加速度（归一化）
        [3] r_dot_norm     : 距离变化率（归一化）
        [4] r_norm         : 弹目距离（归一化）

    动作空间（1维）：
        归一化加速度指令 [-1, 1] → [-a_max, a_max]

    奖励函数（He et al. 2021）：
        reward = r_energy + r_zem + r_approach + r_terminal
    """

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

        # 创建仿真环境
        mp = MissileParams(**missile_params) if missile_params else MissileParams()
        tp = TargetParams(**target_params) if target_params else TargetParams()
        cfg = SimConfig(**sim_config) if sim_config else SimConfig()

        self.env = MissileEngagement2D(missile=mp, target=tp, config=cfg)
        self.render_mode = render_mode

        # 奖励函数权重（He论文的启发式设计）
        default_weights = {
            'k_energy': 0.01,    # 能量惩罚系数
            'k_zem': 0.1,        # ZEM惩罚系数
            'k_approach': 0.5,   # 逼近奖励系数
            'terminal_hit': 200.0,   # 命中奖励
            'terminal_miss': -100.0, # 未命中惩罚
        }
        self.reward_weights = reward_weights or default_weights

        # 定义观测空间和动作空间
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 记录episode信息
        self._episode_reward = 0.0
        self._episode_length = 0
        self._prev_zem = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置仿真环境
        if seed is not None:
            np.random.seed(seed)

        self.env.reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._prev_zem = self.env.compute_zem()

        obs = self.env.get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        # 动作映射：[-1, 1] → [-a_max, a_max]
        ac = float(action[0]) * self.env.mp.a_max

        # 推进仿真
        self.env.step_guidance(ac)
        self._episode_length += 1

        # 获取观测
        obs = self.env.get_obs()

        # 计算奖励（He论文的四项奖励）
        reward = self._compute_reward()
        self._episode_reward += reward

        # 终止条件
        terminated = self.env.state.done
        truncated = False  # 我们在仿真环境内部处理超时

        # 信息字典
        info = self._get_info()
        if terminated:
            info['episode'] = {
                'r': self._episode_reward,
                'l': self._episode_length,
                'hit': self.env.state.hit,
                'reason': self.env.state.reason,
                'miss_distance': self.env.state.r if not self.env.state.hit else 0.0,
            }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """
        计算奖励（He et al. 2021的四项奖励设计）

        1. r_energy: 控制能量惩罚 -k1 * (a/amax)^2
        2. r_zem: ZEM惩罚 -k2 * ZEM
        3. r_approach: 逼近奖励 k3 * (-r_dot) 当r_dot<0时为正
        4. r_terminal: 终端奖励 +200 if hit else -100
        """
        s = self.env.state
        w = self.reward_weights

        # 1. 能量惩罚
        a_norm = s.am / self.env.mp.a_max
        r_energy = -w['k_energy'] * (a_norm ** 2)

        # 2. ZEM惩罚
        zem = self.env.compute_zem()
        r_zem = -w['k_zem'] * zem

        # 3. 逼近奖励（距离在缩短时为正）
        r_approach = w['k_approach'] * (-s.r_dot / self.env.mp.V)

        # 4. 终端奖励
        r_terminal = 0.0
        if s.done:
            if s.hit:
                r_terminal = w['terminal_hit']
                # 额外奖励：飞行时间短
                time_bonus = max(0, 10.0 - s.t)
                r_terminal += time_bonus
            else:
                r_terminal = w['terminal_miss']
                # 额外惩罚：脱靶量大
                r_terminal -= min(50.0, zem / 10.0)

        return r_energy + r_zem + r_approach + r_terminal

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        s = self.env.state
        return {
            'r': s.r,
            'zem': self.env.compute_zem(),
            'look_angle': s.look_angle,
            'am': s.am,
            't': s.t,
        }

    def render(self):
        """渲染（暂不实现）"""
        if self.render_mode == "human":
            pass  # 可以后续添加matplotlib实时绘图

    def close(self):
        """关闭环境"""
        pass


def make_env(rank: int, seed: int = 0, **env_kwargs):
    """
    创建环境的工厂函数（用于多进程向量化）

    Args:
        rank: 进程编号
        seed: 随机种子
        **env_kwargs: 传递给MissileGymEnv的参数
    """
    def _init():
        env = MissileGymEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init
