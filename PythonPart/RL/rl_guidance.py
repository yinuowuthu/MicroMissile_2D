"""
改进版RL制导律 — PNG + 学习修正
================================
核心思路（来自He et al. 2021）：
  不要从零学习制导指令，而是在PNG基础上学习一个修正量
  ac = N_rl(obs) * Vm * lam_dot
  其中 N_rl 是RL学到的"自适应导航比"，初始化为PNG的N=4

这样RL一开始就有PNG级别的性能，只需要学习如何做得更好。
"""

import numpy as np
import pickle
from typing import List

# 导入仿真环境
from missile_env import (MissileEngagement2D, MissileParams, TargetParams,
                         SimConfig, GuidanceLaw)


class SmallNet:
    """极简MLP，纯numpy"""
    def __init__(self, sizes: List[int]):
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(np.random.randn(sizes[i], sizes[i+1]) * scale)
            self.b.append(np.zeros(sizes[i+1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h

    def get_flat(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.W, self.b):
            parts.extend([W.ravel(), b.ravel()])
        return np.concatenate(parts)

    def set_flat(self, flat: np.ndarray):
        idx = 0
        for i in range(len(self.W)):
            s = self.W[i].size
            self.W[i] = flat[idx:idx+s].reshape(self.W[i].shape)
            idx += s
            s = self.b[i].size
            self.b[i] = flat[idx:idx+s].reshape(self.b[i].shape)
            idx += s

    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.W, self.b))


class RLGuidanceLaw(GuidanceLaw):
    """
    RL制导律：PNG + 神经网络修正
    
    ac = N_rl(obs) * Vm * lam_dot + correction(obs) * a_max * 0.1
    
    N_rl: 自适应导航比（网络输出，初始≈4）
    correction: 额外修正项（用于处理PNG无法应对的情况）
    """

    def __init__(self, obs_dim: int = 3):
        self.obs_dim = obs_dim
        # 网络输出2维：[导航比偏移量, 额外修正]
        self.net = SmallNet([obs_dim, 32, 16, 2])
        self.base_N = 4.0  # PNG基准导航比
        self._flat = self.net.get_flat()

    def compute(self, env: MissileEngagement2D) -> float:
        s = env.state
        obs = env.get_obs_ir_only()  # [lam_dot_norm, look_angle_norm, am_norm]
        out = self.net.forward(obs.reshape(1, -1))[0]

        # 输出1: 导航比修正（tanh映射到±3，即N在1~7之间）
        delta_N = np.tanh(out[0]) * 3.0
        N_adaptive = self.base_N + delta_N

        # 输出2: 额外修正量（用于非线性补偿）
        correction = np.tanh(out[1]) * 0.1 * env.mp.a_max

        # 最终指令 = 自适应PNG + 修正
        ac = N_adaptive * env.mp.V * s.lam_dot + correction
        return ac


def evaluate_guidance(guidance, env, n_episodes: int = 5,
                      seeds: List[int] = None) -> float:
    """
    评估制导律的适应度
    改进奖励设计：更稀疏，避免过度塑形
    """
    if seeds is None:
        seeds = list(range(n_episodes))

    total_fitness = 0.0

    for seed in seeds:
        env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0
        max_steps = 1500

        while not env.state.done and steps < max_steps:
            ac = guidance.compute(env)
            env.step_guidance(ac)
            steps += 1

            s = env.state

            # ===== 简化奖励设计 =====
            # 1. 轻微能量惩罚（避免过度机动）
            r_energy = -0.001 * (s.am / env.mp.a_max) ** 2

            # 2. FOV约束（硬约束）
            fov_ratio = abs(s.look_angle) / env.mp.fov
            r_fov = -1.0 if fov_ratio > 0.9 else 0.0

            episode_reward += r_energy + r_fov

        # 终端奖励（主要信号）
        if env.state.hit:
            episode_reward += 200.0  # 命中大奖励
            # 额外奖励：飞行时间短 → 好
            time_bonus = max(0, 10.0 - env.state.t)
            episode_reward += time_bonus
        else:
            # 未命中：严厉惩罚
            episode_reward -= 100.0
            # 按最终ZEM额外惩罚
            final_zem = env.compute_zem()
            episode_reward -= min(50.0, final_zem / 10.0)

        total_fitness += episode_reward

    return total_fitness / n_episodes


def train_es(obs_dim: int = 3, n_generations: int = 150,
             pop_size: int = 60, sigma: float = 0.05,
             lr: float = 0.03, n_eval: int = 8,
             verbose: bool = True):
    """
    用进化策略训练RL制导律
    改进：更多样化的训练场景 + 更稳定的更新策略
    """
    import time

    # 创建制导律
    guidance = RLGuidanceLaw(obs_dim=obs_dim)
    n_params = guidance.net.n_params()

    if verbose:
        print(f"  Network params: {n_params}")

    # 训练环境：更丰富的场景组合（对齐PNG.py）
    envs = [
        # 无机动
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=0, maneuver_type="none")),
        # 低强度机动
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=5*9.81, maneuver_type="step")),
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=5*9.81, maneuver_type="sine")),
        # 中等强度机动
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=8*9.81, maneuver_type="step")),
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=8*9.81, maneuver_type="sine")),
        # 高强度机动
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=10*9.81, maneuver_type="step")),
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=10*9.81, maneuver_type="sine")),
        MissileEngagement2D(
            target=TargetParams(V=50.0, a_max=10*9.81, maneuver_type="random")),
    ]

    # 当前最优
    best_params = guidance._flat.copy()
    best_fitness = -float('inf')

    history = {'gen': [], 'fitness': [], 'best': []}
    t0 = time.time()

    # 评估种子（固定一组，保证可比性）
    eval_seeds = list(range(n_eval))

    for gen in range(n_generations):
        # 生成噪声扰动
        noise = np.random.randn(pop_size, n_params)
        rewards = np.zeros(pop_size)

        # 循环使用不同环境（增加多样性）
        env = envs[gen % len(envs)]

        for i in range(pop_size):
            # 正向扰动
            perturbed = best_params + sigma * noise[i]
            guidance.net.set_flat(perturbed)
            guidance._flat = perturbed
            rewards[i] = evaluate_guidance(guidance, env, n_eval, eval_seeds)

        # 归一化奖励，计算梯度
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        # 参数更新（带动量）
        grad = (noise.T @ rewards_norm) / pop_size
        best_params = best_params + lr * grad

        # 评估当前最优（在多个环境上测试）
        guidance.net.set_flat(best_params)
        guidance._flat = best_params

        # 在3个不同环境上评估
        test_envs = [envs[0], envs[4], envs[7]]  # 无机动、中等、高强度
        current_fitness = sum(
            evaluate_guidance(guidance, e, n_episodes=3, seeds=list(range(100, 103)))
            for e in test_envs
        ) / len(test_envs)

        if current_fitness > best_fitness:
            best_fitness = current_fitness

        history['gen'].append(gen)
        history['fitness'].append(current_fitness)
        history['best'].append(best_fitness)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            elapsed = time.time() - t0
            print(f"  Gen {gen:3d}/{n_generations}  "
                  f"fitness={current_fitness:8.1f}  "
                  f"best={best_fitness:8.1f}  "
                  f"time={elapsed:.0f}s")

    # 恢复最优
    guidance.net.set_flat(best_params)
    guidance._flat = best_params

    return guidance, history


def save_guidance(guidance: RLGuidanceLaw, path: str):
    with open(path, 'wb') as f:
        pickle.dump({
            'params': guidance._flat,
            'obs_dim': guidance.obs_dim,
            'base_N': guidance.base_N,
        }, f)


def load_guidance(path: str) -> RLGuidanceLaw:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    g = RLGuidanceLaw(obs_dim=data['obs_dim'])
    g.net.set_flat(data['params'])
    g._flat = data['params']
    g.base_N = data['base_N']
    return g
