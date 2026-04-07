"""
3D RL制导律 — PNG + 学习修正 + ES训练
======================================
核心思路（He et al. 2021 扩展到3D）：
  不从零学习制导指令，而是在PNG基础上学习修正量
  a_el = N_el(obs) * Vm * lam_el_dot * cos(lam_el) + corr_el(obs)
  a_az = N_az(obs) * Vm * lam_az_dot               + corr_az(obs)

  N_el, N_az: RL学到的自适应导航比（初始≈4）
  corr: 额外修正项

两种使用方式：
  1. ES训练的轻量级SmallNet（纯numpy，可嵌入式部署）
  2. PPO训练的SB3模型包装器（训练效果更好）
"""

import numpy as np
import pickle
from typing import List, Tuple, Optional

from missile_env_3d import (
    MissileEngagement3D, MissileParams, TargetParams, SimConfig, run_episode
)
from guidance_3d import ProportionalNavigation3D

G = 9.81


# ============================================================
#  轻量级MLP（纯numpy，可部署）
# ============================================================

class SmallNet:
    """极简MLP，纯numpy实现"""
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


# ============================================================
#  3D RL制导律（ES版）
# ============================================================

class RLGuidanceLaw3D:
    """
    3D RL制导律：PNG + 神经网络修正

    网络输入：8维观测（与gym环境一致）
    网络输出：4维 [delta_N_el, delta_N_az, corr_el, corr_az]

    最终指令：
      a_el = (N_base + delta_N_el) * Vm * lam_el_dot + corr_el * a_max * 0.1
      a_az = (N_base + delta_N_az) * Vm * lam_az_dot + corr_az * a_max * 0.1
    """

    def __init__(self, obs_dim: int = 8):
        self.obs_dim = obs_dim
        self.net = SmallNet([obs_dim, 64, 32, 4])
        self.base_N = 4.0
        self._flat = self.net.get_flat()

    def compute(self, env: MissileEngagement3D) -> Tuple[float, float]:
        s = env.state
        obs = env.get_obs().reshape(1, -1)
        out = self.net.forward(obs)[0]

        # 输出1-2: 导航比修正（tanh映射到±3，N在1~7之间）
        delta_N_el = np.tanh(out[0]) * 3.0
        delta_N_az = np.tanh(out[1]) * 3.0
        N_el = self.base_N + delta_N_el
        N_az = self.base_N + delta_N_az

        # 输出3-4: 额外修正量
        corr_el = np.tanh(out[2]) * 0.1 * env.mp.a_max
        corr_az = np.tanh(out[3]) * 0.1 * env.mp.a_max

        # 最终指令 = 自适应PNG + 修正
        cos_el = np.cos(s.lam_el)
        a_el = N_el * env.mp.V * s.lam_el_dot * cos_el + corr_el
        a_az = N_az * env.mp.V * s.lam_az_dot + corr_az

        return a_el, a_az


# ============================================================
#  PPO模型包装器（用SB3训练的模型）
# ============================================================

class PPOGuidanceLaw3D:
    """
    将SB3 PPO模型包装成制导律接口

    用法：
        from stable_baselines3 import PPO
        model = PPO.load("models/ppo_3d/best_model.zip")
        guidance = PPOGuidanceLaw3D(model)
        a_el, a_az = guidance.compute(env)
    """

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def compute(self, env: MissileEngagement3D) -> Tuple[float, float]:
        obs = env.get_obs()
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        a_el = float(action[0]) * env.mp.a_max
        a_az = float(action[1]) * env.mp.a_max
        return a_el, a_az


# ============================================================
#  ES训练
# ============================================================

def evaluate_guidance(guidance: RLGuidanceLaw3D, env: MissileEngagement3D,
                      n_episodes: int = 5, seeds: List[int] = None) -> float:
    """评估制导律适应度"""
    if seeds is None:
        seeds = list(range(n_episodes))

    total_fitness = 0.0

    for seed in seeds:
        env.reset(seed=seed)
        steps = 0
        episode_reward = 0.0

        while not env.state.done and steps < 2000:
            a_el, a_az = guidance.compute(env)
            env.step_guidance(a_el, a_az)
            steps += 1

            s = env.state

            # 轻微能量惩罚
            a_norm_sq = (s.am_el / env.mp.a_max)**2 + (s.am_az / env.mp.a_max)**2
            episode_reward -= 0.001 * a_norm_sq

            # FOV约束
            fov_ratio = s.look_total / env.mp.fov
            if fov_ratio > 0.9:
                episode_reward -= 1.0

        # 终端奖励
        if env.state.hit:
            episode_reward += 200.0
            episode_reward += max(0, 10.0 - env.state.t)
        else:
            episode_reward -= 100.0
            episode_reward -= min(50.0, env.state.r_min / 10.0)

        total_fitness += episode_reward

    return total_fitness / n_episodes


def train_es(obs_dim: int = 8, n_generations: int = 200,
             pop_size: int = 80, sigma: float = 0.05,
             lr: float = 0.03, n_eval: int = 8,
             verbose: bool = True):
    """
    用进化策略训练3D RL制导律
    """
    import time

    guidance = RLGuidanceLaw3D(obs_dim=obs_dim)
    n_params = guidance.net.n_params()

    if verbose:
        print(f"  Network params: {n_params}")

    # 训练环境：多场景
    envs = [
        MissileEngagement3D(target=TargetParams(maneuver_type="none")),
        MissileEngagement3D(target=TargetParams(maneuver_type="step", a_max=3*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="sine", a_max=5*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="spiral", a_max=5*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="random", a_max=5*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="step", a_max=5*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="sine", a_max=8*G)),
        MissileEngagement3D(target=TargetParams(maneuver_type="random", a_max=8*G)),
    ]

    best_params = guidance._flat.copy()
    best_fitness = -float('inf')

    history = {'gen': [], 'fitness': [], 'best': []}
    eval_seeds = list(range(n_eval))
    t0 = time.time()

    for gen in range(n_generations):
        noise = np.random.randn(pop_size, n_params)
        rewards = np.zeros(pop_size)

        env = envs[gen % len(envs)]

        for i in range(pop_size):
            perturbed = best_params + sigma * noise[i]
            guidance.net.set_flat(perturbed)
            guidance._flat = perturbed
            rewards[i] = evaluate_guidance(guidance, env, n_eval, eval_seeds)

        # 归一化 + 梯度更新
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        rewards_norm = (rewards - r_mean) / r_std

        grad = (noise.T @ rewards_norm) / pop_size
        best_params = best_params + lr * grad

        # 评估当前最优
        guidance.net.set_flat(best_params)
        guidance._flat = best_params

        test_envs = [envs[0], envs[4], envs[7]]
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

    guidance.net.set_flat(best_params)
    guidance._flat = best_params
    return guidance, history


# ============================================================
#  保存/加载
# ============================================================

def save_guidance(guidance: RLGuidanceLaw3D, path: str):
    with open(path, 'wb') as f:
        pickle.dump({
            'params': guidance._flat,
            'obs_dim': guidance.obs_dim,
            'base_N': guidance.base_N,
        }, f)


def load_guidance(path: str) -> RLGuidanceLaw3D:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    g = RLGuidanceLaw3D(obs_dim=data['obs_dim'])
    g.net.set_flat(data['params'])
    g._flat = data['params']
    g.base_N = data['base_N']
    return g


# ============================================================
#  快速测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  3D RL Guidance Law - Quick Test")
    print("=" * 60)

    # 测试ES制导律（未训练）
    print("\n--- ES Guidance (untrained) ---")
    g_es = RLGuidanceLaw3D()
    env = MissileEngagement3D()
    for seed in range(5):
        traj = run_episode(env, g_es, seed=seed)
        print(f"  seed={seed}: hit={traj['hit']}, miss={traj['miss_distance']:.3f}m, "
              f"t={traj['flight_time']:.2f}s, {traj['reason']}")

    # 测试PNG基线
    print("\n--- PNG Baseline (N=4) ---")
    g_png = ProportionalNavigation3D(N=4)
    for seed in range(5):
        traj = run_episode(env, g_png, seed=seed)
        print(f"  seed={seed}: hit={traj['hit']}, miss={traj['miss_distance']:.3f}m, "
              f"t={traj['flight_time']:.2f}s, {traj['reason']}")

    # 测试PPO包装器（如果模型存在）
    try:
        from stable_baselines3 import PPO
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'ppo_3d', 'best_model.zip')
        if os.path.exists(model_path):
            print(f"\n--- PPO Guidance ({model_path}) ---")
            model = PPO.load(model_path)
            g_ppo = PPOGuidanceLaw3D(model)
            for seed in range(5):
                traj = run_episode(env, g_ppo, seed=seed)
                print(f"  seed={seed}: hit={traj['hit']}, miss={traj['miss_distance']:.3f}m, "
                      f"t={traj['flight_time']:.2f}s, {traj['reason']}")
    except ImportError:
        print("\n(stable-baselines3 not installed, skipping PPO test)")
