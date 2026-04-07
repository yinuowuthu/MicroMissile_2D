"""
基线对比实验：完整观测 vs 部分观测
====================================
核心原则：只改观测空间，其他完全相同

Group 1 (Full Obs, 8D): [lam_dot, look_angle, lam_ddot, am, gamma_m, t, r_dot, r]
Group 2 (Partial Obs, 6D): [lam_dot, look_angle, lam_ddot, am, gamma_m, t]

关键：
- Group 1 = Group 2 + {r, r_dot}（严格超集关系）
- 使用原始的归一化系数（lam_dot/0.5，已验证80%命中率）
- 使用原始的奖励函数（ZEM-based，包含r和r_dot）
- 使用原始的训练配置（4场景，n_envs=4）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym_missile_env import MissileGymEnv


# ============================================================================
# Group 1: 完整观测（原始版本，已验证80%）
# ============================================================================

class FullObsEnv(MissileGymEnv):
    """
    完整观测环境（8维）- Group 2的超集

    观测空间：
        [0] lam_dot_norm   : 视线角速率 / 0.5
        [1] look_angle_norm: 视线偏差角 / fov
        [2] lam_ddot_norm  : 视线角加速度 / 5.0
        [3] am_norm        : 当前加速度 / a_max
        [4] gamma_m_norm   : 弹体航向角 / π
        [5] t_norm         : 飞行时间 / t_max
        [6] r_dot_norm     : 接近速率 / V
        [7] r_norm         : 弹目距离 / r_init_max
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from gymnasium import spaces
        # 重新定义观测空间：8维
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(8,), dtype=np.float32
        )
        # 用于计算lam_ddot的历史缓冲
        self._prev_lam_dot = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # 重置历史缓冲
        self._prev_lam_dot = self.env.state.lam_dot
        # 返回完整观测
        full_obs = self._get_full_obs()
        return full_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # 返回完整观测
        full_obs = self._get_full_obs()
        return full_obs, reward, terminated, truncated, info

    def _get_full_obs(self):
        """构造完整观测向量（8维）"""
        s = self.env.state
        mp = self.env.mp
        cfg = self.env.cfg

        # 计算lam_ddot（视线角加速度）通过差分
        lam_ddot = (s.lam_dot - self._prev_lam_dot) / cfg.decision_dt
        self._prev_lam_dot = s.lam_dot

        obs = np.array([
            s.lam_dot / 0.5,                    # [0] 视线角速率
            s.look_angle / mp.fov,              # [1] 视线偏差角
            lam_ddot / 5.0,                     # [2] 视线角加速度
            s.am / mp.a_max,                    # [3] 当前加速度
            s.gamma_m / np.pi,                  # [4] 弹体航向角
            s.t / cfg.t_max,                    # [5] 飞行时间
            s.r_dot / mp.V,                     # [6] 接近速率
            s.r / cfg.r_init_max,               # [7] 弹目距离
        ], dtype=np.float32)

        return obs


# ============================================================================
# Group 2: 部分观测（移除r和r_dot，增加lam_ddot, gamma_m, t）
# ============================================================================

class PartialObsEnv(MissileGymEnv):
    """
    部分观测环境（6维）

    观测空间：
        [0] lam_dot_norm   : 视线角速率 / 0.5
        [1] look_angle_norm: 视线偏差角 / fov
        [2] lam_ddot_norm  : 视线角加速度 / 5.0
        [3] am_norm        : 当前加速度 / a_max
        [4] gamma_m_norm   : 弹体航向角 / π
        [5] t_norm         : 飞行时间 / t_max
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from gymnasium import spaces
        # 重新定义观测空间：6维
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # 用于计算lam_ddot的历史缓冲
        self._prev_lam_dot = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # 重置历史缓冲
        self._prev_lam_dot = self.env.state.lam_dot
        # 返回部分观测
        partial_obs = self._get_partial_obs()
        return partial_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # 返回部分观测
        partial_obs = self._get_partial_obs()
        return partial_obs, reward, terminated, truncated, info

    def _get_partial_obs(self):
        """构造部分观测向量"""
        s = self.env.state
        mp = self.env.mp
        cfg = self.env.cfg

        # 计算lam_ddot（视线角加速度）通过差分
        lam_ddot = (s.lam_dot - self._prev_lam_dot) / cfg.decision_dt
        self._prev_lam_dot = s.lam_dot

        obs = np.array([
            s.lam_dot / 0.5,                    # [0] 视线角速率（与Group 1一致）
            s.look_angle / mp.fov,              # [1] 视线偏差角
            lam_ddot / 5.0,                     # [2] 视线角加速度（匹配lam_dot量级）
            s.am / mp.a_max,                    # [3] 当前加速度
            s.gamma_m / np.pi,                  # [4] 弹体航向角
            s.t / cfg.t_max,                    # [5] 飞行时间
        ], dtype=np.float32)

        return obs


# ============================================================================
# 训练函数
# ============================================================================

def make_env(env_class, rank: int, seed: int = 0, target_params=None):
    """创建环境的工厂函数"""
    def _init():
        env = env_class(target_params=target_params)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_training_envs(env_class, n_envs: int, seed: int):
    """创建训练环境（4个固定场景）"""
    scenarios = [
        {'maneuver_type': 'none', 'a_max': 0},
        {'maneuver_type': 'step', 'a_max': 5*9.81},
        {'maneuver_type': 'sine', 'a_max': 8*9.81},
        {'maneuver_type': 'random', 'a_max': 10*9.81},
    ]

    env_fns = []
    for i in range(n_envs):
        scenario = scenarios[i % len(scenarios)]
        env_fns.append(make_env(env_class, rank=i, seed=seed, target_params=scenario))

    return DummyVecEnv(env_fns)


def train_model(
    exp_name: str,
    env_class,
    timesteps: int,
    n_envs: int,
    device: str,
    seed: int,
):
    """训练PPO模型"""

    # 创建目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', exp_name)
    model_dir = os.path.join(os.path.dirname(__file__), 'models', exp_name)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Experiment: {exp_name}")
    print("=" * 70)
    print(f"Environment: {env_class.__name__}")
    print(f"Device: {device}")
    print(f"Training timesteps: {timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Seed: {seed}")
    print()
    print("Configuration:")
    print("  - Normalization: lam_dot/0.5 (original, verified 80%)")
    print("  - Reward: ZEM-based (original)")
    print("  - Scenarios: 4 fixed (0g, 5g, 8g, 10g)")
    print()

    # 创建环境
    print("Creating environments...")
    train_env = create_training_envs(env_class, n_envs, seed)
    train_env = VecMonitor(train_env, log_dir)

    eval_env = create_training_envs(env_class, 4, seed + 10000)
    eval_env = VecMonitor(eval_env)

    # 创建PPO模型（使用原始配置）
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # 配置日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix=exp_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # 开始训练
    print("Starting training...")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_callback, checkpoint_callback],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "best_model.zip")
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    # 关闭环境
    train_env.close()
    eval_env.close()

    print(f"\n{exp_name} training completed!")


def main():
    parser = argparse.ArgumentParser(description='Baseline comparison: Full vs Partial Obs')
    parser.add_argument('--group', type=str, required=True,
                        choices=['group1', 'group2', 'both'],
                        help='Which group to train')
    parser.add_argument('--timesteps', type=int, default=2000000,
                        help='Total training timesteps (default: 2000000)')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # 训练Group 1（完整观测，5维）
    if args.group in ['group1', 'both']:
        train_model(
            exp_name='baseline_full_obs',
            env_class=FullObsEnv,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            device=device,
            seed=args.seed,
        )

    # 训练Group 2（部分观测，6维）
    if args.group in ['group2', 'both']:
        train_model(
            exp_name='baseline_partial_obs',
            env_class=PartialObsEnv,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            device=device,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("  All training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
