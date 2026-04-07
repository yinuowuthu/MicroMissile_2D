"""
公平对比实验：统一奖励函数
============================
核心原则：两组使用完全相同的奖励函数，只依赖可观测量

奖励函数设计（基于比例导引物理本质）：
  - 密集奖励：-|λ̇| （鼓励视线角速率趋零，碰撞三角形成）
  - 能量惩罚：-0.01 * (am/amax)² （节能）
  - 终端奖励：+100 if hit else -100 （稀疏信号，可接受）

对比设置：
  Group 1 (Full Obs): [λ, λ̇, r, ṙ, am, γ_m, t] - 7维
  Group 2 (Partial Obs): [λ, λ̇, λ̈, am, γ_m, t] - 6维

唯一区别：观测空间（Group 1多了r和ṙ）
相同条件：奖励函数、训练步数、超参数、随机种子
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

from train_engineering_partial_obs import FullObsWrapper, EngineeringPartialObsWrapper


# ============================================================================
# 统一奖励函数包装器
# ============================================================================

class UnifiedRewardWrapper:
    """
    统一奖励函数基类

    奖励设计（只用可观测量）：
    1. r_los_rate: -|λ̇| （视线角速率趋零 = 碰撞）
    2. r_energy: -0.01 * (am/amax)² （节能）
    3. r_terminal: +100 if hit else -100 （稀疏终端信号）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs, info

    def step(self, action):
        # 动作映射：[-1, 1] → [-a_max, a_max]
        ac = float(action[0]) * self.env.mp.a_max

        # 推进仿真
        self.env.step_guidance(ac)
        self._episode_length += 1

        # 获取观测（子类实现）
        obs = self._get_obs_impl()

        # 计算统一奖励
        reward = self._compute_unified_reward()
        self._episode_reward += reward

        # 终止条件
        terminated = self.env.state.done
        truncated = False

        # 信息字典
        info = {
            'r': self.env.state.r,
            'lam_dot': self.env.state.lam_dot,
            'look_angle': self.env.state.look_angle,
            'am': self.env.state.am,
            't': self.env.state.t,
        }

        if terminated:
            info['episode'] = {
                'r': self._episode_reward,
                'l': self._episode_length,
                'hit': self.env.state.hit,
                'reason': self.env.state.reason,
                'miss_distance': self.env.state.r if not self.env.state.hit else 0.0,
            }

        return obs, reward, terminated, truncated, info

    def _get_obs_impl(self):
        """子类实现：返回对应的观测"""
        raise NotImplementedError

    def _compute_unified_reward(self) -> float:
        """
        统一奖励函数（只用可观测量）+ 时间加权紧迫感

        物理意义：
        - λ̇ → 0 意味着碰撞三角形成（比例导引的核心）
        - 时间加权制造"紧迫感"：越接近末端，λ̇偏差惩罚越重
        - 能量惩罚鼓励节能
        - 终端奖励提供稀疏的成功/失败信号
        """
        s = self.env.state

        # 1. 时间加权的视线角速率惩罚（核心改进）
        # 越接近末端（t → t_max），权重越大，制造"紧迫感"
        t_ratio = s.t / self.env.cfg.t_max  # [0, 1]
        urgency = 1.0 + 10.0 * (t_ratio ** 2)  # [1.0, 11.0]
        r_los_rate = -urgency * abs(s.lam_dot)

        # 2. 能量惩罚（节能，权重降低避免压制主要目标）
        a_norm = s.am / self.env.mp.a_max
        r_energy = -0.001 * (a_norm ** 2)

        # 3. 终端奖励（稀疏信号，权重增大）
        r_terminal = 0.0
        if s.done:
            if s.hit:
                r_terminal = 500.0  # 增大到500
            else:
                r_terminal = -500.0  # 增大到-500

        return r_los_rate + r_energy + r_terminal


class FullObsUnifiedReward(UnifiedRewardWrapper, FullObsWrapper):
    """完整观测 + 统一奖励"""

    def _get_obs_impl(self):
        return self._get_full_obs()


class PartialObsUnifiedReward(UnifiedRewardWrapper, EngineeringPartialObsWrapper):
    """部分观测 + 统一奖励"""

    def _get_obs_impl(self):
        return self._get_partial_obs()


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
    """创建训练环境"""
    # 使用随机机动场景，让单环境也能见到多样性
    # a_max在0-10g之间随机，机动类型随机
    env_fns = []
    for i in range(n_envs):
        # 每个环境用None作为target_params，让环境自己随机化
        env_fns.append(make_env(env_class, rank=i, seed=seed, target_params=None))

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
    print("Unified Reward Function:")
    print("  r = -|λ̇| - 0.01*(am/amax)² + terminal_reward")
    print("  terminal_reward = +100 if hit else -100")
    print()

    # 创建环境
    print("Creating environments...")
    train_env = create_training_envs(env_class, n_envs, seed)
    train_env = VecMonitor(train_env, log_dir)

    eval_env = create_training_envs(env_class, 4, seed + 10000)
    eval_env = VecMonitor(eval_env)

    # 创建PPO模型
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
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
        save_freq=100000,
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
    parser = argparse.ArgumentParser(description='Fair comparison with unified reward')
    parser.add_argument('--group', type=str, required=True,
                        choices=['group1', 'group2', 'both'],
                        help='Which group to train')
    parser.add_argument('--timesteps', type=int, default=2000000,
                        help='Total training timesteps (default: 2000000)')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments (default: 1)')
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

    # 训练Group 1
    if args.group in ['group1', 'both']:
        train_model(
            exp_name='fair_full_obs',
            env_class=FullObsUnifiedReward,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            device=device,
            seed=args.seed,
        )

    # 训练Group 2
    if args.group in ['group2', 'both']:
        train_model(
            exp_name='fair_partial_obs',
            env_class=PartialObsUnifiedReward,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            device=device,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("  All training completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Evaluate both models")
    print("2. Compare performance with fair reward function")


if __name__ == "__main__":
    main()
