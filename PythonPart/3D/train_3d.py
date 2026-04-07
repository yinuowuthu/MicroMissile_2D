"""
3D PPO训练 + PNG基线对比
========================
用stable-baselines3的PPO训练3D导弹制导策略，
并与PNG基线进行对比评估。

用法：
    python train_3d.py --timesteps 500000
    python train_3d.py --eval_only --model_path models/ppo_3d/best_model.zip
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_missile_env_3d import MissileGymEnv3D, make_env
from missile_env_3d import (
    MissileEngagement3D, MissileParams, TargetParams, SimConfig, run_episode
)
from guidance_3d import ProportionalNavigation3D

G = 9.81


# ============================================================
#  训练环境构建
# ============================================================

def create_training_envs(n_envs: int, seed: int):
    """创建多场景训练环境"""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    scenarios = [
        {'maneuver_type': 'none'},
        {'maneuver_type': 'step',   'a_max': 3 * G},
        {'maneuver_type': 'sine',   'a_max': 5 * G},
        {'maneuver_type': 'spiral', 'a_max': 5 * G},
        {'maneuver_type': 'random', 'a_max': 5 * G},
        {'maneuver_type': 'step',   'a_max': 5 * G},
    ]

    env_fns = []
    for i in range(n_envs):
        sc = scenarios[i % len(scenarios)]
        env_fns.append(make_env(rank=i, seed=seed + i, target_params=sc))

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


# ============================================================
#  PNG基线评估
# ============================================================

def evaluate_png(n_runs: int = 100, N: float = 4.0):
    """评估PNG基线"""
    mp = MissileParams()
    cfg = SimConfig()
    guidance = ProportionalNavigation3D(N=N)

    results = {}
    for man in ['none', 'step', 'sine', 'spiral', 'random']:
        tp = TargetParams(maneuver_type=man)
        env = MissileEngagement3D(missile=mp, target=tp, config=cfg)
        hits, misses, times = 0, [], []
        for seed in range(n_runs):
            traj = run_episode(env, guidance, seed=seed)
            if traj['hit']:
                hits += 1
            misses.append(traj['miss_distance'])
            times.append(traj['flight_time'])
        results[man] = {
            'hit_rate': hits / n_runs,
            'miss_mean': np.mean(misses),
            'miss_std': np.std(misses),
            'time_mean': np.mean(times),
        }
    return results


# ============================================================
#  PPO模型评估
# ============================================================

def evaluate_ppo(model, n_runs: int = 100):
    """评估PPO策略"""
    mp = MissileParams()
    cfg = SimConfig()

    results = {}
    for man in ['none', 'step', 'sine', 'spiral', 'random']:
        tp = TargetParams(maneuver_type=man)
        env_inner = MissileEngagement3D(missile=mp, target=tp, config=cfg)
        gym_env = MissileGymEnv3D(target_params={
            'maneuver_type': man,
            'a_max': tp.a_max,
        })

        hits, misses, times = 0, [], []
        for seed in range(n_runs):
            obs, _ = gym_env.reset(seed=seed)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = gym_env.step(action)
                done = terminated or truncated

            s = gym_env.env.state
            if s.hit:
                hits += 1
            misses.append(s.r_min)
            times.append(s.t)

        results[man] = {
            'hit_rate': hits / n_runs,
            'miss_mean': np.mean(misses),
            'miss_std': np.std(misses),
            'time_mean': np.mean(times),
        }
    return results


def print_comparison(png_results, ppo_results):
    """打印PNG vs PPO对比表"""
    print(f"\n{'Maneuver':10s} | {'PNG hit%':>8s} {'miss':>8s} | {'PPO hit%':>8s} {'miss':>8s} | {'Δhit%':>6s}")
    print("-" * 65)
    for man in ['none', 'step', 'sine', 'spiral', 'random']:
        pr = png_results[man]
        pp = ppo_results[man]
        delta = (pp['hit_rate'] - pr['hit_rate']) * 100
        sign = '+' if delta >= 0 else ''
        print(f"{man:10s} | {pr['hit_rate']*100:7.1f}% {pr['miss_mean']:7.3f}m | "
              f"{pp['hit_rate']*100:7.1f}% {pp['miss_mean']:7.3f}m | {sign}{delta:5.1f}%")


# ============================================================
#  训练主函数
# ============================================================

def train(args):
    """训练PPO模型"""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.logger import configure

    # 设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # 目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs', 'ppo_3d')
    model_dir = os.path.join(base_dir, 'models', 'ppo_3d')
    ckpt_dir = os.path.join(base_dir, 'checkpoints', 'ppo_3d')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("  3D PPO Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Envs: {args.n_envs}")
    print(f"Seed: {args.seed}")
    print()

    # --- PNG基线 ---
    print("Evaluating PNG baseline (N=4)...")
    png_results = evaluate_png(n_runs=50)
    for man, r in png_results.items():
        print(f"  {man:8s}: hit={r['hit_rate']*100:.0f}%  miss={r['miss_mean']:.3f}m")
    print()

    # --- 创建环境 ---
    train_env = create_training_envs(args.n_envs, args.seed)
    eval_env = create_training_envs(4, args.seed + 10000)

    # --- PPO模型 ---
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 更大网络
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
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
        tensorboard_log=None,
        seed=args.seed,
    )

    logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(logger)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50000,
        save_path=ckpt_dir,
        name_prefix='ppo_3d',
    )

    # --- 训练 ---
    print("Starting training...")
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed:.0f}s")

    # 保存
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"Model saved: {final_path}.zip")

    # --- 评估PPO ---
    print("\nEvaluating PPO...")
    ppo_results = evaluate_ppo(model, n_runs=50)
    print_comparison(png_results, ppo_results)

    train_env.close()
    eval_env.close()


def eval_only(args):
    """仅评估已有模型"""
    from stable_baselines3 import PPO

    print("Loading model:", args.model_path)
    model = PPO.load(args.model_path)

    print("Evaluating PNG baseline...")
    png_results = evaluate_png(n_runs=100)

    print("Evaluating PPO...")
    ppo_results = evaluate_ppo(model, n_runs=100)

    print_comparison(png_results, ppo_results)


def main():
    parser = argparse.ArgumentParser(description='3D PPO Missile Guidance Training')
    parser.add_argument('--timesteps', type=int, default=2000000)
    parser.add_argument('--n_envs', type=int, default=6)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--model_path', type=str, default='models/ppo_3d/best_model.zip')
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
