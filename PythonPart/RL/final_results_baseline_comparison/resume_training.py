"""
从checkpoint恢复训练
====================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from train_baseline_comparison import PartialObsEnv, create_training_envs


def find_latest_checkpoint(checkpoint_dir):
    """找到最新的checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
    if not checkpoints:
        return None

    # 提取步数并排序
    def get_steps(filename):
        try:
            # baseline_partial_obs_50000_steps.zip
            parts = filename.replace('.zip', '').split('_')
            for i, part in enumerate(parts):
                if part == 'steps' and i > 0:
                    return int(parts[i-1])
        except:
            pass
        return 0

    checkpoints.sort(key=get_steps, reverse=True)
    latest = os.path.join(checkpoint_dir, checkpoints[0])
    steps = get_steps(checkpoints[0])

    return latest, steps


def resume_training(
    checkpoint_path: str,
    current_steps: int,
    total_timesteps: int,
    n_envs: int,
    device: str,
    seed: int,
):
    """从checkpoint恢复训练"""

    exp_name = 'baseline_partial_obs'

    # 创建目录
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', exp_name)
    model_dir = os.path.join(os.path.dirname(__file__), 'models', exp_name)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', exp_name)

    print("=" * 70)
    print(f"  Resuming Training: {exp_name}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Current steps: {current_steps:,}")
    print(f"Remaining steps: {total_timesteps - current_steps:,}")
    print(f"Device: {device}")
    print()

    # 加载模型
    print("Loading model from checkpoint...")
    model = PPO.load(checkpoint_path)

    # 创建环境
    print("Creating environments...")
    train_env = create_training_envs(PartialObsEnv, n_envs, seed)
    train_env = VecMonitor(train_env, log_dir)

    eval_env = create_training_envs(PartialObsEnv, 4, seed + 10000)
    eval_env = VecMonitor(eval_env)

    # 设置环境
    model.set_env(train_env)

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

    # 继续训练
    remaining_steps = total_timesteps - current_steps
    print(f"Continuing training for {remaining_steps:,} steps...")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,  # 关键：不重置步数计数器
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

    print(f"\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: auto-detect latest)')
    parser.add_argument('--total_timesteps', type=int, default=2000000,
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

    # 查找checkpoint
    if args.checkpoint is None:
        checkpoint_dir = 'checkpoints/baseline_partial_obs'
        result = find_latest_checkpoint(checkpoint_dir)
        if result is None:
            print(f"No checkpoint found in {checkpoint_dir}")
            print("Start training from scratch with:")
            print("  python train_baseline_comparison.py --group group2")
            return
        checkpoint_path, current_steps = result
        print(f"Auto-detected checkpoint: {checkpoint_path}")
        print(f"Current steps: {current_steps:,}")
        print()
    else:
        checkpoint_path = args.checkpoint
        # 尝试从文件名提取步数
        current_steps = 0
        try:
            parts = os.path.basename(checkpoint_path).replace('.zip', '').split('_')
            for i, part in enumerate(parts):
                if part == 'steps' and i > 0:
                    current_steps = int(parts[i-1])
                    break
        except:
            pass

    if current_steps >= args.total_timesteps:
        print(f"Training already completed ({current_steps:,} >= {args.total_timesteps:,})")
        return

    resume_training(
        checkpoint_path=checkpoint_path,
        current_steps=current_steps,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
