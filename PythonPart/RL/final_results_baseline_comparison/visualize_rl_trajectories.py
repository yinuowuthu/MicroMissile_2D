"""
生成RL轨迹可视化图
================================
对比Group 1和Group 2在典型场景下的轨迹
"""

import os
import sys

# Add paths for imports
rl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, rl_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from train_baseline_comparison import FullObsEnv, PartialObsEnv

# 设置样式
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})


def run_rl_trajectory(env_class, model_path, a_max, seed=42):
    """运行RL模型获取单次轨迹"""
    env = env_class(
        target_params={
            'maneuver_type': 'random' if a_max > 0 else 'none',
            'a_max': a_max * 9.81
        }
    )
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed)

    trajectory = {
        'x_m': [env.env.state.xm],
        'y_m': [env.env.state.ym],
        'x_t': [env.env.state.xt],
        'y_t': [env.env.state.yt],
        'am': [0.0],
        'r': [env.env.state.r],
        't': [0.0],
    }

    done = False
    step_count = 0
    while not done and step_count < 1000:  # 防止无限循环
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        trajectory['x_m'].append(env.env.state.xm)
        trajectory['y_m'].append(env.env.state.ym)
        trajectory['x_t'].append(env.env.state.xt)
        trajectory['y_t'].append(env.env.state.yt)
        trajectory['am'].append(env.env.state.am)
        trajectory['r'].append(env.env.state.r)
        trajectory['t'].append(env.env.state.t)

    # 转换为numpy数组
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    trajectory['hit'] = env.env.state.hit
    trajectory['miss_distance'] = env.env.state.r_min
    trajectory['energy'] = np.sum(np.array(trajectory['am'])**2) * env.env.cfg.decision_dt

    return trajectory


def plot_single_trajectory(traj, method_name, color, ax_traj, ax_range, ax_accel, ax_energy):
    """绘制单个轨迹的4个子图"""

    # 1. 轨迹图
    ax_traj.plot(traj['x_m'], traj['y_m'], '-', color=color, linewidth=2.5, label='Missile', zorder=3)
    ax_traj.plot(traj['x_t'], traj['y_t'], '--', color='#E74C3C', linewidth=2.5, label='Target', zorder=2)

    # 起点标记（更小更精致）
    ax_traj.plot(traj['x_m'][0], traj['y_m'][0], 'o', color=color, markersize=4,
                markeredgewidth=1.5, markeredgecolor='white', label='M start', zorder=4)
    ax_traj.plot(traj['x_t'][0], traj['y_t'][0], 'o', color='#E74C3C', markersize=4,
                markeredgewidth=1.5, markeredgecolor='white', label='T start', zorder=4)

    # 终点标记（更小更精致）
    ax_traj.plot(traj['x_m'][-1], traj['y_m'][-1], 'X', color=color, markersize=6,
                markeredgewidth=2, label='M end', zorder=5)
    ax_traj.plot(traj['x_t'][-1], traj['y_t'][-1], 'X', color='#E74C3C', markersize=6,
                markeredgewidth=2, label='T end', zorder=5)

    hit_str = "HIT" if traj['hit'] else "MISS"
    ax_traj.set_title(f"{method_name}\n(miss={traj['miss_distance']:.3f}m, {hit_str})",
                     fontweight='bold', fontsize=11)
    ax_traj.set_xlabel('X (m)', fontweight='bold')
    ax_traj.set_ylabel('Y (m)', fontweight='bold')
    ax_traj.grid(True, alpha=0.3, linestyle='--')
    ax_traj.legend(loc='best', fontsize=8, framealpha=0.9)
    ax_traj.axis('equal')

    # 2. 距离-时间图
    ax_range.plot(traj['t'], traj['r'], '-', color=color, linewidth=2.5)
    ax_range.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Hit threshold')
    ax_range.fill_between(traj['t'], 0, 0.5, alpha=0.1, color='green')
    ax_range.set_title('Missile-Target Range', fontweight='bold')
    ax_range.set_xlabel('Time (s)', fontweight='bold')
    ax_range.set_ylabel('Range (m)', fontweight='bold')
    ax_range.grid(True, alpha=0.3, linestyle='--')
    ax_range.legend(fontsize=8, framealpha=0.9)

    # 3. 加速度-时间图
    ax_accel.plot(traj['t'], traj['am'], '-', color=color, linewidth=2.5)
    ax_accel.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_accel.set_title('Missile Acceleration', fontweight='bold')
    ax_accel.set_xlabel('Time (s)', fontweight='bold')
    ax_accel.set_ylabel('Acceleration (m/s²)', fontweight='bold')
    ax_accel.grid(True, alpha=0.3, linestyle='--')

    # 4. 累积能量-时间图
    cumulative_energy = np.cumsum(np.array(traj['am'])**2) * 0.02  # decision_dt = 0.02
    ax_energy.plot(traj['t'], cumulative_energy, '-', color=color, linewidth=2.5)
    ax_energy.set_title(f'Cumulative Energy (total={traj["energy"]:.0f} m²/s³)', fontweight='bold')
    ax_energy.set_xlabel('Time (s)', fontweight='bold')
    ax_energy.set_ylabel('Energy (m²/s³)', fontweight='bold')
    ax_energy.grid(True, alpha=0.3, linestyle='--')


def create_comparison_figure(a_max, seed=42):
    """创建Group 1 vs Group 2的对比图"""
    print(f"\nGenerating trajectory comparison for {a_max}g scenario...")

    print("  Running Group 1 (8D Full Obs)...")
    group1_traj = run_rl_trajectory(
        FullObsEnv,
        '../models/baseline_full_obs/best_model.zip',
        a_max, seed
    )

    print("  Running Group 2 (6D Partial Obs)...")
    group2_traj = run_rl_trajectory(
        PartialObsEnv,
        '../models/baseline_partial_obs/best_model.zip',
        a_max, seed
    )

    # 创建大图
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Group 1 (第一行)
    ax_g1_traj = fig.add_subplot(gs[0, 0])
    ax_g1_range = fig.add_subplot(gs[0, 1])
    ax_g1_accel = fig.add_subplot(gs[0, 2])
    ax_g1_energy = fig.add_subplot(gs[0, 3])
    plot_single_trajectory(group1_traj, 'Group 1 (8D Full Obs)', '#3498DB',
                          ax_g1_traj, ax_g1_range, ax_g1_accel, ax_g1_energy)

    # Group 2 (第二行)
    ax_g2_traj = fig.add_subplot(gs[1, 0])
    ax_g2_range = fig.add_subplot(gs[1, 1])
    ax_g2_accel = fig.add_subplot(gs[1, 2])
    ax_g2_energy = fig.add_subplot(gs[1, 3])
    plot_single_trajectory(group2_traj, 'Group 2 (6D Partial Obs)', '#2ECC71',
                          ax_g2_traj, ax_g2_range, ax_g2_accel, ax_g2_energy)

    fig.suptitle(f'RL Trajectory Comparison: {a_max}g Target Maneuver (seed={seed})',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*70)
    print("  Generating RL Trajectory Visualizations")
    print("="*70)

    # 生成三个典型场景的轨迹对比
    scenarios = [
        (0, 42),    # 0g: 简单场景
        (6, 42),    # 6g: 中等机动
        (10, 42),   # 10g: 高机动
    ]

    for a_max, seed in scenarios:
        try:
            fig = create_comparison_figure(a_max, seed)
            filename = f'rl_trajectory_comparison_{int(a_max)}g.png'
            filepath = os.path.join(base_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error in {a_max}g scenario: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("  RL trajectory visualizations generated successfully!")
    print("="*70)
    print("\nGenerated 3 trajectory comparison figures:")
    print("  rl_trajectory_comparison_0g.png  - Simple scenario")
    print("  rl_trajectory_comparison_6g.png  - Medium maneuver")
    print("  rl_trajectory_comparison_10g.png - High maneuver")
    print("\nEach figure contains:")
    print("  - 2 rows (Group 1, Group 2)")
    print("  - 4 columns (Trajectory, Range, Acceleration, Energy)")


if __name__ == "__main__":
    main()
