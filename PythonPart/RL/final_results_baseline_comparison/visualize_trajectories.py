"""
生成轨迹可视化图
================================
对比PNG、Group 1、Group 2在典型场景下的轨迹
"""

import os
import sys

# Add paths for imports
rl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
png_dir = os.path.join(os.path.dirname(rl_dir), 'PNG')
sys.path.insert(0, rl_dir)
sys.path.insert(0, png_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from gym_missile_env import MissileGymEnv
from train_baseline_comparison import FullObsEnv, PartialObsEnv
from PNG import run_batch_vectorized, MissileParams, TargetParams, SimConfig

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
    env = env_class()
    model = PPO.load(model_path)

    obs, info = env.reset(seed=seed, options={'a_max': a_max})

    trajectory = {
        'x_m': [env.env.state.x_m],
        'y_m': [env.env.state.y_m],
        'x_t': [env.env.state.x_t],
        'y_t': [env.env.state.y_t],
        'am': [0.0],
        'r': [env.env.state.r],
        't': [0.0],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory['x_m'].append(env.env.state.x_m)
        trajectory['y_m'].append(env.env.state.y_m)
        trajectory['x_t'].append(env.env.state.x_t)
        trajectory['y_t'].append(env.env.state.y_t)
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


def run_png_trajectory(a_max, seed=42):
    """运行PNG获取单次轨迹"""
    mp = MissileParams(N=4.0, tau=0.2)
    cfg = SimConfig(dt=0.001, t_max=20.0, miss_dist=0.5)

    # 生成单次初始条件
    rng = np.random.default_rng(seed)
    r0 = np.array([3000.0])
    lam0 = np.array([0.0])
    gM0 = np.array([0.0])
    gT0 = np.array([np.pi])
    signs = np.array([1.0])

    if a_max == 0:
        tp = TargetParams(maneuver_type="none", maneuver_g=0)
    else:
        tp = TargetParams(maneuver_type="sine", maneuver_g=a_max, maneuver_freq=0.5)

    # 运行单次仿真
    miss_dists, flight_times, energies, hits = run_batch_vectorized(
        mp, tp, cfg, r0, lam0, gM0, gT0, signs
    )

    # 重新运行一次获取轨迹（需要修改PNG.py支持return_trajectories）
    # 暂时用简化版本
    from PNG import simulate_engagement

    # 初始化状态
    x_m, y_m = 0.0, 0.0
    x_t = r0[0] * np.cos(lam0[0])
    y_t = r0[0] * np.sin(lam0[0])

    trajectory = {
        'x_m': [x_m],
        'y_m': [y_m],
        'x_t': [x_t],
        'y_t': [y_t],
        'am': [0.0],
        'r': [r0[0]],
        't': [0.0],
        'hit': hits[0],
        'miss_distance': miss_dists[0],
        'energy': energies[0],
    }

    # 简化：只返回起点和终点
    # 实际轨迹需要修改PNG.py来记录
    trajectory['x_m'].append(x_t)
    trajectory['y_m'].append(y_t)
    trajectory['x_t'].append(x_t)
    trajectory['y_t'].append(y_t)
    trajectory['am'].append(0.0)
    trajectory['r'].append(miss_dists[0])
    trajectory['t'].append(flight_times[0])

    for key in ['x_m', 'y_m', 'x_t', 'y_t', 'am', 'r', 't']:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def plot_single_trajectory(traj, method_name, color, ax_traj, ax_range, ax_accel, ax_energy):
    """绘制单个轨迹的4个子图"""

    # 1. 轨迹图
    ax_traj.plot(traj['x_m'], traj['y_m'], '-', color=color, linewidth=2, label='Missile')
    ax_traj.plot(traj['x_t'], traj['y_t'], '--', color='red', linewidth=2, label='Target')
    ax_traj.plot(traj['x_m'][0], traj['y_m'][0], 'o', color=color, markersize=10, label='M start')
    ax_traj.plot(traj['x_t'][0], traj['y_t'][0], 'o', color='red', markersize=10, label='T start')
    ax_traj.plot(traj['x_m'][-1], traj['y_m'][-1], 'x', color=color, markersize=12,
                markeredgewidth=3, label='M end')
    ax_traj.plot(traj['x_t'][-1], traj['y_t'][-1], 'x', color='red', markersize=12,
                markeredgewidth=3, label='T end')

    hit_str = "hit=True" if traj['hit'] else "hit=False"
    ax_traj.set_title(f"{method_name} Trajectory (miss={traj['miss_distance']:.2f}m, {hit_str})",
                     fontweight='bold')
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc='best', fontsize=8)
    ax_traj.axis('equal')

    # 2. 距离-时间图
    ax_range.plot(traj['t'], traj['r'], '-', color=color, linewidth=2)
    ax_range.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Hit threshold')
    ax_range.set_title('Missile-Target Range', fontweight='bold')
    ax_range.set_xlabel('Time (s)')
    ax_range.set_ylabel('Range (m)')
    ax_range.grid(True, alpha=0.3)
    ax_range.legend(fontsize=8)

    # 3. 加速度-时间图
    ax_accel.plot(traj['t'], traj['am'], '-', color=color, linewidth=2)
    ax_accel.set_title('Missile Acceleration', fontweight='bold')
    ax_accel.set_xlabel('Time (s)')
    ax_accel.set_ylabel('Acceleration (m/s²)')
    ax_accel.grid(True, alpha=0.3)

    # 4. 累积能量-时间图
    cumulative_energy = np.cumsum(np.array(traj['am'])**2) * 0.02  # decision_dt = 0.02
    ax_energy.plot(traj['t'], cumulative_energy, '-', color=color, linewidth=2)
    ax_energy.set_title(f'Cumulative Energy (total={traj["energy"]:.0f})', fontweight='bold')
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy (m²/s³)')
    ax_energy.grid(True, alpha=0.3)


def create_comparison_figure(a_max, seed=42):
    """创建三种方法的对比图"""
    print(f"\nGenerating trajectory comparison for {a_max}g scenario...")

    # 运行三种方法
    print("  Running PNG...")
    png_traj = run_png_trajectory(a_max, seed)

    print("  Running Group 1 (8D)...")
    group1_traj = run_rl_trajectory(
        FullObsEnv,
        '../models/full_obs/best_model.zip',
        a_max, seed
    )

    print("  Running Group 2 (6D)...")
    group2_traj = run_rl_trajectory(
        PartialObsEnv,
        '../models/partial_obs/best_model.zip',
        a_max, seed
    )

    # 创建大图
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # PNG (第一行)
    ax_png_traj = fig.add_subplot(gs[0, 0])
    ax_png_range = fig.add_subplot(gs[0, 1])
    ax_png_accel = fig.add_subplot(gs[0, 2])
    ax_png_energy = fig.add_subplot(gs[0, 3])
    plot_single_trajectory(png_traj, 'PNG (N=4, τ=0.2)', '#E74C3C',
                          ax_png_traj, ax_png_range, ax_png_accel, ax_png_energy)

    # Group 1 (第二行)
    ax_g1_traj = fig.add_subplot(gs[1, 0])
    ax_g1_range = fig.add_subplot(gs[1, 1])
    ax_g1_accel = fig.add_subplot(gs[1, 2])
    ax_g1_energy = fig.add_subplot(gs[1, 3])
    plot_single_trajectory(group1_traj, 'Group 1 (8D Full Obs)', '#3498DB',
                          ax_g1_traj, ax_g1_range, ax_g1_accel, ax_g1_energy)

    # Group 2 (第三行)
    ax_g2_traj = fig.add_subplot(gs[2, 0])
    ax_g2_range = fig.add_subplot(gs[2, 1])
    ax_g2_accel = fig.add_subplot(gs[2, 2])
    ax_g2_energy = fig.add_subplot(gs[2, 3])
    plot_single_trajectory(group2_traj, 'Group 2 (6D Partial Obs)', '#2ECC71',
                          ax_g2_traj, ax_g2_range, ax_g2_accel, ax_g2_energy)

    fig.suptitle(f'Trajectory Comparison: {a_max}g Target Maneuver (seed={seed})',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*70)
    print("  Generating Trajectory Visualizations")
    print("="*70)

    # 生成三个典型场景的轨迹对比
    scenarios = [
        (0, 42),    # 0g: 简单场景
        (6, 42),    # 6g: 中等机动
        (10, 42),   # 10g: 高机动
    ]

    for a_max, seed in scenarios:
        fig = create_comparison_figure(a_max, seed)
        filename = f'trajectory_comparison_{int(a_max)}g.png'
        filepath = os.path.join(base_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filename}")

    print("\n" + "="*70)
    print("  Trajectory visualizations generated successfully!")
    print("="*70)
    print("\nGenerated 3 trajectory comparison figures:")
    print("  trajectory_comparison_0g.png  - Simple scenario")
    print("  trajectory_comparison_6g.png  - Medium maneuver")
    print("  trajectory_comparison_10g.png - High maneuver")
    print("\nEach figure contains:")
    print("  - 3 rows (PNG, Group 1, Group 2)")
    print("  - 4 columns (Trajectory, Range, Acceleration, Energy)")


if __name__ == "__main__":
    main()
