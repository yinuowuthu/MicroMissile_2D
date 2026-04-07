"""
PNG vs RL Baseline Comparison
==============================
Compare PNG baseline with Group 1 (8D) and Group 2 (6D) RL models.
Evaluate 4 key metrics: hit rate, miss distance, energy consumption, robustness.
"""

import os
import sys

# Add paths for imports
rl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
png_dir = os.path.join(os.path.dirname(rl_dir), 'PNG')
sys.path.insert(0, rl_dir)
sys.path.insert(0, png_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import PNG simulator
from PNG import run_batch_vectorized, MissileParams, TargetParams, SimConfig

# Import RL environments
from train_baseline_comparison import FullObsEnv, PartialObsEnv
from stable_baselines3 import PPO


def evaluate_png(n_trials=200, seed=42):
    """Evaluate PNG baseline across different maneuver intensities."""
    print("\n" + "="*70)
    print("  Evaluating PNG Baseline")
    print("="*70)

    mp = MissileParams(N=4.0, tau=0.2)
    cfg = SimConfig(dt=0.001, t_max=20.0, miss_dist=0.5)

    a_max_range = [0, 2, 4, 6, 8, 10]
    results = []

    rng = np.random.default_rng(seed)

    for a_max_g in a_max_range:
        print(f"  Testing a_max = {a_max_g}g...", end=" ", flush=True)

        # Generate initial conditions
        r0 = 3000.0 + rng.uniform(-200, 200, n_trials)
        lam0 = rng.uniform(np.deg2rad(-20), np.deg2rad(20), n_trials)
        gM0 = lam0 + rng.uniform(np.deg2rad(-5), np.deg2rad(5), n_trials)
        gT0 = np.pi + lam0 + rng.uniform(np.deg2rad(-30), np.deg2rad(30), n_trials)
        signs = rng.choice([-1.0, 1.0], n_trials)

        # Run PNG simulation
        if a_max_g == 0:
            tp = TargetParams(maneuver_type="none", maneuver_g=0)
        else:
            tp = TargetParams(maneuver_type="random", maneuver_g=a_max_g, maneuver_start=2.0)

        miss_dists, flight_times, energies, hits = run_batch_vectorized(
            mp, tp, cfg, r0, lam0, gM0, gT0, signs
        )

        hit_rate = np.sum(hits) / n_trials * 100
        avg_miss = np.mean(miss_dists)
        avg_energy = np.mean(energies)
        avg_time = np.mean(flight_times)

        results.append({
            'a_max': a_max_g,
            'hit_rate': hit_rate,
            'avg_miss_distance': avg_miss,
            'avg_energy': avg_energy,
            'avg_time': avg_time,
            'n_trials': n_trials
        })

        print(f"Hit rate: {hit_rate:.1f}%")

    df = pd.DataFrame(results)

    overall_hr = df['hit_rate'].mean()
    high_g_hr = df[df['a_max'] >= 8]['hit_rate'].mean()
    avg_energy = df['avg_energy'].mean()
    avg_time = df['avg_time'].mean()

    print()
    print(f"Overall Hit Rate:        {overall_hr:.1f}%")
    print(f"High-g Hit Rate (8-10g): {high_g_hr:.1f}%")
    print(f"Avg Energy:              {avg_energy:.1f} m^2/s^3")
    print(f"Avg Time:                {avg_time:.2f} s")

    return df


def load_rl_results(base_dir):
    """Load RL evaluation results."""
    group1_path = os.path.join(base_dir, 'group_1_full_obs_8d.csv')
    group2_path = os.path.join(base_dir, 'group_2_partial_obs_6d.csv')

    group1_df = pd.read_csv(group1_path) if os.path.exists(group1_path) else None
    group2_df = pd.read_csv(group2_path) if os.path.exists(group2_path) else None

    return group1_df, group2_df


def plot_comparison(png_df, group1_df, group2_df, output_dir):
    """Generate comprehensive comparison plots."""

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    a_max_vals = png_df['a_max'].values

    # --- Plot 1: Hit Rate Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(a_max_vals, png_df['hit_rate'], 'o-', label='PNG (N=4, τ=0.2)', linewidth=2, markersize=6)
    if group1_df is not None:
        ax1.plot(a_max_vals, group1_df['hit_rate'], 's-', label='Group 1 (8D RL)', linewidth=2, markersize=6)
    if group2_df is not None:
        ax1.plot(a_max_vals, group2_df['hit_rate'], '^-', label='Group 2 (6D RL)', linewidth=2, markersize=6)
    ax1.set_xlabel('Target Maneuver Intensity (g)')
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title('Hit Rate vs Maneuver Intensity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 105])

    # --- Plot 2: Miss Distance Comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(a_max_vals, png_df['avg_miss_distance'], 'o-', label='PNG', linewidth=2, markersize=6)
    if group1_df is not None:
        ax2.plot(a_max_vals, group1_df['avg_miss_distance'], 's-', label='Group 1 (8D RL)', linewidth=2, markersize=6)
    if group2_df is not None:
        ax2.plot(a_max_vals, group2_df['avg_miss_distance'], '^-', label='Group 2 (6D RL)', linewidth=2, markersize=6)
    ax2.set_xlabel('Target Maneuver Intensity (g)')
    ax2.set_ylabel('Average Miss Distance (m)')
    ax2.set_title('Miss Distance vs Maneuver Intensity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')

    # --- Plot 3: Energy Consumption Comparison ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(a_max_vals, png_df['avg_energy'], 'o-', label='PNG', linewidth=2, markersize=6)
    if group1_df is not None:
        ax3.plot(a_max_vals, group1_df['avg_energy'], 's-', label='Group 1 (8D RL)', linewidth=2, markersize=6)
    if group2_df is not None:
        ax3.plot(a_max_vals, group2_df['avg_energy'], '^-', label='Group 2 (6D RL)', linewidth=2, markersize=6)
    ax3.set_xlabel('Target Maneuver Intensity (g)')
    ax3.set_ylabel('Average Energy (m²/s³)')
    ax3.set_title('Energy Consumption vs Maneuver Intensity')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # --- Plot 4: Flight Time Comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(a_max_vals, png_df['avg_time'], 'o-', label='PNG', linewidth=2, markersize=6)
    if group1_df is not None:
        ax4.plot(a_max_vals, group1_df['avg_time'], 's-', label='Group 1 (8D RL)', linewidth=2, markersize=6)
    if group2_df is not None:
        ax4.plot(a_max_vals, group2_df['avg_time'], '^-', label='Group 2 (6D RL)', linewidth=2, markersize=6)
    ax4.set_xlabel('Target Maneuver Intensity (g)')
    ax4.set_ylabel('Average Flight Time (s)')
    ax4.set_title('Flight Time vs Maneuver Intensity')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # --- Plot 5: Overall Performance Summary (Bar Chart) ---
    ax5 = fig.add_subplot(gs[2, :])
    methods = ['PNG']
    overall_hrs = [png_df['hit_rate'].mean()]
    high_g_hrs = [png_df[png_df['a_max'] >= 8]['hit_rate'].mean()]

    if group1_df is not None:
        methods.append('Group 1\n(8D RL)')
        overall_hrs.append(group1_df['hit_rate'].mean())
        high_g_hrs.append(group1_df[group1_df['a_max'] >= 8]['hit_rate'].mean())

    if group2_df is not None:
        methods.append('Group 2\n(6D RL)')
        overall_hrs.append(group2_df['hit_rate'].mean())
        high_g_hrs.append(group2_df[group2_df['a_max'] >= 8]['hit_rate'].mean())

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax5.bar(x - width/2, overall_hrs, width, label='Overall (0-10g)', alpha=0.8)
    bars2 = ax5.bar(x + width/2, high_g_hrs, width, label='High-g (8-10g)', alpha=0.8)

    ax5.set_ylabel('Hit Rate (%)')
    ax5.set_title('Overall Performance Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 105])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.savefig(os.path.join(output_dir, 'png_vs_rl_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(output_dir, 'png_vs_rl_comparison.png')}")
    plt.close()


def generate_summary_table(png_df, group1_df, group2_df, output_dir):
    """Generate summary comparison table."""

    summary = []

    # PNG
    summary.append({
        'Method': 'PNG (N=4, τ=0.2)',
        'Overall Hit Rate (%)': png_df['hit_rate'].mean(),
        'High-g Hit Rate (%)': png_df[png_df['a_max'] >= 8]['hit_rate'].mean(),
        'Avg Energy (m²/s³)': png_df['avg_energy'].mean(),
        'Avg Time (s)': png_df['avg_time'].mean(),
    })

    # Group 1
    if group1_df is not None:
        summary.append({
            'Method': 'Group 1 (8D RL)',
            'Overall Hit Rate (%)': group1_df['hit_rate'].mean(),
            'High-g Hit Rate (%)': group1_df[group1_df['a_max'] >= 8]['hit_rate'].mean(),
            'Avg Energy (m²/s³)': group1_df['avg_energy'].mean(),
            'Avg Time (s)': group1_df['avg_time'].mean(),
        })

    # Group 2
    if group2_df is not None:
        summary.append({
            'Method': 'Group 2 (6D RL)',
            'Overall Hit Rate (%)': group2_df['hit_rate'].mean(),
            'High-g Hit Rate (%)': group2_df[group2_df['a_max'] >= 8]['hit_rate'].mean(),
            'Avg Energy (m²/s³)': group2_df['avg_energy'].mean(),
            'Avg Time (s)': group2_df['avg_time'].mean(),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

    print("\n" + "="*70)
    print("  Performance Summary")
    print("="*70)
    # Format output to avoid encoding issues
    for _, row in summary_df.iterrows():
        print(f"{row['Method']:<20} | Overall: {row['Overall Hit Rate (%)']:>6.1f}% | "
              f"High-g: {row['High-g Hit Rate (%)']:>6.1f}% | "
              f"Energy: {row['Avg Energy (m²/s³)']:>10.1f} | "
              f"Time: {row['Avg Time (s)']:>5.2f}s")
    print(f"\nSaved: {summary_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Evaluate PNG baseline
    png_df = evaluate_png(n_trials=200, seed=42)

    # Save PNG results
    png_path = os.path.join(base_dir, 'png_baseline.csv')
    png_df.to_csv(png_path, index=False)
    print(f"\nSaved PNG results: {png_path}")

    # Load RL results
    group1_df, group2_df = load_rl_results(base_dir)

    if group1_df is None or group2_df is None:
        print("\nWarning: RL results not found. Run evaluate_baseline_comparison.py first.")
        return

    # Generate comparison plots
    plot_comparison(png_df, group1_df, group2_df, base_dir)

    # Generate summary table
    generate_summary_table(png_df, group1_df, group2_df, base_dir)

    print("\n" + "="*70)
    print("  Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
