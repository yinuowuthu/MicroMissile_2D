"""
生成论文级别的高质量图表
================================
将对比图分开保存，并添加更多高级可视化
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

# 设置论文级别的样式
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_data(base_dir):
    """加载所有数据"""
    png_df = pd.read_csv(os.path.join(base_dir, 'png_baseline.csv'))
    group1_df = pd.read_csv(os.path.join(base_dir, 'group_1_full_obs_8d.csv'))
    group2_df = pd.read_csv(os.path.join(base_dir, 'group_2_partial_obs_6d.csv'))
    return png_df, group1_df, group2_df


def plot_hit_rate(png_df, group1_df, group2_df, output_dir):
    """图1: 命中率对比"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    ax.plot(a_max_vals, png_df['hit_rate'], 'o-', label='PNG (N=4, τ=0.2)',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, group1_df['hit_rate'], 's-', label='Group 1 (8D Full Obs)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, group2_df['hit_rate'], '^-', label='Group 2 (6D Partial Obs)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Hit Rate (%)', fontsize=13)
    ax.set_title('Hit Rate vs Target Maneuver Intensity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim([0, 105])
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_hit_rate_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: fig1_hit_rate_comparison.png")


def plot_miss_distance(png_df, group1_df, group2_df, output_dir):
    """图2: 脱靶量对比（对数坐标）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    ax.plot(a_max_vals, png_df['avg_miss_distance'], 'o-', label='PNG',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, group1_df['avg_miss_distance'], 's-', label='Group 1 (8D)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, group2_df['avg_miss_distance'], '^-', label='Group 2 (6D)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    # 添加命中标准线
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Hit Threshold (0.5m)')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Average Miss Distance (m)', fontsize=13)
    ax.set_title('Miss Distance vs Target Maneuver Intensity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_yscale('log')
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_miss_distance_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: fig2_miss_distance_comparison.png")


def plot_energy_consumption(png_df, group1_df, group2_df, output_dir):
    """图3: 能量消耗对比"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    ax.plot(a_max_vals, png_df['avg_energy'], 'o-', label='PNG',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, group1_df['avg_energy'], 's-', label='Group 1 (8D)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, group2_df['avg_energy'], '^-', label='Group 2 (6D)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Average Energy Consumption (m²/s³)', fontsize=13)
    ax.set_title('Energy Consumption vs Target Maneuver Intensity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_energy_consumption_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: fig3_energy_consumption_comparison.png")


def plot_flight_time(png_df, group1_df, group2_df, output_dir):
    """图4: 飞行时间对比"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    ax.plot(a_max_vals, png_df['avg_time'], 'o-', label='PNG',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, group1_df['avg_time'], 's-', label='Group 1 (8D)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, group2_df['avg_time'], '^-', label='Group 2 (6D)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Average Flight Time (s)', fontsize=13)
    ax.set_title('Flight Time vs Target Maneuver Intensity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_flight_time_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: fig4_flight_time_comparison.png")


def plot_overall_performance(png_df, group1_df, group2_df, output_dir):
    """图5: 总体性能对比（柱状图）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['PNG', 'Group 1\n(8D RL)', 'Group 2\n(6D RL)']
    overall_hrs = [
        png_df['hit_rate'].mean(),
        group1_df['hit_rate'].mean(),
        group2_df['hit_rate'].mean()
    ]
    high_g_hrs = [
        png_df[png_df['a_max'] >= 8]['hit_rate'].mean(),
        group1_df[group1_df['a_max'] >= 8]['hit_rate'].mean(),
        group2_df[group2_df['a_max'] >= 8]['hit_rate'].mean()
    ]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, overall_hrs, width, label='Overall (0-10g)',
                   alpha=0.8, color='#3498DB')
    bars2 = ax.bar(x + width/2, high_g_hrs, width, label='High-g (8-10g)',
                   alpha=0.8, color='#E74C3C')

    ax.set_ylabel('Hit Rate (%)', fontsize=13)
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 105])

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_overall_performance_bar.png'), dpi=300)
    plt.close()
    print(f"Saved: fig5_overall_performance_bar.png")


def plot_robustness_analysis(png_df, group1_df, group2_df, output_dir):
    """图6: 鲁棒性分析（性能下降曲线）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    # 计算相对于0g场景的性能下降
    png_degradation = 100 - (png_df['hit_rate'].values / png_df['hit_rate'].values[0] * 100)
    g1_degradation = 100 - (group1_df['hit_rate'].values / group1_df['hit_rate'].values[0] * 100)
    g2_degradation = 100 - (group2_df['hit_rate'].values / group2_df['hit_rate'].values[0] * 100)

    ax.plot(a_max_vals, png_degradation, 'o-', label='PNG',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, g1_degradation, 's-', label='Group 1 (8D)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, g2_degradation, '^-', label='Group 2 (6D)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Performance Degradation (%)', fontsize=13)
    ax.set_title('Robustness Analysis: Performance Degradation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_robustness_analysis.png'), dpi=300)
    plt.close()
    print(f"Saved: fig6_robustness_analysis.png")


def plot_energy_efficiency(png_df, group1_df, group2_df, output_dir):
    """图7: 能量效率分析（能量/命中率）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_max_vals = png_df['a_max'].values

    # 计算能量效率（能量消耗/命中率，越低越好）
    png_eff = png_df['avg_energy'].values / (png_df['hit_rate'].values + 1e-6)
    g1_eff = group1_df['avg_energy'].values / (group1_df['hit_rate'].values + 1e-6)
    g2_eff = group2_df['avg_energy'].values / (group2_df['hit_rate'].values + 1e-6)

    ax.plot(a_max_vals, png_eff, 'o-', label='PNG',
            linewidth=2.5, markersize=8, color='#E74C3C')
    ax.plot(a_max_vals, g1_eff, 's-', label='Group 1 (8D)',
            linewidth=2.5, markersize=8, color='#3498DB')
    ax.plot(a_max_vals, g2_eff, '^-', label='Group 2 (6D)',
            linewidth=2.5, markersize=8, color='#2ECC71')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=13)
    ax.set_ylabel('Energy per Hit Rate (m²/s³ per %)', fontsize=13)
    ax.set_title('Energy Efficiency Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(a_max_vals)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_energy_efficiency.png'), dpi=300)
    plt.close()
    print(f"Saved: fig7_energy_efficiency.png")


def plot_radar_chart(png_df, group1_df, group2_df, output_dir):
    """图8: 雷达图（多维性能对比）"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # 定义指标（归一化到0-100）
    categories = ['Hit Rate\n(Overall)', 'Hit Rate\n(High-g)', 'Miss Distance\n(Lower Better)',
                  'Energy\n(Lower Better)', 'Robustness']

    # PNG
    png_overall_hr = png_df['hit_rate'].mean()
    png_highg_hr = png_df[png_df['a_max'] >= 8]['hit_rate'].mean()
    png_miss = 100 - (png_df['avg_miss_distance'].mean() / 0.7 * 100)  # 归一化
    png_energy = 100 - (png_df['avg_energy'].mean() / 25000 * 100)  # 归一化
    png_robust = 100 - (png_df['hit_rate'].std() / 40 * 100)  # 归一化

    # Group 1
    g1_overall_hr = group1_df['hit_rate'].mean()
    g1_highg_hr = group1_df[group1_df['a_max'] >= 8]['hit_rate'].mean()
    g1_miss = 100 - (group1_df['avg_miss_distance'].mean() / 0.7 * 100)
    g1_energy = 100 - (group1_df['avg_energy'].mean() / 25000 * 100)
    g1_robust = 100 - (group1_df['hit_rate'].std() / 40 * 100)

    # Group 2
    g2_overall_hr = group2_df['hit_rate'].mean()
    g2_highg_hr = group2_df[group2_df['a_max'] >= 8]['hit_rate'].mean()
    g2_miss = 100 - (group2_df['avg_miss_distance'].mean() / 0.7 * 100)
    g2_energy = 100 - (group2_df['avg_energy'].mean() / 25000 * 100)
    g2_robust = 100 - (group2_df['hit_rate'].std() / 40 * 100)

    values_png = [png_overall_hr, png_highg_hr, png_miss, png_energy, png_robust]
    values_g1 = [g1_overall_hr, g1_highg_hr, g1_miss, g1_energy, g1_robust]
    values_g2 = [g2_overall_hr, g2_highg_hr, g2_miss, g2_energy, g2_robust]

    # 闭合多边形
    values_png += values_png[:1]
    values_g1 += values_g1[:1]
    values_g2 += values_g2[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, values_png, 'o-', linewidth=2, label='PNG', color='#E74C3C')
    ax.fill(angles, values_png, alpha=0.15, color='#E74C3C')

    ax.plot(angles, values_g1, 's-', linewidth=2, label='Group 1 (8D)', color='#3498DB')
    ax.fill(angles, values_g1, alpha=0.15, color='#3498DB')

    ax.plot(angles, values_g2, '^-', linewidth=2, label='Group 2 (6D)', color='#2ECC71')
    ax.fill(angles, values_g2, alpha=0.15, color='#2ECC71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    ax.set_title('Multi-Dimensional Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig8_radar_chart.png")


def plot_heatmap(png_df, group1_df, group2_df, output_dir):
    """图9: 热力图（性能矩阵）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    a_max_vals = png_df['a_max'].values
    metrics = ['Hit Rate', 'Miss Dist', 'Energy', 'Time']

    # 准备数据矩阵
    png_matrix = np.array([
        png_df['hit_rate'].values,
        png_df['avg_miss_distance'].values,
        png_df['avg_energy'].values / 1000,  # 缩放
        png_df['avg_time'].values
    ])

    g1_matrix = np.array([
        group1_df['hit_rate'].values,
        group1_df['avg_miss_distance'].values,
        group1_df['avg_energy'].values / 1000,
        group1_df['avg_time'].values
    ])

    g2_matrix = np.array([
        group2_df['hit_rate'].values,
        group2_df['avg_miss_distance'].values,
        group2_df['avg_energy'].values / 1000,
        group2_df['avg_time'].values
    ])

    # PNG
    im0 = axes[0].imshow(png_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(a_max_vals)))
    axes[0].set_xticklabels([f'{int(x)}g' for x in a_max_vals])
    axes[0].set_yticks(range(len(metrics)))
    axes[0].set_yticklabels(metrics)
    axes[0].set_title('PNG Performance Matrix', fontweight='bold')
    for i in range(len(metrics)):
        for j in range(len(a_max_vals)):
            axes[0].text(j, i, f'{png_matrix[i, j]:.1f}',
                        ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im0, ax=axes[0])

    # Group 1
    im1 = axes[1].imshow(g1_matrix, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(a_max_vals)))
    axes[1].set_xticklabels([f'{int(x)}g' for x in a_max_vals])
    axes[1].set_yticks(range(len(metrics)))
    axes[1].set_yticklabels(metrics)
    axes[1].set_title('Group 1 (8D) Performance Matrix', fontweight='bold')
    for i in range(len(metrics)):
        for j in range(len(a_max_vals)):
            axes[1].text(j, i, f'{g1_matrix[i, j]:.1f}',
                        ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im1, ax=axes[1])

    # Group 2
    im2 = axes[2].imshow(g2_matrix, cmap='Greens', aspect='auto')
    axes[2].set_xticks(range(len(a_max_vals)))
    axes[2].set_xticklabels([f'{int(x)}g' for x in a_max_vals])
    axes[2].set_yticks(range(len(metrics)))
    axes[2].set_yticklabels(metrics)
    axes[2].set_title('Group 2 (6D) Performance Matrix', fontweight='bold')
    for i in range(len(metrics)):
        for j in range(len(a_max_vals)):
            axes[2].text(j, i, f'{g2_matrix[i, j]:.1f}',
                        ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig9_performance_heatmap.png'), dpi=300)
    plt.close()
    print(f"Saved: fig9_performance_heatmap.png")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*70)
    print("  Generating Publication-Quality Figures")
    print("="*70)

    # 加载数据
    png_df, group1_df, group2_df = load_data(base_dir)

    # 生成所有图表
    plot_hit_rate(png_df, group1_df, group2_df, base_dir)
    plot_miss_distance(png_df, group1_df, group2_df, base_dir)
    plot_energy_consumption(png_df, group1_df, group2_df, base_dir)
    plot_flight_time(png_df, group1_df, group2_df, base_dir)
    plot_overall_performance(png_df, group1_df, group2_df, base_dir)
    plot_robustness_analysis(png_df, group1_df, group2_df, base_dir)
    plot_energy_efficiency(png_df, group1_df, group2_df, base_dir)
    plot_radar_chart(png_df, group1_df, group2_df, base_dir)
    plot_heatmap(png_df, group1_df, group2_df, base_dir)

    print("\n" + "="*70)
    print("  All figures generated successfully!")
    print("="*70)
    print("\nGenerated 9 publication-quality figures:")
    print("  fig1_hit_rate_comparison.png")
    print("  fig2_miss_distance_comparison.png")
    print("  fig3_energy_consumption_comparison.png")
    print("  fig4_flight_time_comparison.png")
    print("  fig5_overall_performance_bar.png")
    print("  fig6_robustness_analysis.png")
    print("  fig7_energy_efficiency.png")
    print("  fig8_radar_chart.png")
    print("  fig9_performance_heatmap.png")


if __name__ == "__main__":
    main()
