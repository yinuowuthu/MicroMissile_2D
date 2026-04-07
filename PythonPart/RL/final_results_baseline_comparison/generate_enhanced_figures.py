"""
生成增强版论文图表
================================
优化视觉效果，修复雷达图bug，添加更多细节
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
from matplotlib.patches import FancyBboxPatch

# 设置更精致的论文级别样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
})


def load_data(base_dir):
    """加载所有数据"""
    png_df = pd.read_csv(os.path.join(base_dir, 'png_baseline.csv'))
    group1_df = pd.read_csv(os.path.join(base_dir, 'group_1_full_obs_8d.csv'))
    group2_df = pd.read_csv(os.path.join(base_dir, 'group_2_partial_obs_6d.csv'))
    return png_df, group1_df, group2_df


def plot_hit_rate_enhanced(png_df, group1_df, group2_df, output_dir):
    """图1: 命中率对比（增强版）"""
    fig, ax = plt.subplots(figsize=(10, 7))

    a_max_vals = png_df['a_max'].values

    # 绘制曲线（带阴影）
    ax.plot(a_max_vals, png_df['hit_rate'], 'o-', label='PNG (N=4, τ=0.2)',
            linewidth=3, markersize=10, color='#E74C3C', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)
    ax.fill_between(a_max_vals, 0, png_df['hit_rate'], alpha=0.1, color='#E74C3C')

    ax.plot(a_max_vals, group1_df['hit_rate'], 's-', label='Group 1 (8D Full Obs)',
            linewidth=3, markersize=10, color='#3498DB', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)
    ax.fill_between(a_max_vals, 0, group1_df['hit_rate'], alpha=0.1, color='#3498DB')

    ax.plot(a_max_vals, group2_df['hit_rate'], '^-', label='Group 2 (6D Partial Obs)',
            linewidth=3, markersize=10, color='#2ECC71', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)
    ax.fill_between(a_max_vals, 0, group2_df['hit_rate'], alpha=0.1, color='#2ECC71')

    # 添加关键数据标注
    for i, val in enumerate(a_max_vals):
        if val in [0, 4, 8, 10]:  # 只标注关键点
            ax.text(val, png_df['hit_rate'].iloc[i] - 8, f"{png_df['hit_rate'].iloc[i]:.0f}%",
                   ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Hit Rate vs Target Maneuver Intensity', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    ax.set_ylim([0, 105])
    ax.set_xlim([-0.5, 10.5])
    ax.set_xticks(a_max_vals)

    # 添加背景色区分
    ax.axvspan(-0.5, 4, alpha=0.05, color='green', zorder=0)
    ax.axvspan(4, 6, alpha=0.05, color='yellow', zorder=0)
    ax.axvspan(6, 10.5, alpha=0.05, color='red', zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_hit_rate_enhanced.png'), dpi=300)
    plt.close()
    print(f"Saved: fig1_hit_rate_enhanced.png")


def plot_miss_distance_enhanced(png_df, group1_df, group2_df, output_dir):
    """图2: 脱靶量对比（增强版）"""
    fig, ax = plt.subplots(figsize=(10, 7))

    a_max_vals = png_df['a_max'].values

    ax.plot(a_max_vals, png_df['avg_miss_distance'], 'o-', label='PNG',
            linewidth=3, markersize=10, color='#E74C3C', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)
    ax.plot(a_max_vals, group1_df['avg_miss_distance'], 's-', label='Group 1 (8D)',
            linewidth=3, markersize=10, color='#3498DB', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)
    ax.plot(a_max_vals, group2_df['avg_miss_distance'], '^-', label='Group 2 (6D)',
            linewidth=3, markersize=10, color='#2ECC71', markeredgewidth=1.5,
            markeredgecolor='white', zorder=3)

    # 添加命中标准线（更醒目）
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
              label='Hit Threshold (0.5m)', zorder=2)
    ax.fill_between(a_max_vals, 0, 0.5, alpha=0.1, color='green', zorder=1)
    ax.fill_between(a_max_vals, 0.5, 1.0, alpha=0.1, color='red', zorder=1)

    ax.set_xlabel('Target Maneuver Intensity (g)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Miss Distance (m)', fontsize=14, fontweight='bold')
    ax.set_title('Miss Distance vs Target Maneuver Intensity', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    ax.set_yscale('log')
    ax.set_xlim([-0.5, 10.5])
    ax.set_xticks(a_max_vals)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_miss_distance_enhanced.png'), dpi=300)
    plt.close()
    print(f"Saved: fig2_miss_distance_enhanced.png")


def plot_radar_chart_fixed(png_df, group1_df, group2_df, output_dir):
    """图8: 雷达图（修复版）"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['Overall\nHit Rate', 'High-g\nHit Rate', 'Miss Distance\n(inverted)',
                  'Energy\n(inverted)', 'Robustness']

    # 计算指标（确保都在0-100范围内）
    def normalize_hit_rate(hr):
        return hr  # 已经是百分比

    def normalize_miss_distance(md, max_md=0.7):
        # 脱靶量越小越好，反转到0-100
        return max(0, 100 * (1 - md / max_md))

    def normalize_energy(e, max_e=25000):
        # 能量越小越好，反转到0-100
        return max(0, 100 * (1 - e / max_e))

    def normalize_robustness(std, max_std=40):
        # 标准差越小越好（鲁棒性越高），反转到0-100
        return max(0, 100 * (1 - std / max_std))

    # PNG
    png_overall_hr = normalize_hit_rate(png_df['hit_rate'].mean())
    png_highg_hr = normalize_hit_rate(png_df[png_df['a_max'] >= 8]['hit_rate'].mean())
    png_miss = normalize_miss_distance(png_df['avg_miss_distance'].mean())
    png_energy = normalize_energy(png_df['avg_energy'].mean())
    png_robust = normalize_robustness(png_df['hit_rate'].std())

    # Group 1
    g1_overall_hr = normalize_hit_rate(group1_df['hit_rate'].mean())
    g1_highg_hr = normalize_hit_rate(group1_df[group1_df['a_max'] >= 8]['hit_rate'].mean())
    g1_miss = normalize_miss_distance(group1_df['avg_miss_distance'].mean())
    g1_energy = normalize_energy(group1_df['avg_energy'].mean())
    g1_robust = normalize_robustness(group1_df['hit_rate'].std())

    # Group 2
    g2_overall_hr = normalize_hit_rate(group2_df['hit_rate'].mean())
    g2_highg_hr = normalize_hit_rate(group2_df[group2_df['a_max'] >= 8]['hit_rate'].mean())
    g2_miss = normalize_miss_distance(group2_df['avg_miss_distance'].mean())
    g2_energy = normalize_energy(group2_df['avg_energy'].mean())
    g2_robust = normalize_robustness(group2_df['hit_rate'].std())

    values_png = [png_overall_hr, png_highg_hr, png_miss, png_energy, png_robust]
    values_g1 = [g1_overall_hr, g1_highg_hr, g1_miss, g1_energy, g1_robust]
    values_g2 = [g2_overall_hr, g2_highg_hr, g2_miss, g2_energy, g2_robust]

    # 闭合多边形
    values_png += values_png[:1]
    values_g1 += values_g1[:1]
    values_g2 += values_g2[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # 绘制（增强视觉效果）
    ax.plot(angles, values_png, 'o-', linewidth=3, label='PNG', color='#E74C3C',
           markersize=10, markeredgewidth=2, markeredgecolor='white')
    ax.fill(angles, values_png, alpha=0.2, color='#E74C3C')

    ax.plot(angles, values_g1, 's-', linewidth=3, label='Group 1 (8D)', color='#3498DB',
           markersize=10, markeredgewidth=2, markeredgecolor='white')
    ax.fill(angles, values_g1, alpha=0.2, color='#3498DB')

    ax.plot(angles, values_g2, '^-', linewidth=3, label='Group 2 (6D)', color='#2ECC71',
           markersize=10, markeredgewidth=2, markeredgecolor='white')
    ax.fill(angles, values_g2, alpha=0.2, color='#2ECC71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)

    # 添加同心圆背景
    for r in [20, 40, 60, 80, 100]:
        ax.plot(angles, [r]*len(angles), 'k-', alpha=0.1, linewidth=0.5)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), framealpha=0.95,
             edgecolor='gray', fancybox=True, shadow=True, fontsize=12)
    ax.set_title('Multi-Dimensional Performance Comparison\n(All metrics normalized: higher is better)',
                fontsize=16, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_radar_chart_fixed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig8_radar_chart_fixed.png")


def plot_overall_performance_enhanced(png_df, group1_df, group2_df, output_dir):
    """图5: 总体性能对比（增强版）"""
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['PNG\n(N=4, τ=0.2)', 'Group 1\n(8D Full Obs)', 'Group 2\n(6D Partial Obs)']
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
    width = 0.38

    # 使用渐变色效果
    bars1 = ax.bar(x - width/2, overall_hrs, width, label='Overall (0-10g)',
                   alpha=0.9, color='#3498DB', edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, high_g_hrs, width, label='High-g (8-10g)',
                   alpha=0.9, color='#E74C3C', edgecolor='white', linewidth=2)

    ax.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.legend(framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 105])

    # 添加数值标签（更醒目）
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加背景色
    ax.axhspan(0, 50, alpha=0.05, color='red', zorder=0)
    ax.axhspan(50, 80, alpha=0.05, color='yellow', zorder=0)
    ax.axhspan(80, 105, alpha=0.05, color='green', zorder=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_overall_performance_enhanced.png'), dpi=300)
    plt.close()
    print(f"Saved: fig5_overall_performance_enhanced.png")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*70)
    print("  Generating Enhanced Publication-Quality Figures")
    print("="*70)

    # 加载数据
    png_df, group1_df, group2_df = load_data(base_dir)

    # 生成增强版图表
    plot_hit_rate_enhanced(png_df, group1_df, group2_df, base_dir)
    plot_miss_distance_enhanced(png_df, group1_df, group2_df, base_dir)
    plot_radar_chart_fixed(png_df, group1_df, group2_df, base_dir)
    plot_overall_performance_enhanced(png_df, group1_df, group2_df, base_dir)

    print("\n" + "="*70)
    print("  Enhanced figures generated successfully!")
    print("="*70)
    print("\nGenerated 4 enhanced figures:")
    print("  fig1_hit_rate_enhanced.png")
    print("  fig2_miss_distance_enhanced.png")
    print("  fig5_overall_performance_enhanced.png")
    print("  fig8_radar_chart_fixed.png")
    print("\n主要改进:")
    print("  - 修复雷达图归一化bug")
    print("  - 增加线条粗细和标记大小")
    print("  - 添加阴影和渐变效果")
    print("  - 优化网格和背景色")
    print("  - 增强数据标注")


if __name__ == "__main__":
    main()

