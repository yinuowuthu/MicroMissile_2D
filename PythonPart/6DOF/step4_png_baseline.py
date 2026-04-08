"""
第四步：PNG基线 Monte Carlo (n=200)
=====================================
场景：初始距离30-300m，目标速度30m/s，多种机动类型
输出：Miss Distance CDF、箱线图、命中率表
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import TargetParams, SimConfig, G
from engagement import Engagement
from seeker import SeekerParams

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 130


def run_mc(n=200, maneuver='none', noise=True, N=4, seed_offset=0):
    """运行一组 Monte Carlo"""
    sp = SeekerParams() if noise else SeekerParams(
        sigma_thermal=0.0, target_size=0.0, sigma_glint_min=0.0)
    tp = TargetParams(maneuver_type=maneuver, V0=30.0, a_max=10*G)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5,
                    r_init_min=30.0, r_init_max=300.0)
    eng = Engagement(tp=tp, cfg=cfg, seeker_params=sp)
    eng.guidance.N = N

    r_mins, hits, reasons, times = [], [], [], []
    for seed in range(n):
        eng.reset(seed=seed + seed_offset)
        while not eng.done and eng.t < cfg.t_max:
            eng.step_guided()
        r_mins.append(eng.r_min)
        hits.append(eng.reason == 'HIT')
        reasons.append(eng.reason)
        times.append(eng.t)

    return {
        'r_min': np.array(r_mins),
        'hit_rate': np.mean(hits),
        'hits': sum(hits),
        'n': n,
        'reasons': reasons,
        'times': np.array(times),
    }


def print_summary(label, res):
    r = res['r_min']
    print(f"  {label:20s}  hits={res['hits']:3d}/{res['n']}  "
          f"({res['hit_rate']*100:4.1f}%)  "
          f"median={np.median(r):.2f}m  "
          f"mean={r.mean():.2f}m  "
          f"p90={np.percentile(r,90):.2f}m")


if __name__ == '__main__':
    N_MC = 200
    print("=" * 70)
    print("  第四步：PNG基线 Monte Carlo (n=200)")
    print("=" * 70)

    # ---- 场景矩阵 ----
    scenarios = [
        ('无机动_无噪声',  'none',   False),
        ('无机动_有噪声',  'none',   True),
        ('正弦机动_有噪声', 'sine',   True),
        ('阶跃机动_有噪声', 'step',   True),
        ('螺旋机动_有噪声', 'spiral', True),
        ('随机机动_有噪声', 'random', True),
    ]

    results = {}
    print(f"\n{'场景':22s}  命中率      中位数   均值    P90")
    print("-" * 70)
    for label, maneuver, noise in scenarios:
        res = run_mc(N_MC, maneuver=maneuver, noise=noise)
        results[label] = res
        print_summary(label, res)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PNG Baseline Monte Carlo (N=4, n=200 per scenario)', fontsize=13)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # CDF
    ax = axes[0]
    for (label, _, _), color in zip(scenarios, colors):
        r = results[label]['r_min']
        sorted_r = np.sort(r)
        cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
        ax.plot(sorted_r, cdf, label=label, color=color, linewidth=1.8)
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.4, label='r_hit=0.5m')
    ax.set_xlabel('Miss Distance (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Miss Distance CDF')
    ax.set_xlim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 箱线图
    ax = axes[1]
    data = [results[label]['r_min'] for label, _, _ in scenarios]
    labels_short = ['无机动\n无噪声', '无机动\n有噪声', '正弦\n有噪声',
                    '阶跃\n有噪声', '螺旋\n有噪声', '随机\n有噪声']
    bp = ax.boxplot(data, labels=labels_short, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Miss Distance (m)')
    ax.set_title('Miss Distance Distribution')
    ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'step4_png_baseline.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  图表已保存: step4_png_baseline.png")

    # ---- 命中率汇总表 ----
    print("\n" + "=" * 70)
    print("  PNG基线命中率汇总（r_hit=0.5m）")
    print("=" * 70)
    for label, _, _ in scenarios:
        res = results[label]
        r = res['r_min']
        print(f"  {label:22s}  {res['hit_rate']*100:5.1f}%  "
              f"[<1m: {(r<1).mean()*100:.0f}%  <2m: {(r<2).mean()*100:.0f}%  <5m: {(r<5).mean()*100:.0f}%]")

    print("\n  第四步完成。PNG基线已建立。")
