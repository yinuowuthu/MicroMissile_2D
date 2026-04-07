"""
3D PNG Monte-Carlo 仿真 + 可视化
=================================
对标 PythonPart/PNG/PNG.py，扩展到3D空间。
支持：单次轨迹3D绘图、批量Monte-Carlo统计。
"""

import sys
import time as _time
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List

from missile_env_3d import (
    MissileParams, TargetParams, SimConfig,
    MissileEngagement3D, run_episode,
)
from guidance_3d import ProportionalNavigation3D, AugmentedPN3D

G = 9.81


# ============================================================
#  单次仿真 + 3D轨迹绘图
# ============================================================

def plot_engagement_3d(traj: dict, title: str = "3D PNG Engagement"):
    """3D交战轨迹可视化（4子图）"""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{title}  |  Miss={traj['miss_distance']:.3f}m  "
        f"T={traj['flight_time']:.3f}s  {traj['reason']}",
        fontsize=13,
    )

    # 1) 3D轨迹
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(traj['xm'], traj['ym'], traj['zm'], 'b-', label='Missile')
    ax.plot(traj['xt'], traj['yt'], traj['zt'], 'r--', label='Target')
    ax.plot([traj['xm'][0]], [traj['ym'][0]], [traj['zm'][0]], 'bo', ms=6)
    ax.plot([traj['xt'][0]], [traj['yt'][0]], [traj['zt'][0]], 'ro', ms=6)
    ax.plot([traj['xm'][-1]], [traj['ym'][-1]], [traj['zm'][-1]], 'bx', ms=8)
    ax.plot([traj['xt'][-1]], [traj['yt'][-1]], [traj['zt'][-1]], 'rx', ms=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend(fontsize=8)

    # 2) 弹目距离
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(traj['t'], traj['r'], 'k-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('Missile-Target Range')
    ax2.grid(True, alpha=0.3)

    # 3) 加速度（两通道）
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(traj['t'], np.array(traj['ac_el']) / G, 'r-', alpha=0.6, label='cmd_el')
    ax3.plot(traj['t'], np.array(traj['ac_az']) / G, 'r--', alpha=0.6, label='cmd_az')
    ax3.plot(traj['t'], np.array(traj['am_el']) / G, 'b-', alpha=0.6, label='act_el')
    ax3.plot(traj['t'], np.array(traj['am_az']) / G, 'b--', alpha=0.6, label='act_az')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.set_title('Lateral Acceleration')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4) ZEM
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(traj['t'], traj['zem'], 'g-')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ZEM (m)')
    ax4.set_title('Zero Effort Miss')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
#  批量 Monte-Carlo（标量循环，简洁可靠）
# ============================================================

def run_monte_carlo(
    mp: MissileParams,
    tp: TargetParams,
    cfg: SimConfig,
    guidance,
    n_runs: int = 100,
    seed_offset: int = 0,
) -> dict:
    """
    运行n_runs次Monte-Carlo仿真，返回统计结果。
    """
    misses = np.zeros(n_runs)
    times = np.zeros(n_runs)
    energies = np.zeros(n_runs)
    hits = np.zeros(n_runs, dtype=bool)
    reasons = []

    env = MissileEngagement3D(missile=mp, target=tp, config=cfg)

    for i in range(n_runs):
        traj = run_episode(env, guidance, seed=seed_offset + i)
        misses[i] = traj['miss_distance']
        times[i] = traj['flight_time']
        energies[i] = traj['energy']
        hits[i] = traj['hit']
        reasons.append(traj['reason'])

    return {
        'misses': misses,
        'times': times,
        'energies': energies,
        'hits': hits,
        'reasons': reasons,
        'hit_rate': np.mean(hits),
    }


def run_batch(
    N_values: List[float] = [3, 4, 5],
    tau_values: List[float] = [0.1, 0.2, 0.5],
    maneuver_types: List[str] = ["none", "step", "sine"],
    maneuver_gs: List[float] = [5, 10, 15],
    n_runs: int = 50,
):
    """参数扫描批量仿真"""
    cfg = SimConfig()
    results = []

    all_cases = [
        (N, tau, man, mg)
        for N, tau, man, mg
        in itertools.product(N_values, tau_values, maneuver_types, maneuver_gs)
        if not (man == "none" and mg != maneuver_gs[0])
    ]

    for idx, (N, tau, man, mg) in enumerate(all_cases, 1):
        print(f"\r  Case {idx}/{len(all_cases)} ...", end="", flush=True)

        mp = MissileParams(a_max=20 * G, tau=tau)
        tp = TargetParams(maneuver_type=man, a_max=mg * G)
        guidance = ProportionalNavigation3D(N=N)

        stats = run_monte_carlo(mp, tp, cfg, guidance, n_runs=n_runs, seed_offset=idx * 1000)

        results.append({
            'N': N, 'tau': tau, 'maneuver': man,
            'max_g': mg if man != "none" else '-',
            'hit_rate': f"{int(np.sum(stats['hits']))}/{n_runs}",
            'miss_mean': f"{np.mean(stats['misses']):.3f}",
            'miss_std': f"{np.std(stats['misses']):.3f}",
            'miss_max': f"{np.max(stats['misses']):.3f}",
            'time_mean': f"{np.mean(stats['times']):.2f}",
            'energy_mean': f"{np.mean(stats['energies']):.1f}",
        })

    print(f"\r  Done ({len(all_cases)} cases).          ")
    return results


def print_table(table: List[dict]):
    if not table:
        return
    headers = list(table[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in table)) for h in headers}
    print(" | ".join(h.center(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in table:
        print(" | ".join(str(row[h]).rjust(widths[h]) for h in headers))


# ============================================================
#  Main
# ============================================================

def main():
    print("=" * 70)
    print("  3D PNG Simulation")
    print("=" * 70)

    mp = MissileParams()
    cfg = SimConfig()

    # --- 单次轨迹演示 ---
    demo_cases = [
        ("No maneuver",    TargetParams(maneuver_type="none")),
        ("Step 5g",        TargetParams(maneuver_type="step", a_max=5*G)),
        ("Sine 5g",        TargetParams(maneuver_type="sine", a_max=5*G)),
        ("Spiral 5g",      TargetParams(maneuver_type="spiral", a_max=5*G)),
    ]

    guidance = ProportionalNavigation3D(N=4)

    print("\n--- Single Engagement Demos ---")
    for title, tp in demo_cases:
        env = MissileEngagement3D(missile=mp, target=tp, config=cfg)
        traj = run_episode(env, guidance, seed=42)
        print(f"  {title}: miss={traj['miss_distance']:.3f}m, "
              f"time={traj['flight_time']:.3f}s, hit={traj['hit']}, {traj['reason']}")
        fig = plot_engagement_3d(traj, title)
        fname = f"engagement_3d_{tp.maneuver_type}.png"
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> saved {fname}")

    # --- Monte-Carlo ---
    print(f"\n--- Monte-Carlo (n=50 per case) ---")
    t0 = _time.perf_counter()
    table = run_batch(
        N_values=[3, 4, 5],
        tau_values=[0.1, 0.2, 0.5],
        maneuver_types=["none", "step", "sine", "spiral"],
        maneuver_gs=[1, 3, 5],
        n_runs=50,
    )
    elapsed = _time.perf_counter() - t0
    print(f"    Elapsed: {elapsed:.1f}s\n")
    print_table(table)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
