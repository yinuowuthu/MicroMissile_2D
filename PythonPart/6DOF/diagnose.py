"""
PNG诊断脚本
===========
诊断PNG在none机动下命中率异常低的原因。
检查：miss distance分布、初始条件影响、boost阶段分析。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import Counter
from config import TargetParams, SimConfig, G
from engagement import Engagement
from seeker import SeekerParams


def diagnose_png(n_episodes=1000, maneuver='none', r0_fixed=100.0, seed_offset=0):
    tp  = TargetParams(maneuver_type=maneuver, V0=30.0)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5)
    sp  = SeekerParams()
    eng = Engagement(tp=tp, cfg=cfg, seeker_params=sp)

    results = []
    boost_stats = []  # boost结束时的状态

    for i in range(n_episodes):
        rng = np.random.RandomState(seed_offset + i)
        eng.reset(r0=r0_fixed, seed=seed_offset + i)

        # 记录初始条件
        r0_actual = eng.get_range()
        # 从engagement内部重建az0/el0（通过目标位置反推）
        t_pos = eng.target.pos
        az0 = np.arctan2(t_pos[1], t_pos[0])
        el0 = np.arctan2(-t_pos[2], np.sqrt(t_pos[0]**2 + t_pos[1]**2))

        boost_recorded = False
        boost_t = None
        boost_los_err = None
        boost_los_rate = None
        boost_speed = None

        while not eng.done and eng.t < cfg.t_max:
            # 记录boost结束时刻（V刚超过50 m/s）
            if not boost_recorded and eng.missile.speed >= 50.0:
                boost_recorded = True
                boost_t = eng.t
                boost_speed = eng.missile.speed
                r_los = eng.get_los()
                r = np.linalg.norm(r_los)
                # LOS角误差（相对导弹体轴）
                los_az = np.arctan2(r_los[1], r_los[0])
                los_el = np.arctan2(-r_los[2], np.sqrt(r_los[0]**2 + r_los[1]**2))
                m_az = eng.missile.state[8]
                m_el = eng.missile.state[7]
                boost_los_err = np.degrees(np.sqrt(
                    (los_az - m_az)**2 + (los_el - m_el)**2
                ))
                sd = eng._seeker_data
                boost_los_rate = np.sqrt(
                    sd.get('los_rate_el', 0)**2 + sd.get('los_rate_az', 0)**2
                )

            eng.step_guided()

        results.append({
            'r0': r0_actual,
            'az0_deg': np.degrees(az0),
            'el0_deg': np.degrees(el0),
            'r_min': eng.r_min,
            'hit': eng.hit,
            'reason': eng.reason,
            't': eng.t,
        })
        boost_stats.append({
            'boost_t': boost_t,
            'boost_speed': boost_speed,
            'boost_los_err': boost_los_err,
            'boost_los_rate': boost_los_rate,
        })

    # ── 汇总 ──────────────────────────────────────────────────
    hits   = [r for r in results if r['hit']]
    misses = [r for r in results if not r['hit']]
    hit_rate = len(hits) / n_episodes

    print(f"\n=== PNG诊断 ({n_episodes} episodes, {maneuver}机动, r0={r0_fixed}m) ===\n")
    print(f"命中率: {hit_rate*100:.1f}% ({len(hits)}/{n_episodes})\n")

    # 脱靶原因
    reasons = Counter(r['reason'] for r in misses)
    print("脱靶原因:")
    for reason, count in reasons.most_common():
        print(f"  {reason:20s}: {count:4d} ({count/len(misses)*100:.1f}%)")

    # miss distance分布
    miss_rmins = np.array([r['r_min'] for r in misses])
    print(f"\n脱靶miss distance分布 (n={len(misses)}):")
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
    for lo, hi in bins:
        count = ((miss_rmins >= lo) & (miss_rmins < hi)).sum()
        label = f"<{hi}m" if hi < 1e9 else f">{lo}m"
        bar = '█' * int(count / n_episodes * 200)
        print(f"  {lo:5.1f}-{hi:5.1f}m: {count:4d} ({count/n_episodes*100:4.1f}%)  {bar}")
    print(f"  median={np.median(miss_rmins):.2f}m  "
          f"p75={np.percentile(miss_rmins,75):.2f}m  "
          f"p90={np.percentile(miss_rmins,90):.2f}m  "
          f"max={miss_rmins.max():.2f}m")

    # 按初始方位角分组
    print(f"\n按初始方位角分组命中率:")
    az_bins = [(-20, -10), (-10, 0), (0, 10), (10, 20)]
    for lo, hi in az_bins:
        group = [r for r in results if lo <= r['az0_deg'] < hi]
        if group:
            hr = sum(r['hit'] for r in group) / len(group)
            print(f"  az0 ∈ [{lo:+3d}°,{hi:+3d}°]: {len(group):3d} eps, 命中率={hr*100:.1f}%")

    # boost阶段分析
    valid_boost = [b for b in boost_stats if b['boost_los_err'] is not None]
    if valid_boost:
        los_errs  = np.array([b['boost_los_err']  for b in valid_boost])
        los_rates = np.array([b['boost_los_rate'] for b in valid_boost])
        speeds    = np.array([b['boost_speed']     for b in valid_boost])
        boost_ts  = np.array([b['boost_t']         for b in valid_boost])
        print(f"\nboost阶段分析 (V≥50 m/s时刻, n={len(valid_boost)}):")
        print(f"  boost结束时间:    median={np.median(boost_ts):.3f}s")
        print(f"  boost结束速度:    median={np.median(speeds):.1f} m/s")
        print(f"  LOS角误差:        median={np.median(los_errs):.1f}°  "
              f"p90={np.percentile(los_errs,90):.1f}°")
        print(f"  LOS角速率:        median={np.median(los_rates):.3f} rad/s  "
              f"p90={np.percentile(los_rates,90):.3f} rad/s")

        # 估算所需vs可用加速度
        V_med = np.median(speeds)
        from config import RHO, MissileParams
        mp = MissileParams()
        qbar = 0.5 * RHO * V_med**2
        mass_after_boost = mp.m0 - mp.m_propellant
        a_max = qbar * mp.S_ref * mp.CNa * np.radians(12) / mass_after_boost
        los_rate_med = np.median(los_rates)
        a_required = 4.0 * V_med * los_rate_med
        print(f"\n  导弹可用侧向加速度 (V={V_med:.0f} m/s): {a_max:.1f} m/s²")
        print(f"  PNG所需加速度 (N=4, Vc={V_med:.0f}, λ̇={los_rate_med:.3f}): {a_required:.1f} m/s²")
        ratio = a_required / max(a_max, 1)
        print(f"  需求/可用比: {ratio:.2f}x  {'⚠ 饱和' if ratio > 1 else '✓ 充足'}")

    # r_hit敏感性分析
    all_rmins = np.array([r['r_min'] for r in results])
    print(f"\nr_hit敏感性分析（不同命中阈值下的命中率）:")
    for r_hit in [0.5, 1.0, 2.0, 3.0, 5.0]:
        hr = (all_rmins <= r_hit).mean()
        print(f"  r_hit={r_hit:.1f}m: {hr*100:.1f}%")


if __name__ == '__main__':
    # 主诊断：none机动，r0=100m（与gym env一致）
    diagnose_png(n_episodes=1000, maneuver='none', r0_fixed=100.0)

    # 对比：random机动
    print("\n" + "="*60)
    diagnose_png(n_episodes=500, maneuver='random', r0_fixed=100.0)
