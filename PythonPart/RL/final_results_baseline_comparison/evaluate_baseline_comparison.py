"""
评估基线对比实验
================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from train_baseline_comparison import FullObsEnv, PartialObsEnv


def evaluate_model(model_path, env_class, model_name, n_trials=200):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Environment: {env_class.__name__}")
    print()

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    model = PPO.load(model_path)

    a_max_range = [0, 2, 4, 6, 8, 10]
    results = []

    for a_max_g in a_max_range:
        print(f"  Testing a_max = {a_max_g}g...", end=" ", flush=True)

        hits = 0
        miss_distances = []
        energies = []
        times = []

        for trial in range(n_trials):
            env = env_class(
                target_params={
                    'maneuver_type': 'random',
                    'a_max': a_max_g * 9.81
                }
            )

            obs, _ = env.reset(seed=trial)
            done = False
            energy = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 累积能量消耗
                am = env.env.state.am
                energy += (am ** 2) * env.env.cfg.decision_dt

            # 使用环境内部的命中判定（r <= 0.5m）
            hit = env.env.state.hit
            miss_dist = env.env.state.r if not hit else 0.0
            time_elapsed = env.env.state.t

            if hit:
                hits += 1
            miss_distances.append(miss_dist)
            energies.append(energy)
            times.append(time_elapsed)

        hit_rate = hits / n_trials * 100
        avg_miss = np.mean(miss_distances)
        avg_energy = np.mean(energies)
        avg_time = np.mean(times)

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

    # 计算总体和高机动命中率
    overall_hr = df['hit_rate'].mean()
    high_g_hr = df[df['a_max'] >= 8]['hit_rate'].mean()
    avg_energy = df['avg_energy'].mean()
    avg_time = df['avg_time'].mean()

    print()
    print(f"Overall Hit Rate:        {overall_hr:.1f}%")
    print(f"High-g Hit Rate (8-10g): {high_g_hr:.1f}%")
    print(f"Avg Energy:              {avg_energy:.1f} m^2/s^3")
    print(f"Avg Time:                {avg_time:.2f} s")

    return {
        'name': model_name,
        'overall_hr': overall_hr,
        'high_g_hr': high_g_hr,
        'avg_energy': avg_energy,
        'avg_time': avg_time,
        'details': df
    }


def main():
    print("=" * 70)
    print("  Baseline Comparison Evaluation")
    print("=" * 70)
    print("Both models trained with:")
    print("  - Original normalization: lam_dot/0.5 (verified 80%)")
    print("  - Original reward: ZEM-based")
    print("  - Original config: 4 scenarios, n_envs=4")
    print()

    base_dir = os.path.dirname(__file__)
    results = []

    # 评估Group 1（完整观测，8维）
    group1_path = os.path.join(base_dir, 'models/baseline_full_obs/best_model.zip')
    if os.path.exists(group1_path):
        r1 = evaluate_model(group1_path, FullObsEnv,
                           'Group 1: Full Obs (8D)', n_trials=200)
        if r1:
            results.append(r1)
    else:
        print(f"\nGroup 1 model not found at: {group1_path}")
        print("Train it with: python train_baseline_comparison.py --group group1")

    # 评估Group 2（部分观测，6维）
    group2_path = os.path.join(base_dir, 'models/baseline_partial_obs/best_model.zip')
    if os.path.exists(group2_path):
        r2 = evaluate_model(group2_path, PartialObsEnv,
                           'Group 2: Partial Obs (6D)', n_trials=200)
        if r2:
            results.append(r2)
    else:
        print(f"\nGroup 2 model not found at: {group2_path}")
        print("Train it with: python train_baseline_comparison.py --group group2")

    # 对比分析
    if len(results) >= 1:
        print("\n" + "=" * 70)
        print("  Comparison Summary")
        print("=" * 70)
        print(f"{'Method':<30} | {'Overall HR':>10} | {'High-g HR':>10} | {'Avg Energy':>12}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<30} | {r['overall_hr']:>9.1f}% | {r['high_g_hr']:>9.1f}% | {r['avg_energy']:>10.1f} m^2/s^3")

        # 如果两组都有结果，计算性能差距
        if len(results) == 2:
            print()
            print("Performance Gap:")
            hr_gap = results[0]['overall_hr'] - results[1]['overall_hr']
            print(f"  Overall Hit Rate: {hr_gap:+.1f} percentage points")
            print(f"  (Group 1 - Group 2)")
            print()

            if abs(hr_gap) < 10:
                print("Conclusion: Distance information (r, r_dot) provides minimal advantage.")
                print("  Partial observability is viable for this guidance problem.")
            elif abs(hr_gap) < 30:
                print("Conclusion: Distance information provides moderate advantage.")
                print("  Partial observability has performance tradeoff but remains functional.")
            else:
                print("Conclusion: Distance information is critical.")
                print("  Partial observability shows significant performance degradation.")

        # 保存结果
        output_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        for r in results:
            filename = r['name'].lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '') + '.csv'
            filepath = os.path.join(output_dir, filename)
            r['details'].to_csv(filepath, index=False)
            print(f"\nSaved: {filepath}")


if __name__ == "__main__":
    main()
