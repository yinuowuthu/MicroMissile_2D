"""
PPO训练脚本 — 6DOF导弹制导
============================
使用stable-baselines3的PPO训练MissileGym6DOF环境。
训练完成后与PNG基线对比。
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from missile_gym_env import MissileGym6DOF
from config import TargetParams, SimConfig
from seeker import SeekerParams
from engagement import Engagement


# ── 训练配置 ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
N_ENVS = 8
EVAL_FREQ = 20_000
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ppo_models')
LOG_DIR  = os.path.join(os.path.dirname(__file__), 'ppo_logs')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


def make_env(seed=0):
    def _init():
        tp  = TargetParams(maneuver_type='random', V0=30.0)
        cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5)
        sp  = SeekerParams()
        env = MissileGym6DOF(tp=tp, cfg=cfg, seeker_params=sp,
                             a_max_cmd=150.0,
                             hit_reward=100.0, miss_penalty=10.0,
                             seed=seed)
        return Monitor(env)
    return _init


def evaluate_policy(model, n_episodes=200, seed_offset=10000):
    """评估PPO策略命中率"""
    tp  = TargetParams(maneuver_type='random', V0=30.0)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5)
    sp  = SeekerParams()
    env = MissileGym6DOF(tp=tp, cfg=cfg, seeker_params=sp)

    hits = 0
    r_mins = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        r_mins.append(info['r_min'])
        if info['hit']:
            hits += 1

    r_mins = np.array(r_mins)
    return {
        'hits': hits,
        'n': n_episodes,
        'hit_rate': hits / n_episodes,
        'median_miss': float(np.median(r_mins)),
        'mean_miss': float(r_mins.mean()),
        'min_miss': float(r_mins.min()),
    }


def evaluate_png(n_episodes=200, seed_offset=10000):
    """评估PNG基线命中率（相同场景）"""
    tp  = TargetParams(maneuver_type='random', V0=30.0)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5)
    sp  = SeekerParams()
    eng = Engagement(tp=tp, cfg=cfg, seeker_params=sp)

    hits = 0
    r_mins = []
    for i in range(n_episodes):
        eng.reset(r0=100.0, seed=seed_offset + i)
        while not eng.done and eng.t < cfg.t_max:
            eng.step_guided()
        r_mins.append(eng.r_min)
        if eng.hit:
            hits += 1

    r_mins = np.array(r_mins)
    return {
        'hits': hits,
        'n': n_episodes,
        'hit_rate': hits / n_episodes,
        'median_miss': float(np.median(r_mins)),
        'mean_miss': float(r_mins.mean()),
        'min_miss': float(r_mins.min()),
    }


if __name__ == '__main__':
    print('=== PPO训练：6DOF导弹制导 ===')
    print(f'总步数: {TOTAL_TIMESTEPS:,}  并行环境: {N_ENVS}')
    print()

    # ── PNG基线 ──────────────────────────────────────────
    print('评估PNG基线...')
    png_res = evaluate_png(n_episodes=200)
    print(f'PNG基线: hits={png_res["hits"]}/200 ({png_res["hit_rate"]*100:.1f}%)'
          f'  median={png_res["median_miss"]:.2f}m')
    print()

    # ── 创建训练环境 ──────────────────────────────────────
    vec_env = make_vec_env(make_env(0), n_envs=N_ENVS)

    # 评估环境（无机动，干净场景）
    eval_env = Monitor(MissileGym6DOF(
        tp=TargetParams(maneuver_type='none', V0=30.0),
        cfg=SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5),
        seeker_params=SeekerParams(),
    ))

    # ── PPO模型 ───────────────────────────────────────────
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
    )

    # ── 回调 ─────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=SAVE_DIR,
        name_prefix='ppo_missile',
    )

    # ── 训练 ─────────────────────────────────────────────
    print('开始训练...')
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        progress_bar=False,
    )

    # ── 保存最终模型 ──────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, 'ppo_missile_final')
    model.save(final_path)
    print(f'\n模型已保存: {final_path}')

    # ── 最终对比 ──────────────────────────────────────────
    print('\n=== 最终对比 (200 episodes, random maneuver) ===')
    ppo_res = evaluate_policy(model, n_episodes=200)
    print(f'PPO : hits={ppo_res["hits"]}/200 ({ppo_res["hit_rate"]*100:.1f}%)'
          f'  median={ppo_res["median_miss"]:.2f}m  min={ppo_res["min_miss"]:.2f}m')
    print(f'PNG : hits={png_res["hits"]}/200 ({png_res["hit_rate"]*100:.1f}%)'
          f'  median={png_res["median_miss"]:.2f}m  min={png_res["min_miss"]:.2f}m')

    improvement = ppo_res["hit_rate"] - png_res["hit_rate"]
    print(f'\nPPO vs PNG 命中率提升: {improvement*100:+.1f}%')
