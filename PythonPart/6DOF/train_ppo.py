"""
PPO微调脚本 — 6DOF导弹制导 v3
============================
在BC预训练基础上进行PPO微调：
  - 加载BC预训练权重（策略+价值函数）
  - 三阶段课程：none → none+sine → random
  - 总步数5M，学习率1e-4
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from missile_gym_env import MissileGym6DOF
from config import TargetParams, SimConfig
from seeker import SeekerParams
from engagement import Engagement


SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ppo_models')
LOG_DIR  = os.path.join(os.path.dirname(__file__), 'ppo_logs')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

N_ENVS     = 8
EVAL_FREQ  = 50_000


def make_env(seed=0, maneuver_type='none'):
    def _init():
        tp  = TargetParams(maneuver_type=maneuver_type, V0=30.0)
        cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=1.0)
        sp  = SeekerParams()
        env = MissileGym6DOF(tp=tp, cfg=cfg, seeker_params=sp,
                             a_max_cmd=150.0,
                             hit_reward=100.0, miss_penalty=10.0,
                             seed=seed)
        return Monitor(env)
    return _init


def evaluate_policy(model, n_episodes=200, seed_offset=10000):
    tp  = TargetParams(maneuver_type='random', V0=30.0)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=1.0)
    sp  = SeekerParams()
    env = MissileGym6DOF(tp=tp, cfg=cfg, seeker_params=sp)

    hits, r_mins = 0, []
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
        'hits': hits, 'n': n_episodes,
        'hit_rate': hits / n_episodes,
        'median_miss': float(np.median(r_mins)),
        'min_miss': float(r_mins.min()),
    }


def evaluate_png(n_episodes=200, seed_offset=10000):
    tp  = TargetParams(maneuver_type='random', V0=30.0)
    cfg = SimConfig(dt=0.0005, t_max=3.0, r_hit=1.0)
    sp  = SeekerParams()
    eng = Engagement(tp=tp, cfg=cfg, seeker_params=sp)

    hits, r_mins = 0, []
    for i in range(n_episodes):
        eng.reset(r0=100.0, seed=seed_offset + i)
        while not eng.done and eng.t < cfg.t_max:
            eng.step_guided()
        r_mins.append(eng.r_min)
        if eng.hit:
            hits += 1

    r_mins = np.array(r_mins)
    return {
        'hits': hits, 'n': n_episodes,
        'hit_rate': hits / n_episodes,
        'median_miss': float(np.median(r_mins)),
        'min_miss': float(r_mins.min()),
    }


def train_phase(model, vec_env, steps, phase_name, save_prefix):
    """训练一个阶段"""
    print(f'\n--- {phase_name} ({steps:,} 步) ---')
    checkpoint_cb = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=SAVE_DIR,
        name_prefix=save_prefix,
    )
    model.set_env(vec_env)
    model.learn(total_timesteps=steps, callback=checkpoint_cb,
                reset_num_timesteps=False, progress_bar=False)
    return model


if __name__ == '__main__':
    print('=== PPO微调：6DOF导弹制导 v3 ===')

    # ── PNG基线 ──────────────────────────────────────────────
    print('评估PNG基线...')
    png_res = evaluate_png(n_episodes=200)
    print(f'PNG基线: hits={png_res["hits"]}/200 ({png_res["hit_rate"]*100:.1f}%)'
          f'  median={png_res["median_miss"]:.2f}m')

    # ── 阶段一环境：全部none机动 ──────────────────────────────
    env_fns_none = [make_env(seed=i, maneuver_type='none') for i in range(N_ENVS)]
    vec_env_none = SubprocVecEnv(env_fns_none)

    # ── 初始化PPO模型 ─────────────────────────────────────────
    model = PPO(
        "MlpPolicy", vec_env_none,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cuda",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # ── 加载BC预训练权重 ──────────────────────────────────────
    bc_path = os.path.join(SAVE_DIR, 'bc_pretrained.zip')
    if os.path.exists(bc_path):
        print(f'\n加载BC预训练权重: {bc_path}')
        bc_model = PPO.load(bc_path)
        model.policy.load_state_dict(bc_model.policy.state_dict())
        with torch.no_grad():
            model.policy.log_std.fill_(-1.0)  # std≈0.37，BC会把log_std压到极小值
        print('BC权重加载成功，log_std已重置为-1.0')

        # 评估BC初始性能
        print('\nBC初始性能评估 (200 episodes, random):')
        bc_res = evaluate_policy(model, n_episodes=200)
        print(f'BC策略: hits={bc_res["hits"]}/200 ({bc_res["hit_rate"]*100:.1f}%)'
              f'  median={bc_res["median_miss"]:.2f}m')
    else:
        print(f'\n警告：未找到BC预训练权重 {bc_path}，从随机初始化开始训练')

    # ── 阶段一：1M步，全部none机动 ───────────────────────────
    model = train_phase(model, vec_env_none, 1_000_000,
                        '阶段一：none机动', 'ppo_v3_phase1')

    res = evaluate_policy(model, n_episodes=200)
    print(f'阶段一结束: hits={res["hits"]}/200 ({res["hit_rate"]*100:.1f}%)'
          f'  median={res["median_miss"]:.2f}m')

    # ── 阶段二：2M步，none+sine混合 ──────────────────────────
    curriculum2 = ['none', 'none', 'none', 'none', 'sine', 'sine', 'sine', 'sine']
    env_fns_mix = [make_env(seed=i, maneuver_type=curriculum2[i]) for i in range(N_ENVS)]
    vec_env_mix = SubprocVecEnv(env_fns_mix)

    model = train_phase(model, vec_env_mix, 2_000_000,
                        '阶段二：none+sine混合', 'ppo_v3_phase2')

    res = evaluate_policy(model, n_episodes=200)
    print(f'阶段二结束: hits={res["hits"]}/200 ({res["hit_rate"]*100:.1f}%)'
          f'  median={res["median_miss"]:.2f}m')

    # ── 阶段三：2M步，全部random机动 ─────────────────────────
    env_fns_rand = [make_env(seed=i, maneuver_type='random') for i in range(N_ENVS)]
    vec_env_rand = SubprocVecEnv(env_fns_rand)

    model = train_phase(model, vec_env_rand, 2_000_000,
                        '阶段三：random机动', 'ppo_v3_phase3')

    # ── 保存最终模型 ──────────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, 'ppo_v3_final')
    model.save(final_path)
    print(f'\n模型已保存: {final_path}')

    # ── 最终对比 ──────────────────────────────────────────────
    print('\n=== 最终对比 (200 episodes, random maneuver) ===')
    ppo_res = evaluate_policy(model, n_episodes=200)
    print(f'PPO : hits={ppo_res["hits"]}/200 ({ppo_res["hit_rate"]*100:.1f}%)'
          f'  median={ppo_res["median_miss"]:.2f}m  min={ppo_res["min_miss"]:.2f}m')
    print(f'PNG : hits={png_res["hits"]}/200 ({png_res["hit_rate"]*100:.1f}%)'
          f'  median={png_res["median_miss"]:.2f}m  min={png_res["min_miss"]:.2f}m')

    improvement = ppo_res["hit_rate"] - png_res["hit_rate"]
    print(f'\nPPO vs PNG 命中率提升: {improvement*100:+.1f}%')
