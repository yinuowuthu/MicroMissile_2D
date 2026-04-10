"""
行为克隆预训练脚本
==================
阶段一：收集PNG轨迹 → BC训练策略网络 → 预训练价值函数
阶段二：见 train_ppo.py（加载BC权重后PPO微调）
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from missile_gym_env import MissileGym6DOF
from config import TargetParams, SimConfig
from seeker import SeekerParams

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ppo_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 配置 ──────────────────────────────────────────────────────
N_EPISODES   = 5000   # 收集轨迹数
BC_EPOCHS    = 100    # BC训练轮数
VF_EPOCHS    = 50     # 价值函数预训练轮数
BATCH_SIZE   = 512
BC_LR        = 3e-4
VF_LR        = 1e-3
GAMMA        = 0.99


# ── 步骤1：收集PNG轨迹 ────────────────────────────────────────

def collect_png_trajectories(n_episodes=5000, maneuver='none', seed_offset=0):
    """
    用PNG动作驱动RL环境，收集(obs, action, reward)轨迹。
    返回：obs_array, action_array, episode_rewards列表, miss_per_sample数组
    """
    env = MissileGym6DOF(
        tp=TargetParams(maneuver_type=maneuver, V0=30.0),
        cfg=SimConfig(dt=0.0005, t_max=3.0, r_hit=1.0),
    )

    all_obs, all_actions = [], []
    episode_rewards = []
    all_ep_obs = []
    all_ep_rmins = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        ep_obs, ep_actions, ep_rewards = [], [], []

        while not done:
            sd   = env.eng._seeker_data
            V    = env.eng.missile.speed
            mass = env.eng.missile.mass
            a_el, a_az = env.eng.guidance.compute(sd, V, env.cfg.decision_dt, mass)

            action = np.clip(
                [a_el / env.a_max_cmd, a_az / env.a_max_cmd], -1.0, 1.0
            ).astype(np.float32)

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(float(reward))
            done = terminated or truncated

        r_min = info['r_min']
        all_ep_obs.append(ep_obs)
        all_ep_rmins.append(r_min)
        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        episode_rewards.append(ep_rewards)

        if (i + 1) % 500 == 0:
            hits = sum(1 for ep in episode_rewards[-500:] if ep[-1] > 50)
            print(f"  收集进度 {i+1}/{n_episodes}  近500ep命中: {hits}/500")

    # 每个样本对应的episode r_min（用于加权采样）
    miss_per_sample = []
    for ep_obs_list, r_min in zip(all_ep_obs, all_ep_rmins):
        miss_per_sample.extend([r_min] * len(ep_obs_list))

    return (np.array(all_obs, dtype=np.float32),
            np.array(all_actions, dtype=np.float32),
            episode_rewards,
            np.array(miss_per_sample, dtype=np.float32))


# ── 步骤2：计算MC回报 ─────────────────────────────────────────

def compute_mc_returns(episode_rewards, gamma=0.99):
    returns = []
    for ep_rewards in episode_rewards:
        G = 0.0
        ep_returns = []
        for r in reversed(ep_rewards):
            G = r + gamma * G
            ep_returns.insert(0, G)
        returns.extend(ep_returns)
    return np.array(returns, dtype=np.float32)


# ── 步骤3：BC训练策略网络 ─────────────────────────────────────

def train_bc_policy(model, obs_data, action_data, miss_per_sample,
                    n_epochs=100, batch_size=512, lr=3e-4):
    policy = model.policy
    policy.train()

    # 按miss distance加权：miss越小权重越大（5m为特征尺度）
    weights = np.exp(-miss_per_sample / 5.0)
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(
        torch.FloatTensor(weights), num_samples=len(weights), replacement=True
    )
    dataset = TensorDataset(
        torch.FloatTensor(obs_data),
        torch.FloatTensor(action_data)
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    print(f"BC训练：{len(obs_data)} 样本，{n_epochs} epochs")
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, act_b in loader:
            dist = policy.get_distribution(obs_b)
            loss = -dist.log_prob(act_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  BC epoch {epoch+1}/{n_epochs}: loss={total_loss/len(loader):.4f}")


# ── 步骤4：预训练价值函数 ─────────────────────────────────────

def pretrain_value_function(model, obs_data, returns_data,
                            n_epochs=50, batch_size=512, lr=1e-3):
    policy = model.policy
    # 冻结extractor（防止梯度回传破坏BC学到的策略特征）
    policy.mlp_extractor.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.value_net.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.FloatTensor(obs_data),
        torch.FloatTensor(returns_data)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"价值函数预训练：{len(obs_data)} 样本，{n_epochs} epochs")
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_b, ret_b in loader:
            values = policy.predict_values(obs_b).squeeze()
            loss = F.mse_loss(values, ret_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  VF epoch {epoch+1}/{n_epochs}: loss={total_loss/len(loader):.4f}")

    # 恢复梯度，PPO微调时extractor可更新
    policy.mlp_extractor.requires_grad_(True)


# ── 评估BC策略 ────────────────────────────────────────────────

def evaluate_bc(model, n_episodes=200, seed_offset=10000):
    env = MissileGym6DOF(
        tp=TargetParams(maneuver_type='random', V0=30.0),
        cfg=SimConfig(dt=0.0005, t_max=3.0, r_hit=1.0),
        seeker_params=SeekerParams(),
    )
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
    print(f"BC策略评估: hits={hits}/{n_episodes} ({hits/n_episodes*100:.1f}%)"
          f"  median={np.median(r_mins):.2f}m  min={r_mins.min():.2f}m")
    return hits / n_episodes


# ── 主流程 ────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== BC预训练：6DOF导弹制导 ===\n')

    # 创建一个dummy env用于初始化PPO模型
    def _make_env():
        env = MissileGym6DOF(
            tp=TargetParams(maneuver_type='none', V0=30.0),
            cfg=SimConfig(dt=0.0005, t_max=3.0, r_hit=0.5),
            seeker_params=SeekerParams(),
            a_max_cmd=150.0, hit_reward=100.0, miss_penalty=10.0,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])

    # 初始化PPO模型（只用于获取策略网络结构）
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.005,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
    )

    # 步骤1：收集PNG轨迹
    print(f'步骤1：收集PNG轨迹 ({N_EPISODES} episodes, none机动)...')
    obs_data, action_data, ep_rewards, miss_per_sample = collect_png_trajectories(
        n_episodes=N_EPISODES, maneuver='none'
    )
    print(f'  收集完成：{len(obs_data)} 个样本')
    print(f'  miss distance: median={np.median(miss_per_sample):.2f}m'
          f'  min={miss_per_sample.min():.2f}m')

    # 步骤2：计算MC回报
    print('\n步骤2：计算MC回报...')
    returns_data = compute_mc_returns(ep_rewards, gamma=GAMMA)
    print(f'  回报范围: [{returns_data.min():.2f}, {returns_data.max():.2f}]'
          f'  mean={returns_data.mean():.2f}')

    # 步骤3：BC训练策略网络
    print('\n步骤3：BC训练策略网络...')
    train_bc_policy(model, obs_data, action_data, miss_per_sample,
                    n_epochs=BC_EPOCHS, batch_size=BATCH_SIZE, lr=BC_LR)

    # 步骤4：预训练价值函数
    print('\n步骤4：预训练价值函数...')
    pretrain_value_function(model, obs_data, returns_data,
                            n_epochs=VF_EPOCHS, batch_size=BATCH_SIZE, lr=VF_LR)

    # 保存BC预训练模型
    bc_path = os.path.join(SAVE_DIR, 'bc_pretrained')
    model.save(bc_path)
    print(f'\nBC预训练模型已保存: {bc_path}')

    # 评估BC策略
    print('\n=== BC策略评估 (200 episodes, random maneuver) ===')
    evaluate_bc(model)
