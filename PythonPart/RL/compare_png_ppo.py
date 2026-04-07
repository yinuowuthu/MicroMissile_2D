"""
对比PNG和PPO在相同场景下的行为
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from gym_missile_env import MissileGymEnv
from missile_env import MissileEngagement2D, TargetParams, ProportionalNavigation
import numpy as np

# 创建相同的初始条件
seed = 42
np.random.seed(seed)

print("=" * 60)
print("PNG vs PPO Comparison (same initial conditions)")
print("=" * 60)

# PNG测试
print("\n[PNG Guidance]")
tp = TargetParams(V=50.0, a_max=0, maneuver_type='none')
env_png = MissileEngagement2D(target=tp)
env_png.reset(seed=seed)

png = ProportionalNavigation(N=4.0)

for i in range(10):
    obs = env_png.get_obs()
    ac = png.compute(env_png)
    env_png.step_guidance(ac)

    s = env_png.state
    zem = env_png.compute_zem()

    print(f"Step {i+1}: r={s.r:.1f}m, lam_dot={s.lam_dot:.4f}, ac={ac:.1f} m/s^2, zem={zem:.2f}m")

    if s.done:
        break

print(f"PNG Result: hit={env_png.state.hit}, final_r={env_png.state.r:.2f}m")

# PPO测试
print("\n[PPO Guidance]")
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.zip')
model = PPO.load(model_path)

env_ppo = MissileGymEnv(target_params={'maneuver_type': 'none', 'a_max': 0})
obs, _ = env_ppo.reset(seed=seed)

for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env_ppo.step(action)

    s = env_ppo.env.state
    ac = action[0] * env_ppo.env.mp.a_max

    print(f"Step {i+1}: r={s.r:.1f}m, lam_dot={s.lam_dot:.4f}, ac={ac:.1f} m/s^2, zem={info['zem']:.2f}m")
    print(f"         obs={obs}, action_raw={action[0]:.4f}")

    if done:
        break

print(f"PPO Result: hit={env_ppo.env.state.hit}, final_r={env_ppo.env.state.r:.2f}m")

print("\n" + "=" * 60)
print("Analysis:")
print("PNG uses strong commands (100+ m/s^2) based on lambda_dot")
print("PPO uses weak commands (~8 m/s^2) - policy is too conservative")
print("Possible causes:")
print("  1. Observation normalization issues")
print("  2. Reward function doesn't penalize weak actions enough")
print("  3. Training converged to local minimum (safe but ineffective)")
print("=" * 60)
