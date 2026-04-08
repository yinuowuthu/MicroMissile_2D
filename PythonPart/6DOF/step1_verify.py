"""
第一步验证脚本：6DOF无制导弹道
================================
验证标准：
1. 从静止加速，0.5s内达到100-150m/s
2. 速度剖面：推力段加速，滑行段因阻力减速
3. 有重力弹道：水平发射时弹道下垂合理
4. 3s内飞行距离约200-300m
5. 攻角全程接近0（无制导时）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from config import MissileParams, PropulsionParams, TargetParams, SimConfig
from dynamics import Missile6DOF
from engagement import Engagement, Target, run_ballistic


def test_single_ballistic():
    """测试1：单次无制导直飞弹道（水平发射）"""
    print("=" * 60)
    print("  测试1：水平发射无制导弹道")
    print("=" * 60)

    mp = MissileParams()
    prop = PropulsionParams()
    cfg = SimConfig(t_max=3.0)

    missile = Missile6DOF(mp, prop)
    # 水平发射，初始速度0
    missile.init_state(
        pos=np.array([0.0, 0.0, 0.0]),
        euler=np.array([0.0, 0.0, 0.0]),  # 水平前方
        speed=0.0
    )

    # 记录轨迹
    ts, xs, zs, Vs, alphas, masses, thrusts = [], [], [], [], [], [], []

    dt = cfg.dt
    while missile.t < 3.0:
        ts.append(missile.t)
        xs.append(missile.state[0])
        zs.append(missile.state[2])
        Vs.append(missile.speed)
        alphas.append(np.degrees(missile.alpha))
        masses.append(missile.mass)
        thrusts.append(prop.get_thrust(missile.t))

        missile.step(dt)

    ts = np.array(ts)
    xs = np.array(xs)
    zs = np.array(zs)
    Vs = np.array(Vs)
    alphas = np.array(alphas)
    masses = np.array(masses)
    thrusts = np.array(thrusts)

    # 打印关键数值
    idx_05 = np.searchsorted(ts, 0.5)
    idx_10 = np.searchsorted(ts, 1.0)
    idx_20 = np.searchsorted(ts, 2.0)
    idx_30 = np.searchsorted(ts, 2.99)

    print(f"\n  t=0.5s: V={Vs[idx_05]:.1f} m/s, x={xs[idx_05]:.1f} m, z={zs[idx_05]:.3f} m")
    print(f"  t=1.0s: V={Vs[idx_10]:.1f} m/s, x={xs[idx_10]:.1f} m, z={zs[idx_10]:.3f} m")
    print(f"  t=2.0s: V={Vs[idx_20]:.1f} m/s, x={xs[idx_20]:.1f} m, z={zs[idx_20]:.3f} m")
    print(f"  t=3.0s: V={Vs[idx_30]:.1f} m/s, x={xs[idx_30]:.1f} m, z={zs[idx_30]:.3f} m")
    print(f"  质量: {masses[0]:.3f} → {masses[idx_05]:.3f} → {masses[-1]:.3f} kg")
    print(f"  攻角范围: [{alphas.min():.3f}, {alphas.max():.3f}] deg")

    # 验证
    V_05 = Vs[idx_05]
    V_30 = Vs[idx_30]
    x_30 = xs[idx_30]

    checks = [
        ("V(0.5s) 在 100-200 m/s", 100 <= V_05 <= 200),
        ("V(3.0s) 在 50-180 m/s", 50 <= V_30 <= 180),
        ("x(3.0s) 在 100-500 m", 100 <= x_30 <= 500),
        ("|alpha| < 10 deg", np.max(np.abs(alphas)) < 10),
        ("z(3.0s) > 0 (重力下垂)", zs[idx_30] > 0),
    ]

    print("\n  验证结果:")
    all_pass = True
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {desc}")
        if not ok:
            all_pass = False

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Step 1: Ballistic Trajectory (Horizontal Launch, No Guidance)', fontsize=14)

    axes[0, 0].plot(ts, Vs, 'b-')
    axes[0, 0].axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Burn end')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (m/s)')
    axes[0, 0].set_title('Speed vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(ts, xs, 'b-')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X position (m)')
    axes[0, 1].set_title('Downrange Distance')
    axes[0, 1].grid(True)

    axes[0, 2].plot(ts, zs, 'b-')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Z position (m, NED down)')
    axes[0, 2].set_title('Altitude Drop (Z down = positive)')
    axes[0, 2].grid(True)

    axes[1, 0].plot(ts, alphas, 'b-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Alpha (deg)')
    axes[1, 0].set_title('Angle of Attack')
    axes[1, 0].grid(True)

    axes[1, 1].plot(ts, thrusts, 'r-')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Thrust (N)')
    axes[1, 1].set_title('Thrust Profile')
    axes[1, 1].grid(True)

    axes[1, 2].plot(ts, masses, 'g-')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Mass (kg)')
    axes[1, 2].set_title('Mass vs Time')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step1_ballistic.png'), dpi=150)
    print(f"\n  图表已保存: step1_ballistic.png")

    return all_pass


def test_elevation_angles():
    """测试2：不同发射仰角的弹道族"""
    print("\n" + "=" * 60)
    print("  测试2：不同发射仰角弹道族")
    print("=" * 60)

    mp = MissileParams()
    prop = PropulsionParams()
    cfg = SimConfig(t_max=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ballistic Trajectories at Different Launch Elevations', fontsize=14)

    angles = [0, 15, 30, 45]
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)

        missile = Missile6DOF(mp, prop)
        # NED标准: theta正 = 弹头上仰
        # DCM中body-x的Z分量 = -sin(theta), theta>0时Z<0 = 向上
        missile.init_state(
            pos=np.array([0.0, 0.0, 0.0]),
            euler=np.array([0.0, angle_rad, 0.0]),  # 正theta = 弹头上仰
            speed=0.0
        )

        ts, xs, zs, Vs = [], [], [], []
        dt = cfg.dt
        while missile.t < 3.0:
            ts.append(missile.t)
            xs.append(missile.state[0])
            zs.append(-missile.state[2])  # 转为高度（NED中z向下，取负为高度）
            Vs.append(missile.speed)
            missile.step(dt)

        ts = np.array(ts)
        label = f'{angle_deg}°'
        axes[0].plot(np.array(xs), np.array(zs), label=label)
        axes[1].plot(ts, np.array(Vs), label=label)

        print(f"  {angle_deg}°: x(3s)={xs[-1]:.0f}m, h(3s)={zs[-1]:.1f}m, V(3s)={Vs[-1]:.1f}m/s")

    axes[0].set_xlabel('Downrange X (m)')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].set_title('Trajectory Profile')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_aspect('equal')

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].set_title('Speed vs Time')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step1_elevation.png'), dpi=150)
    print(f"\n  图表已保存: step1_elevation.png")


def test_engagement_setup():
    """测试3：交战场景初始化"""
    print("\n" + "=" * 60)
    print("  测试3：交战场景初始化验证")
    print("=" * 60)

    eng = Engagement(
        tp=TargetParams(maneuver_type="none"),
        cfg=SimConfig(t_max=3.0)
    )
    eng.reset(r0=200.0, seed=42)

    print(f"  导弹位置: {eng.missile.pos}")
    print(f"  导弹姿态: {np.degrees(eng.missile.euler)} deg")
    print(f"  目标位置: {eng.target.pos}")
    print(f"  目标速度: {eng.target.vel}")
    print(f"  初始距离: {eng.get_range():.1f} m")

    traj = run_ballistic(eng, t_end=3.0)

    print(f"\n  仿真结束: t={traj['flight_time']:.2f}s, reason={traj['reason']}")
    print(f"  导弹末位置: x={traj['x'][-1]:.1f}, y={traj['y'][-1]:.1f}, z={traj['z'][-1]:.1f}")
    print(f"  导弹末速度: {traj['V'][-1]:.1f} m/s")
    print(f"  最小距离: {traj['miss_distance']:.2f} m")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  6DOF动力学第一步验证")
    print("#" * 60)

    ok = test_single_ballistic()
    test_elevation_angles()
    test_engagement_setup()

    print("\n" + "#" * 60)
    if ok:
        print("#  第一步验证通过！弹道剖面物理合理。")
    else:
        print("#  第一步验证有失败项，需要诊断。")
    print("#" * 60)

    plt.show()
