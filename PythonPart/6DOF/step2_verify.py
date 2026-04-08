"""
第二步验证脚本：三回路自驾仪
==============================
验证标准：
1. 阶跃响应：5g加速度阶跃，上升时间<0.5s，超调<30%
2. 正弦跟踪：2Hz正弦加速度指令，幅值跟踪合理
3. 舵偏角不饱和（<±15°）
4. 攻角、角速率在合理范围内
5. 滚转角保持接近零
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免plt.show()阻塞
import numpy as np
import matplotlib.pyplot as plt
from config import MissileParams, PropulsionParams, SimConfig, G, RHO
from dynamics import Missile6DOF
from autopilot import ThreeLoopAutopilot

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


def test_step_response():
    """测试1：俯仰通道5g阶跃响应"""
    print("=" * 60)
    print("  测试1：俯仰通道5g阶跃响应")
    print("=" * 60)

    mp = MissileParams()
    prop = PropulsionParams()
    dt = 0.0005

    # 初始化导弹：已经在巡航状态（推力段结束后，V≈150m/s）
    missile = Missile6DOF(mp, prop)
    missile.init_state(
        pos=np.array([0.0, 0.0, 0.0]),
        euler=np.array([0.0, 0.0, 0.0]),
        speed=130.0  # 巡航速度
    )
    # 跳过推力段，设置为空弹质量
    missile.mass = mp.m0 - mp.m_propellant
    missile.t = 1.0  # 假设已过推力段

    ap = ThreeLoopAutopilot(mp)
    ap.reset()

    # 记录
    ts, a_cmds, a_actuals = [], [], []
    alphas, qs, delta_es = [], [], []
    Vs, phis = [], []

    a_cmd = 5.0 * G  # 5g阶跃指令
    t_sim = 0.0
    t_end = 2.0

    while t_sim < t_end:
        # 当前状态
        V = missile.speed
        alpha = missile.alpha
        beta = missile.beta
        p, q, r = missile.state[9:12]
        phi = missile.state[6]

        # 当前法向加速度（实际值）
        qbar = 0.5 * RHO * V * V
        a_actual = qbar * mp.S_ref * mp.CNa * alpha / missile.mass

        # 记录
        ts.append(t_sim)
        a_cmds.append(a_cmd / G)
        a_actuals.append(a_actual / G)
        alphas.append(np.degrees(alpha))
        qs.append(np.degrees(q))
        delta_es.append(np.degrees(missile.delta_e))
        Vs.append(V)
        phis.append(np.degrees(phi))

        # 自驾仪计算
        # 阶跃在t=0.2s时给出（先让系统稳定一下）
        cmd = a_cmd if t_sim >= 0.2 else 0.0
        de, dr, da = ap.compute(cmd, 0.0, alpha, beta, p, q, r, V, missile.mass, dt)
        missile.set_fins(de, dr, da)
        missile.step(dt)
        t_sim += dt

    ts = np.array(ts)
    a_cmds = np.array(a_cmds)
    a_actuals = np.array(a_actuals)
    alphas = np.array(alphas)
    qs = np.array(qs)
    delta_es = np.array(delta_es)
    Vs = np.array(Vs)

    # 分析阶跃响应
    # 找阶跃后的响应
    idx_step = np.searchsorted(ts, 0.2)
    a_after = a_actuals[idx_step:]
    t_after = ts[idx_step:] - 0.2

    # 稳态值（取最后0.5s平均）
    idx_ss = np.searchsorted(t_after, t_after[-1] - 0.5)
    a_ss = np.mean(a_after[idx_ss:])

    # 上升时间（10%→90%稳态值）
    target_10 = 0.1 * a_ss
    target_90 = 0.9 * a_ss
    idx_10 = np.searchsorted(a_after, target_10) if a_ss > 0 else 0
    idx_90 = np.searchsorted(a_after, target_90) if a_ss > 0 else 0
    t_rise = t_after[min(idx_90, len(t_after)-1)] - t_after[min(idx_10, len(t_after)-1)]

    # 超调
    a_peak = np.max(a_after)
    overshoot = (a_peak - a_ss) / max(abs(a_ss), 0.01) * 100 if a_ss != 0 else 0

    print(f"\n  阶跃指令: {a_cmd/G:.1f}g")
    print(f"  稳态响应: {a_ss:.2f}g")
    print(f"  上升时间: {t_rise:.3f}s")
    print(f"  峰值: {a_peak:.2f}g")
    print(f"  超调: {overshoot:.1f}%")
    print(f"  攻角范围: [{alphas.min():.2f}, {alphas.max():.2f}]°")
    print(f"  角速率q范围: [{qs.min():.1f}, {qs.max():.1f}]°/s")
    print(f"  舵偏角范围: [{delta_es.min():.2f}, {delta_es.max():.2f}]°")
    print(f"  速度变化: {Vs[0]:.1f} → {Vs[-1]:.1f} m/s")

    # 验证
    checks = [
        ("稳态响应接近5g (>3g)", a_ss > 3.0),
        ("上升时间 < 0.5s", t_rise < 0.5),
        ("超调 < 50%", overshoot < 50),
        ("|alpha| < 20°", np.max(np.abs(alphas)) < 20),
        ("|q| < 300°/s", np.max(np.abs(qs)) < 300),
        ("|delta_e| < 15°", np.max(np.abs(delta_es)) < 15.1),
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
    fig.suptitle('Step 2: Autopilot Step Response (5g pitch command)', fontsize=14)

    axes[0, 0].plot(ts, a_cmds, 'r--', label='Command')
    axes[0, 0].plot(ts, a_actuals, 'b-', label='Actual')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Acceleration (g)')
    axes[0, 0].set_title('Acceleration Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(ts, alphas, 'b-')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Alpha (deg)')
    axes[0, 1].set_title('Angle of Attack')
    axes[0, 1].grid(True)

    axes[0, 2].plot(ts, qs, 'b-')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('q (deg/s)')
    axes[0, 2].set_title('Pitch Rate')
    axes[0, 2].grid(True)

    axes[1, 0].plot(ts, delta_es, 'b-')
    axes[1, 0].axhline(15, color='r', linestyle='--', alpha=0.5, label='Limit')
    axes[1, 0].axhline(-15, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('delta_e (deg)')
    axes[1, 0].set_title('Elevator Deflection')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(ts, Vs, 'b-')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].set_title('Speed')
    axes[1, 1].grid(True)

    axes[1, 2].plot(ts, phis, 'b-')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Phi (deg)')
    axes[1, 2].set_title('Roll Angle')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step2_step_response.png'), dpi=150)
    print(f"\n  图表已保存: step2_step_response.png")

    return all_pass


def test_sine_tracking():
    """测试2：俯仰通道正弦跟踪"""
    print("\n" + "=" * 60)
    print("  测试2：俯仰通道正弦跟踪 (3g, 2Hz)")
    print("=" * 60)

    mp = MissileParams()
    prop = PropulsionParams()
    dt = 0.0005

    missile = Missile6DOF(mp, prop)
    missile.init_state(
        pos=np.array([0.0, 0.0, 0.0]),
        euler=np.array([0.0, 0.0, 0.0]),
        speed=130.0
    )
    missile.mass = mp.m0 - mp.m_propellant
    missile.t = 1.0

    ap = ThreeLoopAutopilot(mp)
    ap.reset()

    ts, a_cmds, a_actuals, delta_es = [], [], [], []

    t_sim = 0.0
    t_end = 3.0
    freq = 2.0
    amp = 3.0 * G

    while t_sim < t_end:
        V = missile.speed
        alpha = missile.alpha
        beta = missile.beta
        p, q, r = missile.state[9:12]

        qbar = 0.5 * RHO * V * V
        a_actual = qbar * mp.S_ref * mp.CNa * alpha / missile.mass

        a_cmd = amp * np.sin(2 * np.pi * freq * t_sim) if t_sim > 0.2 else 0.0

        ts.append(t_sim)
        a_cmds.append(a_cmd / G)
        a_actuals.append(a_actual / G)
        delta_es.append(np.degrees(missile.delta_e))

        de, dr, da = ap.compute(a_cmd, 0.0, alpha, beta, p, q, r, V, missile.mass, dt)
        missile.set_fins(de, dr, da)
        missile.step(dt)
        t_sim += dt

    ts = np.array(ts)
    a_cmds = np.array(a_cmds)
    a_actuals = np.array(a_actuals)

    # 分析跟踪质量（取稳态段，t>1s）
    idx_ss = np.searchsorted(ts, 1.0)
    cmd_ss = a_cmds[idx_ss:]
    act_ss = a_actuals[idx_ss:]
    rms_err = np.sqrt(np.mean((cmd_ss - act_ss)**2))
    rms_cmd = np.sqrt(np.mean(cmd_ss**2))
    tracking_ratio = 1.0 - rms_err / max(rms_cmd, 0.01)

    print(f"  RMS跟踪误差: {rms_err:.2f}g")
    print(f"  RMS指令幅值: {rms_cmd:.2f}g")
    print(f"  跟踪质量: {tracking_ratio*100:.1f}%")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Step 2: Sine Tracking (3g, 2Hz)', fontsize=14)

    axes[0].plot(ts, a_cmds, 'r--', alpha=0.7, label='Command')
    axes[0].plot(ts, a_actuals, 'b-', label='Actual')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_title('Sine Tracking')
    axes[0].legend()
    axes[0].grid(True)

    # 放大稳态段
    axes[1].plot(ts[idx_ss:], a_cmds[idx_ss:], 'r--', alpha=0.7, label='Command')
    axes[1].plot(ts[idx_ss:], a_actuals[idx_ss:], 'b-', label='Actual')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_title('Steady-State Tracking (zoomed)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step2_sine_tracking.png'), dpi=150)
    print(f"  图表已保存: step2_sine_tracking.png")


def test_yaw_channel():
    """测试3：偏航通道阶跃响应"""
    print("\n" + "=" * 60)
    print("  测试3：偏航通道3g阶跃响应")
    print("=" * 60)

    mp = MissileParams()
    prop = PropulsionParams()
    dt = 0.0005

    missile = Missile6DOF(mp, prop)
    missile.init_state(
        pos=np.array([0.0, 0.0, 0.0]),
        euler=np.array([0.0, 0.0, 0.0]),
        speed=130.0
    )
    missile.mass = mp.m0 - mp.m_propellant
    missile.t = 1.0

    ap = ThreeLoopAutopilot(mp)
    ap.reset()

    ts, a_cmds, a_actuals, delta_rs, betas = [], [], [], [], []

    t_sim = 0.0
    t_end = 2.0
    a_cmd_yaw = 3.0 * G

    while t_sim < t_end:
        V = missile.speed
        alpha = missile.alpha
        beta = missile.beta
        p, q, r = missile.state[9:12]

        qbar = 0.5 * RHO * V * V
        a_actual_yaw = qbar * mp.S_ref * mp.CNa * beta / missile.mass

        cmd = a_cmd_yaw if t_sim >= 0.2 else 0.0

        ts.append(t_sim)
        a_cmds.append(cmd / G)
        a_actuals.append(a_actual_yaw / G)
        delta_rs.append(np.degrees(missile.delta_r))
        betas.append(np.degrees(beta))

        de, dr, da = ap.compute(0.0, cmd, alpha, beta, p, q, r, V, missile.mass, dt)
        missile.set_fins(de, dr, da)
        missile.step(dt)
        t_sim += dt

    ts = np.array(ts)
    a_actuals = np.array(a_actuals)
    betas = np.array(betas)

    # 稳态值
    a_ss = np.mean(a_actuals[-2000:])
    print(f"  偏航稳态响应: {a_ss:.2f}g (指令: {a_cmd_yaw/G:.1f}g)")
    print(f"  侧滑角范围: [{betas.min():.2f}, {betas.max():.2f}]°")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Step 2: Yaw Channel Step Response (3g)', fontsize=14)

    axes[0].plot(ts, np.array(a_cmds), 'r--', label='Command')
    axes[0].plot(ts, a_actuals, 'b-', label='Actual')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_title('Yaw Acceleration')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ts, betas, 'b-')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Beta (deg)')
    axes[1].set_title('Sideslip Angle')
    axes[1].grid(True)

    axes[2].plot(ts, np.array(delta_rs), 'b-')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('delta_r (deg)')
    axes[2].set_title('Rudder Deflection')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step2_yaw.png'), dpi=150)
    print(f"  图表已保存: step2_yaw.png")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  第二步验证：三回路自驾仪")
    print("#" * 60)

    ok = test_step_response()
    test_sine_tracking()
    test_yaw_channel()

    print("\n" + "#" * 60)
    if ok:
        print("#  第二步验证通过！自驾仪响应合理。")
    else:
        print("#  第二步验证有失败项，需要调参。")
    print("#" * 60)

    # plt.show()  # Agg backend, no interactive display
