"""
2D Point-Mass Augmented Proportional Navigation (APN) Simulation
================================================================
Vectorized version: all Monte-Carlo runs advance simultaneously.
Supports GPU acceleration via CuPy (auto-selects based on batch size).

Kinematic model based on 2D relative motion equations.
Guidance law: ac = N * VM * lambda_dot + 0.5 * N * aT_normal
              (APN with target acceleration compensation)
Autopilot: first-order lag  tau * a_dot + a = ac
Target: constant velocity + random step/sinusoidal maneuvers

Key difference from PNG:
- APN requires estimation of target normal acceleration
- Better performance against maneuvering targets
- Higher computational cost (needs target state estimation)
"""

import time as _time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Backend selection: CuPy (GPU) for large batches, NumPy (CPU) otherwise
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[Backend] CuPy (CUDA GPU) detected.")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("[Backend] CuPy not found, using NumPy (CPU).")

GPU_THRESHOLD = 500
SYNC_INTERVAL = 100
G = 9.81  # m/s^2


def get_backend(n_runs: int):
    """Return (xp_module, is_gpu) based on batch size."""
    if GPU_AVAILABLE and n_runs >= GPU_THRESHOLD:
        return cp, True
    return np, False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MissileParams:
    V: float = 300.0
    N: float = 4.0
    tau: float = 0.2
    max_accel: float = 40 * G
    fov: float = np.deg2rad(60)


@dataclass
class TargetParams:
    V: float = 50.0
    maneuver_type: str = "none"   # "none", "step", "sine"
    maneuver_g: float = 5.0
    maneuver_freq: float = 1.0
    maneuver_start: float = 0.0


@dataclass
class SimConfig:
    dt: float = 0.001
    t_max: float = 20.0
    miss_dist: float = 0.5


@dataclass
class SimResult:
    """Result for a single engagement (used for plotting)."""
    t_arr: np.ndarray = None
    xM_arr: np.ndarray = None
    yM_arr: np.ndarray = None
    xT_arr: np.ndarray = None
    yT_arr: np.ndarray = None
    r_arr: np.ndarray = None
    ac_arr: np.ndarray = None
    am_arr: np.ndarray = None
    aT_est_arr: np.ndarray = None  # 新增：目标加速度估计
    miss_distance: float = 0.0
    flight_time: float = 0.0
    energy: float = 0.0
    hit: bool = False


# ---------------------------------------------------------------------------
# Vectorized batch simulation (all n_runs in parallel) - APN version
# ---------------------------------------------------------------------------
def run_batch_vectorized(
    mp: MissileParams,
    tp: TargetParams,
    cfg: SimConfig,
    init_r: np.ndarray,
    init_lambda: np.ndarray,
    init_gamma_M: np.ndarray,
    init_gamma_T: np.ndarray,
    signs: np.ndarray,
):
    """
    Run n_runs simulations simultaneously using APN guidance.
    Auto-selects GPU when n_runs >= GPU_THRESHOLD.
    Returns: (miss_distances, flight_times, energies, hits) as numpy arrays.
    """
    n = init_r.shape[0]
    xp, is_gpu = get_backend(n)

    # Transfer to GPU if needed
    if is_gpu:
        init_r     = xp.asarray(init_r)
        init_lambda  = xp.asarray(init_lambda)
        init_gamma_M = xp.asarray(init_gamma_M)
        init_gamma_T = xp.asarray(init_gamma_T)
        signs        = xp.asarray(signs)

    VM = mp.V
    VT = tp.V
    dt = cfg.dt
    max_steps = int(cfg.t_max / dt) + 1
    a_max_tgt = tp.maneuver_g * G

    # Pre-compute sine wave amplitude scalar for sine maneuver
    two_pi_f = 2.0 * float(np.pi) * tp.maneuver_freq

    # State vectors  (n_runs,)
    xM      = xp.zeros(n)
    yM      = xp.zeros(n)
    xT      = init_r * xp.cos(init_lambda)
    yT      = init_r * xp.sin(init_lambda)
    gamma_M = init_gamma_M.copy()
    gamma_T = init_gamma_T.copy()
    am      = xp.zeros(n)

    # Target acceleration estimation (简单的一阶滤波器)
    aT_est  = xp.zeros(n)
    aT_prev = xp.zeros(n)

    # Per-run tracking
    min_r      = init_r.copy()
    min_r_time = xp.zeros(n)
    energy     = xp.zeros(n)
    active     = xp.ones(n, dtype=bool)

    # Rolling range history
    prev_r      = init_r.copy()
    prev_prev_r = init_r.copy()

    t = 0.0
    for step in range(max_steps):

        # ---- EARLY EXIT: only sync GPU every SYNC_INTERVAL steps --------
        if step % SYNC_INTERVAL == 0:
            if not bool(xp.any(active)):
                break

        # Relative geometry
        dx  = xT - xM
        dy  = yT - yM
        r   = xp.sqrt(dx * dx + dy * dy)
        lam = xp.arctan2(dy, dx)

        # Update min range
        closer    = active & (r < min_r)
        min_r     = xp.where(closer, r, min_r)
        min_r_time = xp.where(closer, t, min_r_time)

        # Termination flags
        hit_now = active & (r < cfg.miss_dist)
        passed  = (active
                   & (step > 10)
                   & (r > prev_r)
                   & (prev_r > prev_prev_r)
                   & (min_r < init_r * 0.3))
        timeout = active & (t >= cfg.t_max)
        angle_off = (lam - gamma_M + xp.pi) % (2.0 * xp.pi) - xp.pi
        out_fov   = active & (xp.abs(angle_off) > mp.fov) & (t > 0.1)

        active = active & ~(hit_now | passed | timeout | out_fov)

        # Rotate range history
        prev_prev_r = prev_r
        prev_r      = r

        # ---- Guidance & control (APN) -----------------------------------
        r_safe  = xp.maximum(r, 1e-6)
        lam_dot = (VT * xp.sin(gamma_T - lam) - VM * xp.sin(gamma_M - lam)) / r_safe

        # Target maneuver (ground truth for simulation)
        if t < tp.maneuver_start or tp.maneuver_type == "none":
            aT = 0.0
        elif tp.maneuver_type == "step":
            aT = signs * a_max_tgt
        else:  # "sine"
            aT = a_max_tgt * float(np.sin(two_pi_f * (t - tp.maneuver_start)))

        # Target acceleration estimation (简单差分 + 低通滤波)
        # 实际中需要用卡尔曼滤波或其他估计器
        if isinstance(aT, float):
            aT_true = aT * xp.ones(n)
        else:
            aT_true = aT

        # 一阶低通滤波: aT_est = alpha * aT_true + (1-alpha) * aT_prev
        alpha = 0.1  # 滤波系数（实际中需要调优）
        aT_est = alpha * aT_true + (1.0 - alpha) * aT_prev
        aT_prev = aT_est.copy()

        # 计算目标法向加速度分量（垂直于视线方向）
        aT_normal = aT_est * xp.cos(lam - gamma_T)

        # APN guidance law: ac = N * VM * lam_dot + 0.5 * N * aT_normal
        ac = mp.N * VM * lam_dot + 0.5 * mp.N * aT_normal
        ac = xp.clip(ac, -mp.max_accel, mp.max_accel)

        # Autopilot dynamics
        am_new = xp.clip(am + (ac - am) * (dt / mp.tau), -mp.max_accel, mp.max_accel)

        # State update — mask inactive runs with zero delta
        act_f   = active.astype(xp.float64)
        d_gM    = (am_new / VM) * dt * act_f
        gamma_M = gamma_M + d_gM
        xM      = xM + VM * xp.cos(gamma_M) * dt * act_f
        yM      = yM + VM * xp.sin(gamma_M) * dt * act_f
        am      = xp.where(active, am_new, am)

        if isinstance(aT, float):
            d_gT = (aT / VT) * dt * act_f
        else:
            d_gT = (aT / VT) * dt * act_f
        gamma_T = gamma_T + d_gT
        xT      = xT + VT * xp.cos(gamma_T) * dt * act_f
        yT      = yT + VT * xp.sin(gamma_T) * dt * act_f

        energy  = energy + am * am * dt * act_f
        t += dt

    # Collect results
    hits = (min_r < cfg.miss_dist)

    # Transfer back to CPU if on GPU
    if is_gpu:
        min_r      = cp.asnumpy(min_r)
        min_r_time = cp.asnumpy(min_r_time)
        energy     = cp.asnumpy(energy)
        hits       = cp.asnumpy(hits)

    return min_r, min_r_time, energy, hits


# ---------------------------------------------------------------------------
# Single run (scalar, for plotting only) - APN version
# ---------------------------------------------------------------------------
def run_simulation_single(
    mp: MissileParams,
    tp: TargetParams,
    cfg: SimConfig,
    init_r: float = 3000.0,
    init_lambda: float = 0.0,
    init_gamma_M: float = 0.0,
    init_gamma_T: float = np.pi,
    seed: int = 0,
) -> SimResult:
    """Run a single engagement with full trajectory storage (for plotting)."""
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1, 1])

    xM, yM = 0.0, 0.0
    xT = init_r * np.cos(init_lambda)
    yT = init_r * np.sin(init_lambda)
    gamma_M = init_gamma_M
    gamma_T = init_gamma_T
    am = 0.0
    aT_est = 0.0
    aT_prev = 0.0
    VM, VT = mp.V, tp.V

    max_steps = int(cfg.t_max / cfg.dt) + 1
    t_arr = np.zeros(max_steps)
    xM_arr = np.zeros(max_steps)
    yM_arr = np.zeros(max_steps)
    xT_arr = np.zeros(max_steps)
    yT_arr = np.zeros(max_steps)
    r_arr = np.zeros(max_steps)
    ac_arr = np.zeros(max_steps)
    am_arr = np.zeros(max_steps)
    aT_est_arr = np.zeros(max_steps)

    t = 0.0
    hit = False
    min_r = init_r
    idx_min_r = 0
    energy = 0.0

    for i in range(max_steps):
        dx, dy = xT - xM, yT - yM
        r = np.sqrt(dx**2 + dy**2)
        lam = np.arctan2(dy, dx)

        t_arr[i] = t
        xM_arr[i], yM_arr[i] = xM, yM
        xT_arr[i], yT_arr[i] = xT, yT
        r_arr[i] = r
        am_arr[i] = am
        aT_est_arr[i] = aT_est

        if r < min_r:
            min_r = r
            idx_min_r = i

        if r < cfg.miss_dist:
            hit = True
            ac_arr[i] = ac_arr[max(0, i-1)]
            break
        if i > 10 and r > r_arr[i-1] and r_arr[i-1] > r_arr[i-2] and min_r < init_r * 0.3:
            ac_arr[i] = ac_arr[max(0, i-1)]
            break
        if t >= cfg.t_max:
            ac_arr[i] = ac_arr[max(0, i-1)]
            break
        angle_off = (lam - gamma_M + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_off) > mp.fov and t > 0.1:
            ac_arr[i] = ac_arr[max(0, i-1)]
            break

        lam_dot = (VT * np.sin(gamma_T - lam) - VM * np.sin(gamma_M - lam)) / max(r, 1e-6)

        # Target maneuver
        a_max = tp.maneuver_g * G
        if t < tp.maneuver_start or tp.maneuver_type == "none":
            aT = 0.0
        elif tp.maneuver_type == "step":
            aT = sign * a_max
        elif tp.maneuver_type == "sine":
            aT = a_max * np.sin(2 * np.pi * tp.maneuver_freq * (t - tp.maneuver_start))
        else:
            aT = 0.0

        # Target acceleration estimation (一阶低通滤波)
        alpha = 0.1
        aT_est = alpha * aT + (1.0 - alpha) * aT_prev
        aT_prev = aT_est

        # 计算目标法向加速度
        aT_normal = aT_est * np.cos(lam - gamma_T)

        # APN guidance law
        ac = mp.N * VM * lam_dot + 0.5 * mp.N * aT_normal
        ac = np.clip(ac, -mp.max_accel, mp.max_accel)
        ac_arr[i] = ac

        am_new = np.clip(am + (ac - am) / mp.tau * cfg.dt, -mp.max_accel, mp.max_accel)

        gamma_M += (am_new / VM) * cfg.dt
        xM += VM * np.cos(gamma_M) * cfg.dt
        yM += VM * np.sin(gamma_M) * cfg.dt
        am = am_new

        gamma_T += (aT / VT) * cfg.dt
        xT += VT * np.cos(gamma_T) * cfg.dt
        yT += VT * np.sin(gamma_T) * cfg.dt

        energy += am**2 * cfg.dt
        t += cfg.dt

    n = i + 1
    return SimResult(
        t_arr=t_arr[:n], xM_arr=xM_arr[:n], yM_arr=yM_arr[:n],
        xT_arr=xT_arr[:n], yT_arr=yT_arr[:n], r_arr=r_arr[:n],
        ac_arr=ac_arr[:n], am_arr=am_arr[:n], aT_est_arr=aT_est_arr[:n],
        miss_distance=min_r, flight_time=t_arr[min(idx_min_r, n-1)],
        energy=energy, hit=hit,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_engagement(result: SimResult, filename: str = "engagement_apn.png"):
    """Plot trajectory and key metrics for a single engagement."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory
    ax = axes[0, 0]
    ax.plot(result.xM_arr, result.yM_arr, 'b-', label='Missile', linewidth=1.5)
    ax.plot(result.xT_arr, result.yT_arr, 'r--', label='Target', linewidth=1.5)
    ax.plot(result.xM_arr[0], result.yM_arr[0], 'bo', markersize=8, label='M start')
    ax.plot(result.xT_arr[0], result.yT_arr[0], 'ro', markersize=8, label='T start')
    ax.plot(result.xM_arr[-1], result.yM_arr[-1], 'bx', markersize=10, label='M end')
    ax.plot(result.xT_arr[-1], result.yT_arr[-1], 'rx', markersize=10, label='T end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'APN Trajectory (miss={result.miss_distance:.2f}m, hit={result.hit})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Range vs time
    ax = axes[0, 1]
    ax.plot(result.t_arr, result.r_arr, 'k-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Range (m)')
    ax.set_title('Missile-Target Range')
    ax.grid(True, alpha=0.3)

    # Acceleration commands
    ax = axes[1, 0]
    ax.plot(result.t_arr, result.ac_arr / G, 'b-', label='Command (ac)', linewidth=1.5)
    ax.plot(result.t_arr, result.am_arr / G, 'r--', label='Actual (am)', linewidth=1.5)
    if result.aT_est_arr is not None:
        ax.plot(result.t_arr, result.aT_est_arr / G, 'g:', label='Target est.', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.set_title('APN Guidance Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[1, 1]
    energy_arr = np.cumsum(result.am_arr**2) * (result.t_arr[1] - result.t_arr[0])
    ax.plot(result.t_arr, energy_arr, 'purple', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Energy (m²/s⁴)')
    ax.set_title(f'Control Energy (total={result.energy:.1f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved to {filename}")


# ---------------------------------------------------------------------------
# Batch Monte-Carlo runner
# ---------------------------------------------------------------------------
def run_batch(
    N_values: List[float],
    tau_values: List[float],
    maneuver_types: List[str],
    maneuver_gs: List[float],
    n_runs: int = 50,
    init_r: float = 3000.0,
) -> dict:
    """
    Run Monte-Carlo batch for all parameter combinations.
    Returns a dict with results indexed by (N, tau, maneuver_type, maneuver_g).
    """
    cfg = SimConfig()
    results = {}

    total_cases = len(N_values) * len(tau_values) * len(maneuver_types) * len(maneuver_gs)
    case_idx = 0

    for N, tau, mtype, mg in itertools.product(N_values, tau_values, maneuver_types, maneuver_gs):
        case_idx += 1
        mp = MissileParams(N=N, tau=tau)
        tp = TargetParams(maneuver_type=mtype, maneuver_g=mg, maneuver_start=2.0)

        rng = np.random.default_rng(42)
        r0 = init_r + rng.uniform(-200, 200, n_runs)
        lam0 = rng.uniform(np.deg2rad(-20), np.deg2rad(20), n_runs)
        gM0 = lam0 + rng.uniform(np.deg2rad(-5), np.deg2rad(5), n_runs)
        gT0 = np.pi + lam0 + rng.uniform(np.deg2rad(-30), np.deg2rad(30), n_runs)
        signs = rng.choice([-1.0, 1.0], n_runs)

        misses, times, energies, hits = run_batch_vectorized(mp, tp, cfg, r0, lam0, gM0, gT0, signs)

        results[(N, tau, mtype, mg)] = {
            'miss_mean': np.mean(misses),
            'miss_std': np.std(misses),
            'hit_rate': np.sum(hits) / n_runs,
            'time_mean': np.mean(times),
            'energy_mean': np.mean(energies),
        }

        if case_idx % 10 == 0:
            print(f"  [{case_idx}/{total_cases}] N={N}, tau={tau}, {mtype}, {mg}g")

    return results


def print_table(results: dict):
    """Print results in a readable table format."""
    print("\n" + "="*80)
    print(f"{'N':<6} {'tau':<6} {'Maneuver':<10} {'g':<5} {'Hit%':<8} {'Miss(m)':<12} {'Time(s)':<10} {'Energy':<10}")
    print("="*80)

    for key in sorted(results.keys()):
        N, tau, mtype, mg = key
        r = results[key]
        print(f"{N:<6.1f} {tau:<6.2f} {mtype:<10} {mg:<5.0f} "
              f"{r['hit_rate']*100:<8.1f} {r['miss_mean']:<6.3f}±{r['miss_std']:<5.3f} "
              f"{r['time_mean']:<10.2f} {r['energy_mean']:<10.1f}")
    print("="*80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n=== Augmented Proportional Navigation (APN) Simulation ===\n")

    cfg = SimConfig()

    # --- Single engagement plots (different scenarios) ---
    scenarios = [
        ("none", 0, "engagement_apn_none.png"),
        ("step", 10, "engagement_apn_step10g.png"),
        ("sine", 10, "engagement_apn_sine10g.png"),
    ]

    for mtype, mg, fname in scenarios:
        mp = MissileParams(N=4, tau=0.2)
        tp = TargetParams(maneuver_type=mtype, maneuver_g=mg, maneuver_start=2.0)
        result = run_simulation_single(mp, tp, cfg, init_r=3000.0, init_lambda=0.0,
                                       init_gamma_M=0.0, init_gamma_T=np.pi, seed=42)
        plot_engagement(result, fname)

    # --- Performance benchmark ---
    print("\n--- Performance Benchmark (APN) ---")
    mp_bench = MissileParams(N=4, tau=0.2)
    tp_bench = TargetParams(maneuver_type="step", maneuver_g=10, maneuver_start=2.0)

    for n_runs in [20, 500, 2000]:
        rng = np.random.default_rng(42)
        r0 = 3000.0 + rng.uniform(-200, 200, n_runs)
        lam0 = rng.uniform(np.deg2rad(-20), np.deg2rad(20), n_runs)
        gM0 = lam0 + rng.uniform(np.deg2rad(-5), np.deg2rad(5), n_runs)
        gT0 = np.pi + lam0 + rng.uniform(np.deg2rad(-30), np.deg2rad(30), n_runs)
        signs = rng.choice([-1.0, 1.0], n_runs)

        _, is_gpu = get_backend(n_runs)
        backend_name = "GPU" if is_gpu else "CPU"

        t0 = _time.perf_counter()
        misses, _, _, hits = run_batch_vectorized(mp_bench, tp_bench, cfg, r0, lam0, gM0, gT0, signs)
        elapsed = _time.perf_counter() - t0
        print(f"  n_runs={n_runs:>5d} ({backend_name}): {elapsed:.2f}s, "
              f"hit_rate={np.sum(hits)}/{n_runs}, miss_mean={np.mean(misses):.3f}m")

    # --- Full Monte-Carlo table ---
    print(f"\n--- Monte-Carlo Batch Simulation (APN, n=50 per case) ---")
    t0 = _time.perf_counter()
    table = run_batch(
        N_values=[2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
        tau_values=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
        maneuver_types=["none", "step", "sine"],
        maneuver_gs=[3, 5, 7, 10, 12, 15, 18, 20],
        n_runs=50,
        init_r=3000.0,
    )
    elapsed = _time.perf_counter() - t0
    print(f"    Elapsed: {elapsed:.2f}s\n")
    print_table(table)

    print("\n--- Done. APN engagement plots saved as engagement_apn_*.png ---")


if __name__ == "__main__":
    main()

