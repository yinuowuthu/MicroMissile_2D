"""
2D Point-Mass Proportional Navigation Guidance (PNG) Simulation
===============================================================
Vectorized version: all Monte-Carlo runs advance simultaneously.
Supports GPU acceleration via CuPy (auto-selects based on batch size).

Kinematic model based on 2D relative motion equations.
Guidance law: ac = N * VM * lambda_dot  (Pure PNG)
Autopilot: first-order lag  tau * a_dot + a = ac
Target: constant velocity + random step/sinusoidal maneuvers

GPU performance notes:
  - xp.any(active) forces a GPU->CPU sync on every call.
    We therefore only check for early exit every SYNC_INTERVAL steps.
  - Inactive simulations keep running with zeroed deltas — negligible
    waste compared to the cost of per-step barriers.
  - prev_r bookkeeping is done with scalar rotation instead of two copies.
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

# GPU only pays off with enough parallel runs to saturate GPU cores.
# Below this threshold, CPU NumPy is faster due to lower overhead.
GPU_THRESHOLD = 500

# How many steps between GPU->CPU sync checks for early termination.
# Higher = fewer syncs = faster GPU, but slightly more wasted compute
# for runs that have already ended.  100 steps @ dt=0.001s = 0.1s granularity.
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
    miss_distance: float = 0.0
    flight_time: float = 0.0
    energy: float = 0.0
    hit: bool = False


# ---------------------------------------------------------------------------
# Vectorized batch simulation (all n_runs in parallel)
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
    Run n_runs simulations simultaneously using NumPy or CuPy.
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

    # Per-run tracking
    min_r      = init_r.copy()
    min_r_time = xp.zeros(n)
    energy     = xp.zeros(n)
    active     = xp.ones(n, dtype=bool)

    # Rolling range history — two slots, avoid xp.copy every step
    prev_r      = init_r.copy()   # r at step-1
    prev_prev_r = init_r.copy()   # r at step-2

    t = 0.0
    for step in range(max_steps):

        # ---- EARLY EXIT: only sync GPU every SYNC_INTERVAL steps --------
        # xp.any() triggers a GPU->CPU transfer (barrier).  Doing it every
        # step would serialise the pipeline; we pay the cost only rarely.
        if step % SYNC_INTERVAL == 0:
            if not bool(xp.any(active)):
                break

        # Relative geometry
        dx  = xT - xM
        dy  = yT - yM
        r   = xp.sqrt(dx * dx + dy * dy)
        lam = xp.arctan2(dy, dx)

        # Update min range (no sync needed — stays on device)
        closer    = active & (r < min_r)
        min_r     = xp.where(closer, r, min_r)
        min_r_time = xp.where(closer, t, min_r_time)

        # Termination flags (all on device, no sync)
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

        # Rotate range history without xp.copy (just reassign references)
        prev_prev_r = prev_r
        prev_r      = r

        # ---- Guidance & control -----------------------------------------
        r_safe  = xp.maximum(r, 1e-6)
        lam_dot = (VT * xp.sin(gamma_T - lam) - VM * xp.sin(gamma_M - lam)) / r_safe

        ac     = xp.clip(mp.N * VM * lam_dot, -mp.max_accel, mp.max_accel)
        am_new = xp.clip(am + (ac - am) * (dt / mp.tau),    -mp.max_accel, mp.max_accel)

        # Target maneuver — use scalar broadcast, avoid xp.full/xp.zeros
        if t < tp.maneuver_start or tp.maneuver_type == "none":
            aT = 0.0
        elif tp.maneuver_type == "step":
            aT = signs * a_max_tgt          # shape (n,), already on device
        else:  # "sine"
            aT = a_max_tgt * float(np.sin(two_pi_f * (t - tp.maneuver_start)))

        # State update — mask inactive runs with zero delta
        act_f   = active.astype(xp.float64)   # 1.0 / 0.0, no branch needed
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
# Single run (scalar, for plotting only)
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
        ac = np.clip(mp.N * VM * lam_dot, -mp.max_accel, mp.max_accel)
        ac_arr[i] = ac

        am_new = np.clip(am + (ac - am) / mp.tau * cfg.dt, -mp.max_accel, mp.max_accel)

        a_max = tp.maneuver_g * G
        if t < tp.maneuver_start or tp.maneuver_type == "none":
            aT = 0.0
        elif tp.maneuver_type == "step":
            aT = sign * a_max
        elif tp.maneuver_type == "sine":
            aT = a_max * np.sin(2 * np.pi * tp.maneuver_freq * (t - tp.maneuver_start))
        else:
            aT = 0.0

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
        ac_arr=ac_arr[:n], am_arr=am_arr[:n],
        miss_distance=min_r, flight_time=t_arr[min(idx_min_r, n-1)],
        energy=energy, hit=hit,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_engagement(res: SimResult, title: str = "PNG Engagement"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{title}  |  Miss={res.miss_distance:.3f}m  T={res.flight_time:.3f}s", fontsize=13)

    ax = axes[0, 0]
    ax.plot(res.xM_arr, res.yM_arr, 'b-', label='Missile')
    ax.plot(res.xT_arr, res.yT_arr, 'r--', label='Target')
    ax.plot(res.xM_arr[0], res.yM_arr[0], 'bo', ms=6)
    ax.plot(res.xT_arr[0], res.yT_arr[0], 'ro', ms=6)
    ax.plot(res.xM_arr[-1], res.yM_arr[-1], 'bx', ms=8)
    ax.plot(res.xT_arr[-1], res.yT_arr[-1], 'rx', ms=8)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory'); ax.legend()
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(res.t_arr, res.r_arr, 'k-')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Range (m)')
    ax.set_title('Missile-Target Range'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(res.t_arr, res.ac_arr / G, 'r-', label='Commanded (ac)', alpha=0.7)
    ax.plot(res.t_arr, res.am_arr / G, 'b-', label='Actual (am)', alpha=0.7)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Acceleration (g)')
    ax.set_title('Lateral Acceleration'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    dr = np.diff(res.r_arr)
    dt_arr = np.diff(res.t_arr)
    r_dot = np.zeros_like(res.r_arr)
    r_dot[:-1] = dr / np.maximum(dt_arr, 1e-10)
    t_go = res.r_arr / np.maximum(np.abs(r_dot), 1e-3)
    ax.plot(res.t_arr, t_go, 'g-')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('t_go estimate (s)')
    ax.set_title('Estimated Time-to-Go')
    ax.set_ylim(bottom=0, top=min(float(np.max(t_go))*1.2, 50)); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch Monte-Carlo (vectorized, auto GPU/CPU)
# ---------------------------------------------------------------------------
def run_batch(
    N_values: List[float] = [3, 4, 5],
    tau_values: List[float] = [0.1, 0.2, 0.5],
    maneuver_types: List[str] = ["none", "step", "sine"],
    maneuver_gs: List[float] = [5, 10, 15],
    n_runs: int = 20,
    init_r: float = 3000.0,
):
    """Run batch simulations using vectorized engine. Arrays stay as numpy;
    run_batch_vectorized handles GPU transfer internally."""
    cfg = SimConfig()
    results_table = []
    case_idx = 0

    all_cases = [
        (N_val, tau, man_type, man_g)
        for N_val, tau, man_type, man_g
        in itertools.product(N_values, tau_values, maneuver_types, maneuver_gs)
        if not (man_type == "none" and man_g != maneuver_gs[0])
    ]
    n_cases = len(all_cases)

    for N_val, tau, man_type, man_g in all_cases:
        case_idx += 1
        print(f"\r  Case {case_idx}/{n_cases} ...", end="", flush=True)
        mp = MissileParams(N=N_val, tau=tau)
        tp = TargetParams(maneuver_type=man_type, maneuver_g=man_g, maneuver_start=2.0)

        # Generate random initial conditions (numpy, on CPU)
        r0_list, lam0_list, gM0_list, gT0_list, sign_list = [], [], [], [], []
        for run_i in range(n_runs):
            rng = np.random.default_rng(run_i * 1000 + case_idx)
            r0_list.append(init_r + rng.uniform(-200, 200))
            lam0 = rng.uniform(np.deg2rad(-20), np.deg2rad(20))
            lam0_list.append(lam0)
            gM0_list.append(lam0 + rng.uniform(np.deg2rad(-5), np.deg2rad(5)))
            gT0_list.append(np.pi + lam0 + rng.uniform(np.deg2rad(-30), np.deg2rad(30)))
            sign_list.append(rng.choice([-1, 1]))

        r0_arr = np.array(r0_list)
        lam0_arr = np.array(lam0_list)
        gM0_arr = np.array(gM0_list)
        gT0_arr = np.array(gT0_list)
        sign_arr = np.array(sign_list, dtype=np.float64)

        misses, times, energies, hits = run_batch_vectorized(
            mp, tp, cfg, r0_arr, lam0_arr, gM0_arr, gT0_arr, sign_arr
        )

        n_hits = int(np.sum(hits))
        row = {
            'N': N_val,
            'tau': tau,
            'maneuver': man_type,
            'max_g': man_g if man_type != "none" else '-',
            'hit_rate': f"{n_hits}/{n_runs}",
            'miss_mean': f"{np.mean(misses):.3f}",
            'miss_std': f"{np.std(misses):.3f}",
            'miss_max': f"{np.max(misses):.3f}",
            'time_mean': f"{np.mean(times):.2f}",
            'energy_mean': f"{np.mean(energies):.1f}",
        }
        results_table.append(row)

    print(f"\r  Done ({n_cases} cases).          ")
    return results_table


def print_table(table: List[dict]):
    if not table:
        print("No results.")
        return
    headers = list(table[0].keys())
    col_widths = {h: max(len(h), max(len(str(row[h])) for row in table)) for h in headers}
    print(" | ".join(h.center(col_widths[h]) for h in headers))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for row in table:
        print(" | ".join(str(row[h]).rjust(col_widths[h]) for h in headers))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  2D PNG Simulation - Vectorized (auto GPU/CPU)")
    print("=" * 70)

    # --- Demo: single engagement plots ---
    demo_cases = [
        ("Non-maneuvering target", TargetParams(maneuver_type="none")),
        ("Step maneuver (10g)", TargetParams(maneuver_type="step", maneuver_g=10, maneuver_start=2.0)),
        ("Sine maneuver (10g, 1Hz)", TargetParams(maneuver_type="sine", maneuver_g=10, maneuver_freq=1.0, maneuver_start=1.0)),
    ]

    mp = MissileParams(N=4, tau=0.2)
    cfg = SimConfig()

    print("\n--- Single Engagement Demos ---")
    for title, tp in demo_cases:
        res = run_simulation_single(mp, tp, cfg, init_r=3000, init_lambda=np.deg2rad(10),
                                    init_gamma_M=np.deg2rad(10), init_gamma_T=np.pi + np.deg2rad(15))
        print(f"  {title}: miss={res.miss_distance:.3f}m, time={res.flight_time:.3f}s, "
              f"energy={res.energy:.1f}, hit={res.hit}")
        fig = plot_engagement(res, title)
        fig.savefig(f"engagement_{tp.maneuver_type}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)

    # --- Benchmark: CPU (n=20) vs GPU (n=1000) ---
    print("\n--- Benchmark: CPU small batch vs GPU large batch ---")

    # Small batch (CPU, n=20 per case, just one config)
    mp_bench = MissileParams(N=4, tau=0.2)
    tp_bench = TargetParams(maneuver_type="step", maneuver_g=10, maneuver_start=2.0)

    for n_runs in [20]:
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
    print(f"\n--- Monte-Carlo Batch Simulation (vectorized, n=50 per case) ---")
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

    print("\n--- Done. Engagement plots saved as engagement_*.png ---")


if __name__ == "__main__":
    main()
