"""
重新设计的7个图：PNG vs PPO 科学对比
========================================
核心思想：
- PNG: 展示参数敏感性（N和τ如何影响性能）
- PPO: 固定配置（N=4, τ=0.2），展示在不同目标机动下的鲁棒性
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体和样式（与DrawPNG.py一致）
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Fig 1: PNG命中率热力图 (N × τ) + PPO性能条
# ──────────────────────────────────────────────
def fig1_hitrate_heatmap(df_png, df_ppo):
    """2行：step/sine，每行3列：PNG热力图 + PPO条形图"""
    fig = plt.figure(figsize=(14, 9))

    N_vals = sorted(df_png["N"].unique())
    tau_vals = sorted(df_png["tau"].unique())

    for row, maneuver in enumerate(["step", "sine"]):
        # PNG热力图
        ax1 = plt.subplot(2, 3, row*3 + 1)
        sub_png = df_png[df_png["maneuver"] == maneuver]

        matrix = np.zeros((len(tau_vals), len(N_vals)))
        for i, tau in enumerate(tau_vals):
            for j, N in enumerate(N_vals):
                val = sub_png[(sub_png["N"] == N) & (sub_png["tau"] == tau)]["hit_rate"].mean()
                matrix[i, j] = val

        im = ax1.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax1.set_xticks(range(len(N_vals)))
        ax1.set_xticklabels([f"{n:.1f}" for n in N_vals])
        ax1.set_yticks(range(len(tau_vals)))
        ax1.set_yticklabels([f"{t:.2f}" for t in tau_vals])
        ax1.set_ylabel(r"$\tau$ (s)")
        ax1.set_xlabel("$N$")
        ax1.set_title(f"PNG - {maneuver.capitalize()}")

        # 添加数值标注
        for i in range(len(tau_vals)):
            for j in range(len(N_vals)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax1.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.4 else "black")

        # PPO性能条（固定N=4, τ=0.2，只看不同max_g）
        ax2 = plt.subplot(2, 3, row*3 + 2)
        sub_ppo = df_ppo[(df_ppo["maneuver"] == maneuver) &
                         (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
        sub_ppo = sub_ppo.sort_values("max_g")

        colors = ['#2ecc71' if hr >= 0.9 else '#f39c12' if hr >= 0.5 else '#e74c3c'
                  for hr in sub_ppo["hit_rate"]]
        ax2.barh(range(len(sub_ppo)), sub_ppo["hit_rate"], color=colors, alpha=0.8)
        ax2.set_yticks(range(len(sub_ppo)))
        ax2.set_yticklabels([f"{int(g)}g" for g in sub_ppo["max_g"]])
        ax2.set_xlabel("Hit Rate")
        ax2.set_xlim(0, 1.05)
        ax2.axvline(0.5, ls="--", lw=0.8, color="grey", alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_title(f"PPO - {maneuver.capitalize()}\n($N$=4, $\\tau$=0.2)")

        # 添加数值标注
        for i, hr in enumerate(sub_ppo["hit_rate"]):
            ax2.text(hr + 0.02, i, f"{hr:.0%}", va='center', fontsize=8)

        # PNG vs PPO直接对比（固定N=4, τ=0.2）
        ax3 = plt.subplot(2, 3, row*3 + 3)
        sub_png_fixed = df_png[(df_png["maneuver"] == maneuver) &
                               (df_png["N"] == 4) & (df_png["tau"] == 0.2)]
        sub_png_fixed = sub_png_fixed.sort_values("max_g")

        x = np.arange(len(sub_png_fixed))
        width = 0.35
        ax3.bar(x - width/2, sub_png_fixed["hit_rate"], width,
                label='PNG', color='#3498db', alpha=0.8)
        ax3.bar(x + width/2, sub_ppo["hit_rate"], width,
                label='PPO', color='#e74c3c', alpha=0.8)

        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{int(g)}g" for g in sub_png_fixed["max_g"]], rotation=45)
        ax3.set_ylabel("Hit Rate")
        ax3.set_ylim(0, 1.05)
        ax3.axhline(0.5, ls="--", lw=0.8, color="grey", alpha=0.5)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_title(f"PNG vs PPO - {maneuver.capitalize()}\n($N$=4, $\\tau$=0.2)")

    plt.colorbar(im, ax=fig.axes[:2], shrink=0.8, label="Hit Rate", pad=0.02)
    fig.suptitle("Fig 1. Hit Rate Analysis: PNG Parameter Sensitivity vs PPO Robustness",
                 fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_hitrate_heatmap.png"))
    plt.close(fig)
    print("  -> fig1_hitrate_heatmap.png")


# ──────────────────────────────────────────────
# 2. Fig 2: 命中率 vs max_g — PNG多参数 vs PPO单曲线
# ──────────────────────────────────────────────
def fig2_hitrate_vs_g(df_png, df_ppo):
    """1×2子图：step/sine"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    tau_sel = 0.2
    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]
    colors_png = plt.cm.viridis(np.linspace(0.1, 0.9, 8))

    for col, maneuver in enumerate(["step", "sine"]):
        ax = axes[col]

        # PNG多条曲线（不同N）
        sub_png = df_png[(df_png["maneuver"] == maneuver) & (df_png["tau"] == tau_sel)]
        for idx, N in enumerate(sorted(sub_png["N"].unique())):
            s = sub_png[sub_png["N"] == N].sort_values("max_g")
            ax.plot(s["max_g"], s["hit_rate"],
                    marker=markers[idx % len(markers)],
                    color=colors_png[idx], ms=5, lw=1.2, alpha=0.7,
                    label=f"PNG $N$={N}")

        # PPO单条曲线（固定N=4, τ=0.2）
        sub_ppo = df_ppo[(df_ppo["maneuver"] == maneuver) &
                         (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
        sub_ppo = sub_ppo.sort_values("max_g")
        ax.plot(sub_ppo["max_g"], sub_ppo["hit_rate"],
                marker="*", color="#e74c3c", ms=10, lw=2.5,
                label="PPO ($N$=4, $\\tau$=0.2)", zorder=10)

        ax.set_xlabel("Target Maneuver $a_T$ (g)")
        ax.set_title(f"{maneuver.capitalize()} maneuver, $\\tau$={tau_sel}s")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, ls="--", lw=0.6, color="grey", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", ncol=1, fontsize=8)

    axes[0].set_ylabel("Hit Rate")
    fig.suptitle("Fig 2. Hit Rate vs Target Maneuver: PNG Sensitivity vs PPO Robustness", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_hitrate_vs_g.png"))
    plt.close(fig)
    print("  -> fig2_hitrate_vs_g.png")


# ──────────────────────────────────────────────
# 3. Fig 3: 脱靶量 vs τ — PNG参数扫描 vs PPO固定性能
# ──────────────────────────────────────────────
def fig3_miss_vs_tau(df_png, df_ppo):
    """1×2子图：step/sine"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    N_sel = 4
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, 8))

    for col, maneuver in enumerate(["step", "sine"]):
        ax = axes[col]

        # PNG多条曲线（不同max_g，扫描τ）
        sub_png = df_png[(df_png["maneuver"] == maneuver) & (df_png["N"] == N_sel)]
        for idx, g in enumerate(sorted(sub_png["max_g"].unique())):
            s = sub_png[sub_png["max_g"] == g].sort_values("tau")
            if len(s) > 0:
                ax.errorbar(s["tau"], s["miss_mean"], yerr=s["miss_std"],
                            marker="o", ms=4, lw=1, capsize=2.5, alpha=0.7,
                            color=colors[idx], label=f"PNG {int(g)}g")

        # PPO水平线（固定τ=0.2，不同max_g）
        sub_ppo = df_ppo[(df_ppo["maneuver"] == maneuver) &
                         (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
        for idx, g in enumerate(sorted(sub_ppo["max_g"].unique())):
            s = sub_ppo[sub_ppo["max_g"] == g]
            if len(s) > 0:
                miss = s["miss_mean"].values[0]
                ax.axhline(miss, ls="--", lw=1.5, color=colors[idx], alpha=0.8)
                ax.text(0.65, miss * 1.15, f"PPO {int(g)}g", fontsize=7,
                       color=colors[idx], weight='bold')

        ax.set_xlabel(r"Guidance Lag $\tau$ (s)")
        ax.set_title(f"{maneuver.capitalize()}, $N$={N_sel}")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", ncol=2, fontsize=7)

    axes[0].set_ylabel("Miss Distance (m)")
    fig.suptitle("Fig 3. Miss Distance vs $\\tau$: PNG Sensitivity vs PPO Fixed Performance ($N$=4)",
                 y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_miss_vs_tau.png"))
    plt.close(fig)
    print("  -> fig3_miss_vs_tau.png")


# ──────────────────────────────────────────────
# 4. Fig 4: 能量-脱靶量气泡图 — PNG vs PPO
# ──────────────────────────────────────────────
def fig4_energy_bubble(df_png, df_ppo):
    """1×2子图：PNG vs PPO"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tau_sel = 0.2

    for col, (df, title) in enumerate([(df_png, "PNG"), (df_ppo, "PPO")]):
        ax = axes[col]
        sub = df[(df["maneuver"] != "none") & (df["tau"] == tau_sel)]

        if title == "PNG":
            # PNG: 气泡大小=能量，颜色=N
            for maneuver, marker, cmap_name in [
                ("step", "o", "Reds"), ("sine", "s", "Blues")
            ]:
                s = sub[sub["maneuver"] == maneuver]
                if len(s) > 0:
                    sc = ax.scatter(
                        s["max_g"], s["miss_mean"],
                        s=s["energy_mean"] / 80,
                        c=s["N"], cmap=cmap_name, marker=marker,
                        alpha=0.7, edgecolors="k", linewidths=0.3,
                        label=maneuver, vmin=2.5, vmax=6)
        else:
            # PPO: 固定N=4, τ=0.2
            sub = sub[(sub["N"] == 4)]
            for maneuver, marker, color in [
                ("step", "o", "#e74c3c"), ("sine", "s", "#3498db")
            ]:
                s = sub[sub["maneuver"] == maneuver]
                if len(s) > 0:
                    ax.scatter(
                        s["max_g"], s["miss_mean"],
                        s=s["energy_mean"] / 80,
                        marker=marker, color=color,
                        alpha=0.8, edgecolors="k", linewidths=0.5,
                        label=maneuver)

        ax.set_xlabel("Target Maneuver $a_T$ (g)")
        ax.set_ylabel("Mean Miss Distance (m)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"{title}\n(bubble size ∝ energy, $\\tau$={tau_sel}s)")

    fig.suptitle("Fig 4. Energy-Miss Trade-off (PNG: color=$N$, PPO: $N$=4 fixed)", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_energy_bubble.png"))
    plt.close(fig)
    print("  -> fig4_energy_bubble.png")


# ──────────────────────────────────────────────
# 5. Fig 5: PNG临界边界 + PPO性能区域
# ──────────────────────────────────────────────
def fig5_critical_boundary(df_png, df_ppo):
    """1×2子图：step/sine"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    N_vals = sorted(df_png["N"].unique())
    g_vals = sorted(df_png[df_png["maneuver"] == "step"]["max_g"].unique())

    for col, maneuver in enumerate(["step", "sine"]):
        ax = axes[col]

        # PNG等高线（平均所有tau）
        sub_png = df_png[df_png["maneuver"] == maneuver]
        matrix = np.zeros((len(g_vals), len(N_vals)))
        for i, g in enumerate(g_vals):
            for j, N in enumerate(N_vals):
                val = sub_png[(sub_png["N"] == N) & (sub_png["max_g"] == g)]["hit_rate"].mean()
                matrix[i, j] = val

        X, Y = np.meshgrid(N_vals, g_vals)
        cs = ax.contourf(X, Y, matrix, levels=np.linspace(0, 1, 11),
                         cmap="RdYlGn", alpha=0.6)
        ct = ax.contour(X, Y, matrix, levels=[0.5], colors="black",
                        linewidths=2, linestyles="--")
        ax.clabel(ct, inline=True, fontsize=9, fmt="%.0f%%")

        # PPO性能点（N=4固定）
        sub_ppo = df_ppo[(df_ppo["maneuver"] == maneuver) &
                         (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
        sub_ppo = sub_ppo.sort_values("max_g")

        for _, row in sub_ppo.iterrows():
            g = row["max_g"]
            hr = row["hit_rate"]
            color = '#2ecc71' if hr >= 0.9 else '#f39c12' if hr >= 0.5 else '#e74c3c'
            ax.scatter(4, g, s=200, c=color, marker="*",
                      edgecolors="black", linewidths=2, zorder=10)
            ax.text(4.3, g, f"{hr:.0%}", fontsize=8, weight='bold')

        ax.set_xlabel("$N$")
        ax.set_ylabel("Target Maneuver $a_T$ (g)")
        ax.set_title(f"{maneuver.capitalize()}\n(★ = PPO at $N$=4)")
        ax.grid(alpha=0.3)

    fig.colorbar(cs, ax=axes, shrink=0.8, label="Hit Rate (PNG)")
    fig.suptitle("Fig 5. Critical Boundary (50% Hit Rate): PNG Contour + PPO Performance", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_critical_tau.png"))
    plt.close(fig)
    print("  -> fig5_critical_tau.png")


# ──────────────────────────────────────────────
# 6. Fig 6: 无机动场景 — PNG参数敏感性 vs PPO稳定性
# ──────────────────────────────────────────────
def fig6_no_maneuver(df_png, df_ppo):
    """PNG热力图 + PPO单点"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    N_vals = sorted(df_png["N"].unique())
    tau_vals = sorted(df_png["tau"].unique())

    # PNG热力图
    sub_png = df_png[df_png["maneuver"] == "none"]
    matrix = np.zeros((len(tau_vals), len(N_vals)))
    for i, tau in enumerate(tau_vals):
        for j, N in enumerate(N_vals):
            val = sub_png[(sub_png["N"] == N) & (sub_png["tau"] == tau)]["miss_mean"].mean()
            matrix[i, j] = val

    im = ax.imshow(matrix, cmap="YlOrRd_r", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(N_vals)))
    ax.set_xticklabels([f"{n:.1f}" for n in N_vals])
    ax.set_yticks(range(len(tau_vals)))
    ax.set_yticklabels([f"{t:.2f}" for t in tau_vals])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$\tau$ (s)")
    ax.set_title("PNG Miss Distance (No Maneuver)")

    # 添加PNG数值标注
    for i in range(len(tau_vals)):
        for j in range(len(N_vals)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 1 else "black")

    # PPO性能点（N=4, τ=0.2）
    sub_ppo = df_ppo[(df_ppo["maneuver"] == "none") &
                     (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
    if len(sub_ppo) > 0:
        ppo_miss = sub_ppo["miss_mean"].values[0]
        # 找到N=4, τ=0.2的位置
        n_idx = list(N_vals).index(4)
        tau_idx = list(tau_vals).index(0.2)
        ax.scatter(n_idx, tau_idx, s=500, c='none', marker="*",
                  edgecolors="blue", linewidths=3, zorder=10)
        ax.text(n_idx + 0.5, tau_idx, f"PPO: {ppo_miss:.2f}m",
               fontsize=10, weight='bold', color='blue')

    plt.colorbar(im, ax=ax, label="Miss Distance (m)")
    fig.suptitle("Fig 6. No Maneuver: PNG Parameter Sensitivity (★ = PPO)", y=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig6_no_maneuver.png"))
    plt.close(fig)
    print("  -> fig6_no_maneuver.png")


# ──────────────────────────────────────────────
# 7. Fig 7: Step vs Sine 综合对比 — PNG vs PPO
# ──────────────────────────────────────────────
def fig7_step_vs_sine_summary(df_png, df_ppo):
    """PNG多条曲线 vs PPO两个点"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    taus = sorted(df_png["tau"].unique())

    # PNG曲线（N=4，不同τ）
    for maneuver, color, marker, ls in [
        ("step", "#e41a1c", "o", "-"),
        ("sine", "#377eb8", "s", "-")
    ]:
        sub = df_png[(df_png["maneuver"] == maneuver) & (df_png["N"] == 4)]
        mean_hr = sub.groupby("tau")["hit_rate"].mean()
        mean_hr = mean_hr.reindex(taus)
        ax.plot(taus, mean_hr, marker=marker, color=color, ls=ls,
                lw=1.5, ms=6, alpha=0.7, label=f"PNG {maneuver}")

    # PPO点（τ=0.2固定）
    for maneuver, color, marker in [
        ("step", "#e41a1c", "*"),
        ("sine", "#377eb8", "*")
    ]:
        sub = df_ppo[(df_ppo["maneuver"] == maneuver) &
                     (df_ppo["N"] == 4) & (df_ppo["tau"] == 0.2)]
        mean_hr = sub["hit_rate"].mean()
        ax.scatter(0.2, mean_hr, s=300, marker=marker, color=color,
                  edgecolors="black", linewidths=2, zorder=10,
                  label=f"PPO {maneuver}")

    ax.set_xlabel(r"Guidance Lag $\tau$ (s)")
    ax.set_ylabel("Hit Rate (averaged over all $a_T$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_title("Fig 7. Step vs Sine: PNG $\\tau$ Sensitivity vs PPO Fixed Performance ($N$=4)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig7_step_vs_sine.png"))
    plt.close(fig)
    print("  -> fig7_step_vs_sine.png")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("  Drawing 7 Figures (PNG vs PPO) — Scientific Comparison")
    print("="*70)

    csv_path = os.path.join(OUTPUT_DIR, "monte_carlo_full_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit(1)

    df = pd.read_csv(csv_path)

    # 转换宽格式为长格式
    df_png = df[["N", "tau", "maneuver", "max_g"]].copy()
    df_png["hit_rate"] = df["png_hit_rate"]
    df_png["miss_mean"] = df["png_miss_mean"]
    df_png["miss_std"] = df["png_miss_std"]
    df_png["time_mean"] = df["png_time_mean"]
    df_png["energy_mean"] = df["png_energy_mean"]

    df_ppo = df[["N", "tau", "maneuver", "max_g"]].copy()
    df_ppo["hit_rate"] = df["ppo_hit_rate"]
    df_ppo["miss_mean"] = df["ppo_miss_mean"]
    df_ppo["miss_std"] = df["ppo_miss_std"]
    df_ppo["time_mean"] = df["ppo_time_mean"]
    df_ppo["energy_mean"] = df["ppo_energy_mean"]

    df_png["max_g"] = pd.to_numeric(df_png["max_g"], errors="coerce")
    df_ppo["max_g"] = pd.to_numeric(df_ppo["max_g"], errors="coerce")

    print(f"\nLoaded {len(df_png)} PNG records, {len(df_ppo)} PPO records")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 生成7个图
    fig1_hitrate_heatmap(df_png, df_ppo)
    fig2_hitrate_vs_g(df_png, df_ppo)
    fig3_miss_vs_tau(df_png, df_ppo)
    fig4_energy_bubble(df_png, df_ppo)
    fig5_critical_boundary(df_png, df_ppo)
    fig6_no_maneuver(df_png, df_ppo)
    fig7_step_vs_sine_summary(df_png, df_ppo)

    print("\n" + "="*70)
    print("  All 7 figures saved!")
    print("="*70)
