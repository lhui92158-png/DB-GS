# -*- coding: utf-8 -*-
"""DB-GS 论文最终图（matplotlib，英文标注，Okabe-Ito 色盲安全色板，300dpi PNG + 矢量 PDF）。
图1=方法框架 图2=显存模型 图3=消融 图4=Pareto 图5=多场景显存 图6=30K收敛 图7=方差。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = r"D:\DSH_Project\AI_Projects\figs"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colorblind-safe palette (on white)
BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"; VERMILLION = "#D55E00"
PURPLE = "#CC79A7"; SKY = "#56B4E9"; GRAY = "#999999"; BLACK = "#111111"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "savefig.facecolor": "white",
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    print(f"SAVED {name}")
    plt.close(fig)

# =====================================================================
# 图1 方法框架（框图）
# =====================================================================
def fig1_method():
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    def box(x, y, w, h, text, fc, tc="white", fs=10, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                    linewidth=1.4, edgecolor=BLACK, facecolor=fc, linestyle=ls))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", linespacing=1.5)
    def arrow(x0, y0, x1, y1, color=BLACK, lw=1.8, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                    mutation_scale=16, linewidth=lw, color=color, linestyle=ls))
    # 输入（上）：多视图 → 单目深度
    box(2, 43, 17, 9, "Multi-view\nimages", GRAY)
    box(25, 43, 22, 9, "Depth Anything V2\n(monocular depth)", BLUE, fs=9)
    arrow(19, 47.5, 25, 47.5)
    # 输入（下）：SfM → 初始化
    box(2, 29, 17, 9, "SfM point\ncloud", GRAY)
    box(25, 29, 22, 9, "3DGS\ninitialization", GRAY)
    arrow(19, 33.5, 25, 33.5)
    # 训练循环大框
    ax.add_patch(Rectangle((53, 15), 45, 38, fill=False, lw=1.8, edgecolor=BLACK))
    ax.text(75.5, 50.5, "Training loop", ha="center", fontsize=11, fontweight="bold")
    box(55, 40, 18, 8, "Render +\nloss", GRAY)
    box(55, 29, 18, 8, "DAG: depth-\naware gating", GREEN, fs=8.5)
    box(55, 18, 18, 8, "Densification", GRAY)
    box(78, 29, 18, 8, "BGP: budget-\nguided pruning", VERMILLION, fs=8.5)
    box(78, 18, 18, 8, "LDB: Lagrangian\ndual budget", PURPLE, fs=8.5)
    # 训练循环内箭头（垂直链 + 分支）
    arrow(64, 40, 64, 37)
    arrow(64, 29, 64, 26)
    arrow(73, 22, 78, 22)
    arrow(73, 33, 78, 33)
    arrow(78, 26, 78, 22)
    # 输入 → 训练循环
    arrow(47, 47.5, 53, 47.5)
    arrow(47, 33.5, 53, 33.5)
    # 底部：失败尝试（虚线）+ 输出
    box(2, 3, 24, 9, "Failed attempt:\ndepth init (M1)", VERMILLION, fs=8.5, ls="--")
    box(30, 3, 36, 9, "Output: memory-bounded 3DGS\n(within budget, trainable on 4 GB)", GREEN, fs=8.5)
    arrow(14, 29, 14, 12, VERMILLION, ls="--")
    arrow(64, 15, 48, 12)
    save(fig, "fig1_method")

# =====================================================================
# 图2 显存模型（散点 + 拟合 + 饼图）
# =====================================================================
def fig2_vram():
    fig = plt.figure(figsize=(9.6, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0])
    # 11 场景 (N 百万, V GB)
    Ns = np.array([0.469, 0.653, 0.666, 0.749, 0.817, 0.904, 0.909, 1.052, 1.270, 1.300, 1.760])
    Vs = np.array([1.303, 1.855, 1.715, 2.049, 2.195, 2.115, 2.498, 2.842, 3.477, 3.467, 4.773])
    ax.scatter(Ns, Vs, s=64, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3, label="11 scenes (baseline)")
    xf = np.linspace(0.35, 1.85, 10)
    ax.plot(xf, 2.706 * xf, color=ORANGE, lw=2.4, zorder=2,
            label=r"fit: $V = 2.706\times10^{-6}\,N$  ($R^2=0.987$)")
    ax.axhline(4.0, color=VERMILLION, lw=2.0, ls="--", zorder=1)
    ax.text(0.36, 4.12, "4 GB limit", color=VERMILLION, fontsize=11, fontweight="bold")
    gi = np.argmax(Vs)
    ax.annotate("garden\n4.77 GB", xy=(Ns[gi], Vs[gi]), xytext=(1.05, 4.55),
                fontsize=10, fontweight="bold", color=VERMILLION,
                arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=1.2))
    ax.set_xlim(0.35, 1.90); ax.set_ylim(0, 5.3)
    ax.set_xlabel("Gaussian count $N$ (×10$^6$)"); ax.set_ylabel("Peak GPU memory (GB)")
    ax.legend(loc="upper left", framealpha=0.95)
    # 饼图
    axp = fig.add_subplot(gs[1])
    labels = ["Rasterization\nbuffers", "Adam\nstates", "Parameters\n$\\Theta$", "Gradients"]
    fracs = [0.675, 0.162, 0.081, 0.081]
    cols = [GRAY, ORANGE, BLUE, GREEN]
    wedges, _ = axp.pie(fracs, colors=cols, startangle=90, counterclock=False,
                        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axp.legend(wedges, [f"{l}: {f*100:.1f}%" for l, f in zip(labels, fracs)],
               loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    axp.set_title("Per-Gaussian\n2,900 B breakdown", fontsize=12, fontweight="bold")
    save(fig, "fig2_vram_model")

# =====================================================================
# 图3 消融（散点，点数 vs PSNR）
# =====================================================================
def fig3_ablation():
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    pts = [
        ("3DGS baseline", 1.048564, 23.50, GRAY, "o"),
        ("Depth init (M1)", 0.831262, 21.48, VERMILLION, "s"),
        ("DAG", 0.820125, 23.47, GREEN, "^"),
        ("BGP", 0.269992, 21.76, BLUE, "D"),
        ("DAG+BGP", 0.269997, 21.06, PURPLE, "X"),
        ("Depth init+BGP", 0.269996, 20.97, ORANGE, "P"),
    ]
    # 显式标签坐标（避免重叠）
    labelpos = {
        "3DGS baseline": (0.90, 23.72),
        "Depth init (M1)": (0.50, 20.52),
        "DAG": (0.58, 23.62),
        "BGP": (0.46, 22.00),
        "DAG+BGP": (0.46, 20.85),
        "Depth init+BGP": (0.46, 20.42),
    }
    for name, n, p, c, m in pts:
        ax.scatter(n, p, s=120, color=c, marker=m, edgecolor="white", linewidth=1.3, zorder=3)
        lx, ly = labelpos[name]
        ax.annotate(name, (n, p), xytext=(lx, ly), fontsize=10, fontweight="bold", color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.55))
    ax.set_xlabel("Gaussian count (×10$^6$)"); ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(0.15, 1.25); ax.set_ylim(20.2, 24.1)
    save(fig, "fig3_ablation")

# =====================================================================
# 图4 Pareto 前沿（硬预算 vs 对偶预算）
# =====================================================================
def fig4_pareto():
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    hard = np.array([(0.179998, 21.213), (0.269992, 21.762), (0.359995, 22.166), (0.449985, 22.461)])
    dual = np.array([(0.261810, 21.828), (0.395825, 22.305), (0.480228, 22.561)])
    ax.plot(hard[:, 0], hard[:, 1], "-o", color=BLUE, lw=2.2, ms=8, label="BGP (hard budget)")
    ax.plot(dual[:, 0], dual[:, 1], "-s", color=GREEN, lw=2.2, ms=8, label="LDB (dual budget)")
    ax.scatter([1.048564], [23.50], color=GRAY, marker="D", s=110, edgecolor="white",
               linewidth=1.2, zorder=4, label="3DGS baseline")
    ax.annotate("baseline", (1.048564, 23.50), xytext=(0.82, 23.62),
                fontsize=10, fontweight="bold", color=GRAY)
    ax.set_xlabel("Gaussian count (×10$^6$)"); ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(0.10, 1.20); ax.set_ylim(20.8, 24.0)
    ax.legend(loc="lower right", framealpha=0.95)
    save(fig, "fig4_pareto")

# =====================================================================
# 图5 多场景显存（分组柱状）
# =====================================================================
def fig5_multiscene():
    scenes = ["train", "counter", "room", "playroom", "drjohnson", "bonsai",
              "kitchen", "truck", "bicycle", "stump", "garden"]
    base = [1.303, 1.855, 1.715, 2.049, 2.195, 2.115, 2.498, 2.842, 3.477, 3.467, 4.773]
    Ns = [0.469, 0.653, 0.666, 0.749, 0.817, 0.904, 0.909, 1.052, 1.270, 1.300, 1.760]
    measured = {"train": 0.916, "room": 1.039, "garden": 2.691}
    bud = [measured.get(s, 2.706 * 0.5 * Ns[i]) for i, s in enumerate(scenes)]
    x = np.arange(len(scenes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 4.4), layout="constrained")
    ax.bar(x - w/2, base, w, color=GRAY, edgecolor="white", lw=0.6, label="3DGS baseline")
    mcol = [BLUE if s in measured else SKY for s in scenes]
    ax.bar(x + w/2, bud, w, color=mcol, edgecolor="white", lw=0.6, label="50% budget")
    ax.axhline(4.0, color=VERMILLION, lw=2.0, ls="--")
    ax.text(10.45, 4.12, "4 GB", color=VERMILLION, fontsize=11, fontweight="bold", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Peak GPU memory (GB)"); ax.set_ylim(0, 5.3)
    # 图例（区分实测 vs 预测）
    h1 = Rectangle((0, 0), 1, 1, color=GRAY)
    h2 = Rectangle((0, 0), 1, 1, color=BLUE)
    h3 = Rectangle((0, 0), 1, 1, color=SKY)
    ax.legend([h1, h2, h3],
              ["3DGS baseline", "50% budget (measured)", "50% budget (predicted, V=2.706·0.5N)"],
              loc="upper left", fontsize=10, framealpha=0.95)
    save(fig, "fig5_multiscene_vram")

# =====================================================================
# 图6 30K 收敛（峰值显存 + 墙钟时间）
# =====================================================================
def fig6_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), layout="constrained")
    cats = ["baseline\n(thrashed)", "50%\nbudget"]
    # 左：峰值显存
    ax = axes[0]
    ax.bar(cats, [3.80, 1.61], width=0.55, color=[GRAY, BLUE], edgecolor="white")
    ax.text(0, 3.90, "3.80", ha="center", fontweight="bold", color=VERMILLION, fontsize=11)
    ax.text(1, 1.71, "1.61", ha="center", fontweight="bold", color=BLUE, fontsize=11)
    ax.set_ylabel("Peak GPU memory (GB)"); ax.set_ylim(0, 4.5)
    # 右：墙钟时间
    ax = axes[1]
    ax.bar(cats, [90, 78], width=0.55, color=[GRAY, BLUE], edgecolor="white")
    ax.text(0, 95, "≈90\n(32%)", ha="center", fontweight="bold", color=VERMILLION, fontsize=10)
    ax.text(1, 83, "78\n(100%)", ha="center", fontweight="bold", color=BLUE, fontsize=10)
    ax.set_ylabel("Wall-clock time (min)"); ax.set_ylim(0, 115)
    save(fig, "fig6_convergence")

# =====================================================================
# 图7 方差（点数 + PSNR 误差棒，n=3）
# =====================================================================
def fig7_variance():
    methods = ["Baseline", "DAG", "BGP", "LDB"]
    colors = [GRAY, GREEN, BLUE, VERMILLION]
    pm = np.array([1.055604, 0.810875, 0.269999, 0.316573]); ps = np.array([0.002709, 0.011831, 0.000001, 0.022457])
    qm = np.array([23.522, 23.487, 21.765, 21.945]); qs = np.array([0.009, 0.019, 0.008, 0.029])
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), layout="constrained")
    ax = axes[0]
    ax.bar(x, pm, 0.6, color=colors, edgecolor="white")
    ax.errorbar(x, pm, yerr=ps, fmt="none", ecolor=BLACK, elinewidth=1.6, capsize=5, capthick=1.6)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Gaussian count (×10$^6$)"); ax.set_ylim(0, 1.3)
    ax = axes[1]
    ax.bar(x, qm, 0.6, color=colors, edgecolor="white")
    ax.errorbar(x, qm, yerr=qs, fmt="none", ecolor=BLACK, elinewidth=1.6, capsize=5, capthick=1.6)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("PSNR (dB)"); ax.set_ylim(20.4, 24.2)
    fig.text(0.5, 1.005, "Mean ± SD (n = 3 seeds)", ha="center", fontsize=11, fontweight="bold")
    save(fig, "fig7_variance")

for fn in (fig1_method, fig2_vram, fig3_ablation, fig4_pareto, fig5_multiscene, fig6_convergence, fig7_variance):
    try:
        fn()
    except Exception as e:
        import traceback; print(f"[FAIL] {fn.__name__}: {e}"); traceback.print_exc()
print("FIGURES_MPL_DONE")
