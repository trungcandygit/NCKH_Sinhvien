"""
Figure: Static (delta=0, BL_KIO vs TAN) vs Monte Carlo multi-delta (BL_KIO)
Both panels use identical MATLAB-style KDE formatting.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "black",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "grid.linestyle":     ":",
    "grid.linewidth":     0.6,
    "grid.color":         "#aaaaaa",
    "legend.frameon":     True,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,
    "legend.fontsize":    8,
})

df     = pd.read_csv("output/monte_carlo_v2.csv")
deltas = sorted(df["Delta_Noise"].unique())

N_KDE  = 600
MARK_N = 70

# Configs cho panel (b) — 5 delta BL_KIO
MC_CONFIGS = {
    -0.10: dict(ls="-",  marker="^", lw=1.2, ms=4.5, label=r"BL-KIO  $\delta\!=\!-10\%$"),
    -0.05: dict(ls="--", marker="o", lw=1.2, ms=4.5, label=r"BL-KIO  $\delta\!=\!-5\%$"),
     0.00: dict(ls="-",  marker="s", lw=1.6, ms=5.0, label=r"BL-KIO  $\delta\!=\!0$"),
     0.05: dict(ls="--", marker="D", lw=1.2, ms=4.0, label=r"BL-KIO  $\delta\!=\!+5\%$"),
     0.10: dict(ls="-",  marker="*", lw=1.2, ms=6.0, label=r"BL-KIO  $\delta\!=\!+10\%$"),
}
MC_COLORS = {-0.10:"black", -0.05:"black", 0.00:"black", 0.05:"dimgray", 0.10:"dimgray"}

# Configs cho panel (a) — BL_KIO vs TAN tại delta=0
STATIC_CONFIGS = {
    "BL_KIO": dict(ls="-",  marker="s", lw=1.6, ms=5.0,
                   color="black",   label=r"BL-KIO  ($\delta\!=\!0$)"),
    "TAN":    dict(ls="--", marker="o", lw=1.2, ms=4.5,
                   color="dimgray", label=r"TAN  ($\delta\!=\!0$)"),
}


def draw_kde(ax, values, cfg, color):
    kde = gaussian_kde(values, bw_method=0.28)
    x   = np.linspace(values.min() - 0.15, values.max() + 0.15, N_KDE)
    y   = kde(x)
    ax.plot(x, y, color=color, linestyle=cfg["ls"], linewidth=cfg["lw"],
            zorder=3, label="_nolegend_")
    idx = np.round(np.linspace(15, N_KDE - 15, MARK_N)).astype(int)
    ax.plot(x[idx], y[idx],
            color=color, marker=cfg["marker"],
            markersize=cfg["ms"], linestyle="None",
            markerfacecolor="white", markeredgewidth=1.2,
            zorder=4, label=cfg["label"])


def finish_ax(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mật độ xấp xỉ", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    handles = [h for h in ax.get_legend_handles_labels()[0]
               if hasattr(h, 'get_marker') and h.get_marker() != "None"]
    labels  = [l for h, l in zip(*ax.get_legend_handles_labels()) if l != "_nolegend_"]
    ax.legend(handles=handles, labels=labels,
              loc="upper right", handlelength=2.0,
              handletextpad=0.5, borderpad=0.6)


fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.subplots_adjust(wspace=0.38, top=0.82)

# ── Panel (a): BL_KIO & TAN tại delta = 0 (kỳ vọng tĩnh) ─────────────────
ax = axes[0]
ax.grid(True, zorder=0)

d0 = df[df["Delta_Noise"] == 0.0]
for col, cfg_key in [("Sharpe_BL_KIO", "BL_KIO"), ("Sharpe_TAN", "TAN")]:
    vals = d0[col].dropna().values
    cfg  = STATIC_CONFIGS[cfg_key]
    draw_kde(ax, vals, cfg, cfg["color"])

finish_ax(ax,
    xlabel="Tỷ lệ Sharpe (năm hoá)",
    title="Approximate density — fixed signal (δ = 0)\n(BL-KIO vs TAN)")

# ── Panel (b): BL_KIO Monte Carlo 5 mức delta ─────────────────────────────
ax = axes[1]
ax.grid(True, zorder=0)

for delta in deltas:
    d   = round(float(delta), 2)
    sub = df[df["Delta_Noise"] == delta]["Sharpe_BL_KIO"].dropna().values
    draw_kde(ax, sub, MC_CONFIGS[d], MC_COLORS[d])

finish_ax(ax,
    xlabel="Tỷ lệ Sharpe (năm hoá) — BL-KIO",
    title="Approximate density — Monte Carlo (5 mức δ)\n(BL-KIO)")

fig.suptitle(
    "Hình X.  Phân phối Tỷ lệ Sharpe: kỳ vọng tín hiệu tĩnh vs Monte Carlo đa mức δ\n"
    "(B = 2.000 lần lặp, δ ∈ {−10%, −5%, 0, +5%, +10%})",
    fontsize=9.5, y=0.98
)

plt.savefig("output/density_comparison.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: output/density_comparison.png")
