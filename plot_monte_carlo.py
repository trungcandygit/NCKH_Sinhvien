"""
Monte Carlo Sharpe Distribution
Style: Bertsimas, Gupta & Paschalidis (2012) - Operations Research / MATLAB
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

# ── Matplotlib giong MATLAB ────────────────────────────────────────────────
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

# ── Data ───────────────────────────────────────────────────────────────────
df     = pd.read_csv("output/monte_carlo_v2.csv")
deltas = sorted(df["Delta_Noise"].unique())

# TAN la scalar moi delta → tinh excess Sharpe (BL_KIO - TAN)
tan_scalar = df.groupby("Delta_Noise")["Sharpe_TAN"].mean().to_dict()
df["Excess_Sharpe"] = df.apply(
    lambda r: r["Sharpe_BL_KIO"] - tan_scalar[r["Delta_Noise"]], axis=1
)

# ── Style cho 5 delta ─────────────────────────────────────────────────────
#   giong mau: duong dam + marker day dac
CONFIGS = {
    -0.10: dict(ls="-",  marker="^", lw=1.2, ms=4.5, label=r"$\delta\!=\!-10\%$"),
    -0.05: dict(ls="--", marker="o", lw=1.2, ms=4.5, label=r"$\delta\!=\!-5\%$" ),
     0.00: dict(ls="-",  marker="s", lw=1.6, ms=5.0, label=r"$\delta\!=\!0$"    ),
     0.05: dict(ls="--", marker="D", lw=1.2, ms=4.0, label=r"$\delta\!=\!+5\%$" ),
     0.10: dict(ls="-",  marker="*", lw=1.2, ms=6.0, label=r"$\delta\!=\!+10\%$"),
}
COLORS = {
    -0.10: "black",
    -0.05: "black",
     0.00: "black",
     0.05: "dimgray",
     0.10: "dimgray",
}

N_KDE    = 600    # so diem KDE
MARK_N   = 70     # so marker tren moi duong (day dac nhu MATLAB)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.subplots_adjust(wspace=0.38, top=0.82)

PANELS = [
    ("Sharpe_BL_KIO",  "Approximate density of BL-KIO Sharpe",
     "Tỷ lệ Sharpe (năm hoá) — BL-KIO"),
    ("Excess_Sharpe",  "Approximate density of Excess Sharpe (BL-KIO − TAN)",
     "Tỷ lệ Sharpe vượt trội so với TAN"),
]

for (col, title, xlabel), ax in zip(PANELS, axes):
    ax.grid(True, zorder=0)

    for delta in deltas:
        d   = round(float(delta), 2)
        sub = df[df["Delta_Noise"] == delta][col].dropna().values
        kde = gaussian_kde(sub, bw_method=0.28)   # bw nho = peak cao nhu MATLAB
        x   = np.linspace(sub.min() - 0.15, sub.max() + 0.15, N_KDE)
        y   = kde(x)

        cfg = CONFIGS[d]
        col_c = COLORS[d]

        # Ve duong KDE
        ax.plot(x, y,
                color=col_c, linestyle=cfg["ls"], linewidth=cfg["lw"],
                zorder=3, label="_nolegend_")

        # Marker day dac (giong MATLAB: phu kin duong)
        idx = np.round(np.linspace(20, N_KDE - 20, MARK_N)).astype(int)
        ax.plot(x[idx], y[idx],
                color=col_c, marker=cfg["marker"],
                markersize=cfg["ms"], linestyle="None",
                markerfacecolor="white", markeredgewidth=1.2,
                zorder=4, label=cfg["label"])

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mật độ xấp xỉ", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Legend trong o (upper right — y nhu hinh mau)
    handles = [h for h in ax.get_legend_handles_labels()[0]
               if not isinstance(h, matplotlib.lines.Line2D) or h.get_marker() != "None"]
    labels  = [l for h, l in zip(*ax.get_legend_handles_labels())
               if l != "_nolegend_"]
    ax.legend(handles=handles, labels=labels,
              loc="upper right", handlelength=2.0,
              handletextpad=0.5, borderpad=0.6)

fig.suptitle(
    "Hình X.  Phân phối Tỷ lệ Sharpe theo mức nhiễu tín hiệu δ\n"
    "(Monte Carlo, B = 2.000 lần lặp, 5 mức δ ∈ {−10%, −5%, 0, +5%, +10%})",
    fontsize=9.5, y=0.98
)

plt.savefig("output/monte_carlo_density.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: output/monte_carlo_density.png")
