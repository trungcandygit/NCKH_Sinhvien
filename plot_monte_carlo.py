"""
Monte Carlo density: BL_KIO (K-means view noise) vs BL_IO (Pi noise)
3 panels: Sharpe, Return, Volatility — 5 delta levels each portfolio
Style: Bertsimas, Gupta & Paschalidis (2012) MATLAB
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
    "legend.fontsize":    7.5,
})

df     = pd.read_csv("output/monte_carlo_v2.csv")
deltas = sorted(df["Delta_Noise"].unique())

N_KDE  = 600
MARK_N = 70

# ── Style: BL_KIO = solid markers, BL_IO = dashed markers ────────────────
DELTA_CFG = {
    -0.10: dict(ls="-",  marker="^", lw=1.2, ms=4.5),
    -0.05: dict(ls="--", marker="o", lw=1.2, ms=4.5),
     0.00: dict(ls="-",  marker="s", lw=1.6, ms=5.0),
     0.05: dict(ls="--", marker="D", lw=1.2, ms=4.0),
     0.10: dict(ls="-",  marker="*", lw=1.2, ms=6.0),
}
KIO_COLORS = {-0.10:"black",   -0.05:"black",   0.00:"black",
               0.05:"dimgray",  0.10:"dimgray"}
IO_COLORS  = {-0.10:"#555555", -0.05:"#555555", 0.00:"#555555",
               0.05:"#888888",  0.10:"#888888"}

DELTA_LABELS = {
    -0.10: r"$\delta\!=\!-10\%$",
    -0.05: r"$\delta\!=\!-5\%$",
     0.00: r"$\delta\!=\!0$",
     0.05: r"$\delta\!=\!+5\%$",
     0.10: r"$\delta\!=\!+10\%$",
}


def draw_kde(ax, values, cfg, color, label):
    kde = gaussian_kde(values, bw_method=0.28)
    x   = np.linspace(values.min() - 0.05, values.max() + 0.05, N_KDE)
    y   = kde(x)
    ax.plot(x, y, color=color, linestyle=cfg["ls"], linewidth=cfg["lw"],
            zorder=3, label="_nolegend_")
    idx = np.round(np.linspace(15, N_KDE - 15, MARK_N)).astype(int)
    ax.plot(x[idx], y[idx], color=color, marker=cfg["marker"],
            markersize=cfg["ms"], linestyle="None",
            markerfacecolor="white", markeredgewidth=1.2,
            zorder=4, label=label)


def finish_ax(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mật độ xấp xỉ", fontsize=9)
    ax.set_title(title, fontsize=9, pad=5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    handles = [h for h in ax.get_legend_handles_labels()[0]
               if hasattr(h, 'get_marker') and h.get_marker() != "None"]
    labels  = [l for h, l in zip(*ax.get_legend_handles_labels())
               if l != "_nolegend_"]
    ax.legend(handles=handles, labels=labels, loc="upper right",
              handlelength=1.8, handletextpad=0.4, borderpad=0.5,
              ncol=2, columnspacing=0.8)


# ── 3-panel figure ───────────────────────────────────────────────────────────
PANELS = [
    ("Sharpe_BL_KIO", "Sharpe_BL_IO",
     "Tỷ lệ Sharpe (năm hoá)",
     "Sharpe — BL-KIO vs BL-IO"),
    ("Return_BL_KIO", "Return_BL_IO",
     "Lợi suất kỳ vọng (năm hoá)",
     "Return — BL-KIO vs BL-IO"),
    ("Vol_BL_KIO", "Vol_BL_IO",
     "Độ lệch chuẩn (năm hoá)",
     "Volatility — BL-KIO vs BL-IO"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.subplots_adjust(wspace=0.38, top=0.82)

for (col_kio, col_io, xlabel, title), ax in zip(PANELS, axes):
    ax.grid(True, zorder=0)

    for delta in deltas:
        d    = round(float(delta), 2)
        cfg  = DELTA_CFG[d]
        sub  = df[df["Delta_Noise"] == delta]

        kio_vals = sub[col_kio].dropna().values
        io_vals  = sub[col_io].dropna().values

        # BL_KIO — solid style
        draw_kde(ax, kio_vals, cfg, KIO_COLORS[d],
                 f"KIO {DELTA_LABELS[d]}")

        # BL_IO — dashed style (override ls to be more dashed)
        io_cfg = dict(cfg, ls=":")
        draw_kde(ax, io_vals,  io_cfg, IO_COLORS[d],
                 f"IO  {DELTA_LABELS[d]}")

    finish_ax(ax, xlabel, title)

fig.suptitle(
    "Hình X.  Phân phối Monte Carlo: BL-KIO (nhiễu K-means) vs BL-IO (nhiễu cân bằng Pi)\n"
    "(B = 2.000 lần lặp, 5 mức δ ∈ {−10%, −5%, 0, +5%, +10%})",
    fontsize=9.5, y=0.98
)

plt.savefig("output/monte_carlo_density.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: output/monte_carlo_density.png")
