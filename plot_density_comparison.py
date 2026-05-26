"""
Figure: Static vs Monte Carlo approach comparison
Left : "Kỳ vọng tĩnh" – single delta=0 curve (what other studies do)
Right: Monte Carlo 5 delta levels (this study) – shows robustness
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

tan_scalar = df.groupby("Delta_Noise")["Sharpe_TAN"].mean().to_dict()
df["Excess_Sharpe"] = df.apply(
    lambda r: r["Sharpe_BL_KIO"] - tan_scalar[r["Delta_Noise"]], axis=1
)

N_KDE  = 600
MARK_N = 70

CONFIGS = {
    -0.10: dict(ls="-",  marker="^", lw=1.2, ms=4.5, label=r"$\delta\!=\!-10\%$"),
    -0.05: dict(ls="--", marker="o", lw=1.2, ms=4.5, label=r"$\delta\!=\!-5\%$"),
     0.00: dict(ls="-",  marker="s", lw=1.6, ms=5.0, label=r"$\delta\!=\!0$ (tĩnh)"),
     0.05: dict(ls="--", marker="D", lw=1.2, ms=4.0, label=r"$\delta\!=\!+5\%$"),
     0.10: dict(ls="-",  marker="*", lw=1.2, ms=6.0, label=r"$\delta\!=\!+10\%$"),
}
COLORS = {
    -0.10: "black",
    -0.05: "black",
     0.00: "black",
     0.05: "dimgray",
     0.10: "dimgray",
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
fig.subplots_adjust(wspace=0.40, top=0.80)

# ── Panel LEFT: chỉ delta=0 (tiếp cận kỳ vọng tĩnh duy nhất) ──────────────
ax = axes[0]
ax.grid(True, zorder=0)

sub0 = df[df["Delta_Noise"] == 0.0]["Sharpe_BL_KIO"].dropna().values
kde0 = gaussian_kde(sub0, bw_method=0.28)
x0   = np.linspace(sub0.min() - 0.15, sub0.max() + 0.15, N_KDE)
y0   = kde0(x0)

ax.plot(x0, y0, color="black", linestyle="-", linewidth=1.6, zorder=3, label="_nolegend_")
idx0 = np.round(np.linspace(15, N_KDE - 15, MARK_N)).astype(int)
ax.plot(x0[idx0], y0[idx0],
        color="black", marker="s", markersize=5.0, linestyle="None",
        markerfacecolor="white", markeredgewidth=1.2, zorder=4,
        label=r"$\delta\!=\!0$ (điểm tĩnh)")

ax.set_xlabel("Tỷ lệ Sharpe (năm hoá) — BL-KIO", fontsize=9)
ax.set_ylabel("Mật độ xấp xỉ", fontsize=9)
ax.set_title("(a)  Kỳ vọng tín hiệu tĩnh (δ = 0)\n"
             "Tiếp cận phổ biến trong các nghiên cứu trước",
             fontsize=9, pad=6)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

handles0 = [h for h in ax.get_legend_handles_labels()[0]
            if hasattr(h, 'get_marker') and h.get_marker() != "None"]
labels0  = [l for h, l in zip(*ax.get_legend_handles_labels()) if l != "_nolegend_"]
ax.legend(handles=handles0, labels=labels0,
          loc="upper right", handlelength=2.0, handletextpad=0.5, borderpad=0.6)

# Chú thích: "một phân phối hẹp — không phản ánh bất định tín hiệu"
ax.annotate("Một phân phối hẹp:\nkhông phản ánh\nbất định tín hiệu",
            xy=(x0[y0.argmax()], y0.max()), xycoords="data",
            xytext=(x0[y0.argmax()] + 0.15, y0.max() * 0.65),
            fontsize=7.5, color="dimgray",
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))

# ── Panel RIGHT: Monte Carlo 5 mức δ (tiếp cận của nghiên cứu này) ─────────
ax = axes[1]
ax.grid(True, zorder=0)

for delta in deltas:
    d   = round(float(delta), 2)
    sub = df[df["Delta_Noise"] == delta]["Sharpe_BL_KIO"].dropna().values
    kde = gaussian_kde(sub, bw_method=0.28)
    x   = np.linspace(sub.min() - 0.15, sub.max() + 0.15, N_KDE)
    y   = kde(x)

    cfg   = CONFIGS[d]
    col_c = COLORS[d]

    ax.plot(x, y, color=col_c, linestyle=cfg["ls"], linewidth=cfg["lw"],
            zorder=3, label="_nolegend_")
    idx = np.round(np.linspace(15, N_KDE - 15, MARK_N)).astype(int)
    ax.plot(x[idx], y[idx],
            color=col_c, marker=cfg["marker"],
            markersize=cfg["ms"], linestyle="None",
            markerfacecolor="white", markeredgewidth=1.2,
            zorder=4, label=cfg["label"])

ax.set_xlabel("Tỷ lệ Sharpe (năm hoá) — BL-KIO", fontsize=9)
ax.set_ylabel("Mật độ xấp xỉ", fontsize=9)
ax.set_title("(b)  Monte Carlo với 5 mức nhiễu tín hiệu δ\n"
             "Tiếp cận của nghiên cứu này (B = 2.000 lần lặp)",
             fontsize=9, pad=6)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

handles = [h for h in ax.get_legend_handles_labels()[0]
           if hasattr(h, 'get_marker') and h.get_marker() != "None"]
labels  = [l for h, l in zip(*ax.get_legend_handles_labels()) if l != "_nolegend_"]
ax.legend(handles=handles, labels=labels,
          loc="upper right", handlelength=2.0, handletextpad=0.5, borderpad=0.6)

fig.suptitle(
    "Hình X.  So sánh tiếp cận kỳ vọng tĩnh vs Monte Carlo đa mức nhiễu tín hiệu δ\n"
    "BL-KIO  ·  Monte Carlo B = 2.000  ·  δ ∈ {−10%, −5%, 0, +5%, +10%}",
    fontsize=9.5, y=0.98
)

plt.savefig("output/density_comparison.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: output/density_comparison.png")
