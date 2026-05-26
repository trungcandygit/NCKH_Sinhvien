"""
Monte Carlo Sharpe Distribution Plot
Phong cach: Bertsimas, Gupta & Paschalidis (2012) - INFORMS
Hai subplot: BL-KIO va TAN, 5 muc delta
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("output/monte_carlo_v2.csv")
deltas   = sorted(df["Delta_Noise"].unique())   # [-0.1, -0.05, 0.0, 0.05, 0.1]

# ── Style: B&W voi marker nhu hinh mau ────────────────────────────────────
STYLES = {
    -0.10: dict(color="black",    linestyle="-",  marker="^", label=r"$\delta = -10\%$"),
    -0.05: dict(color="black",    linestyle="--", marker="o", label=r"$\delta = -5\%$"),
     0.00: dict(color="black",    linestyle="-.", marker="s", label=r"$\delta = 0$"   ),
     0.05: dict(color="dimgray",  linestyle=":",  marker="D", label=r"$\delta = +5\%$"),
     0.10: dict(color="dimgray",  linestyle="-",  marker="*", label=r"$\delta = +10\%$"),
}

MARK_EVERY = 40   # marker spacing tren duong KDE
LW         = 1.4
MS         = 5    # marker size

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.35)

PANELS = [
    ("Sharpe_BL_KIO", "BL-KIO (K-means Signal)", axes[0]),
    ("Sharpe_TAN",    "TAN (Tangency)",           axes[1]),
]

for col, title, ax in PANELS:
    for delta in deltas:
        sub  = df[df["Delta_Noise"] == delta][col].dropna().values
        kde  = gaussian_kde(sub, bw_method="scott")
        x    = np.linspace(sub.min() - 0.2, sub.max() + 0.2, 500)
        y    = kde(x)
        st   = STYLES[round(delta, 2)]

        ax.plot(x, y,
                color=st["color"], linestyle=st["linestyle"],
                linewidth=LW, label=st["label"])

        # Marker trai deu tren duong
        idx = np.linspace(0, len(x)-1, MARK_EVERY, dtype=int)
        ax.plot(x[idx], y[idx],
                color=st["color"], marker=st["marker"],
                markersize=MS, linestyle="None",
                markerfacecolor="white", markeredgewidth=1.2)

    ax.set_xlabel("Tỷ lệ Sharpe (năm hoá)", fontsize=10)
    ax.set_ylabel("Mật độ xấp xỉ", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # Vertical line: baseline (delta=0)
    base = df[df["Delta_Noise"] == 0.0][col].mean()
    ax.axvline(base, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)

# Legend chung o axes[1]
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="upper center", ncol=5,
           fontsize=8.5, frameon=True,
           bbox_to_anchor=(0.5, 1.02),
           handlelength=2.5)

fig.suptitle(
    "Hình X. Phân phối Tỷ lệ Sharpe theo mức nhiễu tín hiệu δ\n"
    "(Mô phỏng Monte Carlo, B = 2.000 lần, 5 mức δ)",
    fontsize=10, y=1.08
)

plt.tight_layout()
plt.savefig("output/monte_carlo_density.png",
            dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved: output/monte_carlo_density.png")
