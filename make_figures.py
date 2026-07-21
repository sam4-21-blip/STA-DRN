"""
Regenerate the presentation figures from the REAL training logs.
Run on the uni machine inside the sta_drn_conda environment:

    cd ~/STA-DRN
    python3 make_figures.py

Outputs: ablation_mae.png  and  lr_curves.png  in the current folder.
Then swap these into the PowerPoint (right-click image -> Change Picture).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ---------- palette ----------
DEEP="#065A82"; TEAL="#1C7293"; MID="#21295C"; CORAL="#F96167"; GREY="#5A6B7B"
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- helper: load a log safely ----------
def load(path):
    """Return a clean dataframe with numeric epoch/val_mae/val_rmse, or None."""
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        return None
    df = pd.read_csv(path, names=["epoch","val_mae","val_rmse"], skiprows=1)
    for c in ["epoch","val_mae","val_rmse"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def best_mae(path):
    df = load(path)
    if df is None or len(df)==0:
        return None
    return df["val_mae"].min()

# ======================================================================
# FIGURE 1 — ablation bar chart (auto-fills from whatever logs exist)
# ======================================================================
# edit these paths if your filenames differ
logs_005 = {
    "TANet":     "training_log_tanet.csv",
    "SANet":     "training_log_sanet.csv",
    "STANet":    "training_log_stanet.csv",
    "STANet-AF": "training_log_stanet_af.csv",   # may be the short 8-row one
}
logs_001 = {
    "TANet":     "training_log_tanet_lr001.csv",
    "SANet":     "training_log_sanet_lr001.csv",
    "STANet":    "training_log_stanet_lr001.csv",
    "STANet-AF": "training_log_stanet_af_lr001.csv",
}
# hard-coded paper values (AVEC 2014, k=2) for reference
paper = {"TANet":7.07, "SANet":7.30, "STANet":6.97, "STANet-AF":6.00}

models = list(paper.keys())
mae005 = [best_mae(logs_005[m]) for m in models]
mae001 = [best_mae(logs_001[m]) for m in models]
maep   = [paper[m] for m in models]

print("lr=0.005 best MAE:", mae005)
print("lr=0.001 best MAE:", mae001)

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(9.2,4.6), dpi=150)
b1 = ax.bar(x-w, [v if v else 0 for v in mae005], w, label="This work (lr=0.005)", color=TEAL)
b2 = ax.bar(x,   [v if v else 0 for v in mae001], w, label="This work (lr=0.001)", color=DEEP)
b3 = ax.bar(x+w, maep, w, label="Pan et al. (2024)", color=CORAL)

def label(vals, xs):
    for xi, v in zip(xs, vals):
        if v: ax.text(xi, v+0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#33404A")
        else: ax.text(xi, 0.3, "in\nprogress", ha="center", va="bottom", fontsize=7.5, color=GREY, style="italic")
label(mae005, x-w); label(mae001, x); label(maep, x+w)

ax.set_ylabel("Validation MAE (lower is better)", fontsize=10.5, color="#33404A")
ax.set_title("Ablation study: mean absolute error by model variant",
             fontsize=12.5, fontweight="bold", color=MID, pad=12)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10.5, color="#33404A")
top = max([v for v in mae005+mae001+maep if v]) + 1.5
ax.set_ylim(0, top)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#DEE6EC", linewidth=0.7); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("ablation_mae.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote ablation_mae.png")

# ======================================================================
# FIGURE 2 — lr comparison curves (STANet-AF: 0.005 vs 0.001), real data
# ======================================================================
d005 = load("training_log_stanet_af.csv")          # full 0.005 run if you have it
d001 = load("training_log_stanet_af_lr001.csv")     # the 0.001 run

fig, axes = plt.subplots(1,2, figsize=(10.6,4.4), dpi=150)

ax = axes[0]
if d005 is not None and len(d005) > 10:
    ax.plot(d005["epoch"], d005["val_mae"], color=CORAL, linewidth=1.1)
    peak = d005["val_mae"].max()
    ax.annotate(f"instability spike\n(MAE {peak:.0f})",
                xy=(d005.loc[d005['val_mae'].idxmax(),'epoch'], peak),
                xytext=(60, peak*0.75), fontsize=8.5, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
else:
    ax.text(0.5,0.5,"stanet_af lr=0.005 log\nnot available\n(only short run saved)",
            ha="center", va="center", transform=ax.transAxes, color=GREY, fontsize=10)
ax.set_title("lr = 0.005 (released code)", fontsize=11.5, fontweight="bold", color=MID)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color="#E4EBF0", linewidth=0.6); ax.set_axisbelow(True)

ax = axes[1]
if d001 is not None and len(d001) > 10:
    ax.plot(d001["epoch"], d001["val_mae"], color=DEEP, linewidth=1.3)
    bm = d001["val_mae"].min()
    ax.axhline(bm, color=TEAL, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(d001["epoch"].max()*0.55, bm+0.4, f"best MAE {bm:.2f}", fontsize=8.5, color=TEAL)
    ax.set_ylim(0, min(30, d001["val_mae"].quantile(0.95)*1.4))
ax.set_title("lr = 0.001 (paper value)", fontsize=11.5, fontweight="bold", color=MID)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color="#E4EBF0", linewidth=0.6); ax.set_axisbelow(True)

fig.suptitle("STA-DRN (attention fusion): effect of learning-rate on training stability",
             fontsize=12, fontweight="bold", color=MID, y=1.02)
plt.tight_layout()
plt.savefig("lr_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote lr_curves.png")
print("\nDone. Swap these two PNGs into slides 5 and 7.")