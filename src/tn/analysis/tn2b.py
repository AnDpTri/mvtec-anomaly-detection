"""Biểu đồ và bảng TN2b (góc xoay của screw)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.tn.analysis.common import COLORS, SHOT_ORDER


def plot_rotation(rot, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mcol, mname in zip(axes, ["i_auroc", "aupro"], ["Image AUROC", "AUPRO@0.3"]):
        for (meth, var), g in rot.groupby(["method", "variant"]):
            s = g.groupby("shots")[mcol].agg(["mean", "std"]).reindex(SHOT_ORDER)
            ax.errorbar(range(len(SHOT_ORDER)), s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                        color=COLORS[meth], ls="-" if var == "can_goc" else "--",
                        label=f'{meth} — {"đã căn góc" if var == "can_goc" else "gốc"}')
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xticks(range(len(SHOT_ORDER)))
        ax.set_xticklabels(SHOT_ORDER)
        ax.set_xlabel("số ảnh train")
        ax.set_title(f"screw — {mname}")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def rotation_correlation(sc, shots=("1", "2", "4", "8")):
    """Spearman giữa điểm và khoảng cách góc tới ảnh train gần nhất (ảnh gốc), tách ảnh good / lỗi."""
    few = sc[(sc.variant == "goc") & sc.shots.isin(shots)]
    rows = []
    for (meth, k), g in few.groupby(["method", "shots"]):
        for lab, name in [(0, "good"), (1, "loi")]:
            h = g[g.label == lab]
            rho = np.nanmean([spearmanr(x.angle_dist, x.score)[0] for _, x in h.groupby("seed")])
            rows.append(dict(method=meth, shots=k, nhom=name, spearman_rho=round(rho, 3),
                             kc_goc_tb=round(h.angle_dist.mean(), 1)))
    return pd.DataFrame(rows).pivot_table(index=["method", "shots"], columns="nhom",
                                          values=["spearman_rho", "kc_goc_tb"])


def plot_score_vs_angle(sc, path, method="patchcore", shots="4", seed=0):
    g = sc[(sc.variant == "goc") & (sc.method == method) & (sc.shots == shots) & (sc.seed == seed)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, col, name in [(0, "tab:green", "good"), (1, "tab:red", "lỗi")]:
        h = g[g.label == lab]
        ax.scatter(h.angle_dist, h.score, s=18, c=col, alpha=.7, label=name)
    ax.set_xlabel("khoảng cách góc tới ảnh train gần nhất (°)")
    ax.set_ylabel("điểm bất thường")
    ax.set_title(f"{method}, k={shots}, seed {seed} (ảnh gốc)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def plot_alignment_check(R, path, n=8):
    """Hàng trên: ảnh gốc (góc đo được); hàng dưới: sau khi căn góc, viền đỏ = mask đã xoay theo."""
    idx = np.linspace(0, len(R["al_te"]) - 1, n).astype(int)
    fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for j, i in enumerate(idx):
        ax[0, j].imshow(R["C"]["imgs_te"][i])
        ax[0, j].set_title(f"{R['ang_te'][i]:.0f}°", fontsize=9)
        ax[1, j].imshow(R["al_te"][i])
        if R["variants"]["can_goc"][1][i].any():
            ax[1, j].contour(R["variants"]["can_goc"][1][i], colors="r", linewidths=0.8)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Trên: ảnh gốc (góc đo được) | Dưới: sau khi căn góc")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.show()
