"""Biểu đồ và bảng TN3 (đổi loại sản phẩm, thích nghi sang bo mới)."""
import matplotlib.pyplot as plt
import numpy as np

from src.tn.analysis.common import COLORS
from src.tn.config import METHODS


def plot_matrices(df, cats, title, path):
    """Hàng = nguồn (train), cột = đích (test). Đường chéo = train và test cùng category."""
    meths = [m for m in METHODS if m in set(df.method)]
    panels = [("i_auroc", "Image AUROC", "RdYlGn", 0.3, 1.0), ("aupro", "AUPRO@0.3", "RdYlGn", 0.0, 1.0),
              ("img_fpr_thr", "FPR @ ngưỡng nguồn", "Reds", 0.0, 1.0)]
    fig, axes = plt.subplots(len(panels), len(meths), figsize=(3.6 * len(meths), 3.3 * len(panels)), squeeze=False)
    for r, (col, name, cmap, vmin, vmax) in enumerate(panels):
        for c, m in enumerate(meths):
            M = df[df.method == m].pivot_table(index="source", columns="target", values=col).reindex(index=cats, columns=cats)
            ax = axes[r, c]
            ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(len(cats)):
                for j in range(len(cats)):
                    v = M.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=30, fontsize=8)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats, fontsize=8)
            ax.set_title(f"{m} — {name}", fontsize=9)
            if c == 0:
                ax.set_ylabel("NGUỒN (train)")
            if r == len(panels) - 1:
                ax.set_xlabel("ĐÍCH (test)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()


def diagonal_summary(df):
    """Trung bình trên đường chéo (cùng category) và ngoài đường chéo."""
    s = df.groupby(["method", "same"])[["i_auroc", "aupro", "img_fpr_thr", "img_tpr_thr"]].mean().round(3)
    s.index = s.index.set_levels(["khác category", "cùng category"], level=1)
    return s


def plot_adapt(ad, mx, source, targets, ks, path):
    """Đường liền = nguồn + k ảnh đích, chấm = chỉ k ảnh đích, gạch ngang = train đầy đủ trên đích."""
    fig, axes = plt.subplots(2, len(targets), figsize=(4.6 * len(targets), 6.4), squeeze=False)
    for c, tgt in enumerate(targets):
        for r, (col, name) in enumerate([("i_auroc", "Image AUROC"), ("aupro", "AUPRO@0.3")]):
            ax = axes[r, c]
            for m in METHODS:
                if m not in set(ad.method):
                    continue
                base = mx[(mx.source == source) & (mx.target == tgt) & (mx.method == m)][col]    # k=0: chỉ nguồn
                full = mx[(mx.source == tgt) & (mx.target == tgt) & (mx.method == m)][col]
                for regime, ls in [("nguon+k", "-"), ("chi_k", ":")]:
                    g = ad[(ad.target == tgt) & (ad.method == m) & (ad.regime == regime)].groupby("k")[col].mean()
                    xs, ys = list(g.index), list(g.values)
                    if regime == "nguon+k" and len(base):
                        xs, ys = [0] + xs, [float(base.iloc[0])] + ys
                    ax.plot(xs, ys, ls=ls, marker="o", color=COLORS[m],
                            label=f'{m} — {"nguồn + k ảnh đích" if regime == "nguon+k" else "chỉ k ảnh đích"}')
                if len(full):
                    ax.axhline(float(full.iloc[0]), color=COLORS[m], lw=0.8, ls="--", alpha=.6)
            ax.set_xticks([0] + list(ks))
            ax.set_xlabel("k = số ảnh good của bo đích")
            ax.grid(alpha=.3)
            ax.set_title(f"{source} → {tgt}: {name}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Thích nghi sang bo mới: đường liền = nguồn + k ảnh đích, chấm = chỉ k ảnh đích, "
                 "gạch ngang = train đầy đủ trên đích")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()
