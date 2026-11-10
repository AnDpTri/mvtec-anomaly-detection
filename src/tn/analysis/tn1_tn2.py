"""Biểu đồ TN1 (train lẫn ảnh lỗi) và TN2 (few-shot)."""
import math

import matplotlib.pyplot as plt

from src.tn.analysis.common import COLORS


def plot_curves(df, xcol, xlabel, cats, path, metrics, xticks=None):
    """Mỗi hàng một category, mỗi cột một metric; trung bình ± độ lệch chuẩn theo seed."""
    fig, axes = plt.subplots(len(cats), len(metrics), figsize=(4.2 * len(metrics), 3.2 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        for c, (mcol, mname) in enumerate(metrics):
            ax = axes[r, c]
            for meth, g in df[df.category == cat].groupby("method"):
                s = g.groupby(xcol)[mcol].agg(["mean", "std"]).sort_index()
                ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3, label=meth,
                            color=COLORS.get(meth))
            ax.set_title(f"{cat} — {mname}", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=.3)
            if xticks is not None:
                ax.set_xticks(xticks[0])
                ax.set_xticklabels(xticks[1])
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def shots_to_x(shots):
    """Trục log2 số ảnh; 'all' vẽ ở vị trí cuối (log2 64)."""
    return [math.log2(float(s)) if s != "all" else math.log2(64) for s in shots]
