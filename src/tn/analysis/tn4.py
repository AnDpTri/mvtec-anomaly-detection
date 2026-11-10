"""Biểu đồ và bảng TN4 (domain shift: ánh sáng MVTec AD 2, shift nhân tạo)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tn import data as D
from src.tn.analysis.common import COLORS, COND_ORDER, mask_overlay, overlay
from src.tn.config import METHODS


def plot_conditions(a2, cat, path):
    """TN4a: AUROC (không cần ngưỡng) so với FPR/TPR tại ngưỡng cố định, theo từng điều kiện ánh sáng."""
    panels = [("i_auroc", "Image AUROC"), ("aupro", "AUPRO@0.3"),
              ("img_fpr_thr", "FPR @ ngưỡng val (ảnh good bị báo lỗi)"),
              ("img_tpr_thr", "TPR @ ngưỡng val (ảnh lỗi bắt được)")]
    d = a2[(a2.category == cat) & (a2.cond != "all")]
    conds = [c for c in COND_ORDER if c in set(d.cond)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6))
    w = 0.8 / max(1, d.method.nunique())
    for ax, (col, name) in zip(axes, panels):
        for j, (meth, g) in enumerate(sorted(d.groupby("method"))):
            v = g.set_index("cond").reindex(conds)[col]
            ax.bar(np.arange(len(conds)) + (j - (d.method.nunique() - 1) / 2) * w, v, w, label=meth, color=COLORS[meth])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"TN4a — {cat}: train ở ánh sáng thường, test ở nhiều điều kiện ánh sáng")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()


def plot_score_shift(sc, cat, path):
    """TN4a: điểm ảnh chia cho ngưỡng theo điều kiện (>1 = bị báo lỗi)."""
    d = sc[sc.category == cat]
    conds = [c for c in COND_ORDER if c in set(d.cond)]
    fig, axes = plt.subplots(1, d.method.nunique(), figsize=(5 * d.method.nunique(), 3.4), squeeze=False)
    for ax, (meth, g) in zip(axes[0], sorted(d.groupby("method"))):
        for lab, col, off in [(0, "tab:green", -0.15), (1, "tab:red", 0.15)]:
            for i, c in enumerate(conds):
                v = g[(g.cond == c) & (g.label == lab)].ratio
                ax.scatter(np.full(len(v), i + off), v, s=12, c=col, alpha=.7,
                           label=("good" if lab == 0 else "lỗi") if i == 0 else None)
        ax.axhline(1, color="k", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{meth}: điểm / ngưỡng", fontsize=10)
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"TN4a — {cat}: điểm ảnh so với ngưỡng (đường đứt = ngưỡng)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.show()


def plot_same_scene(ad2_root, cat, hw, maps_dir, thr, path):
    """TN4a: cùng một cảnh lỗi (vùng lỗi lớn nhất) dưới mọi điều kiện ánh sáng, heatmap của 3 phương pháp."""
    d = D.load_ad2(ad2_root, cat)
    M = {m: np.load(maps_dir / f"ad2_{cat}_{m}.npz") for m in METHODS if (maps_dir / f"ad2_{cat}_{m}.npz").exists()}
    if not M:
        return
    reg = [i for i, t in enumerate(d["test"]) if t["label"] == 1 and t["cond"] == "regular"]
    best = max(reg, key=lambda i: D.read_mask(d["test"][i]["mask"], hw).sum())
    scene = d["test"][best]["scene"]
    rows = sorted([i for i, t in enumerate(d["test"]) if t["scene"] == scene],
                  key=lambda i: COND_ORDER.index(d["test"][i]["cond"]) if d["test"][i]["cond"] in COND_ORDER else 99)
    rng = {m: np.percentile(z["maps"].astype(np.float32), [1, 99.5]) for m, z in M.items()}
    cols = ["Ảnh", "Ground truth"] + [m.upper() for m in METHODS]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.4 * len(cols) * hw[1] / hw[0], 2.5 * len(rows)),
                             squeeze=False)
    for r, i in enumerate(rows):
        t = d["test"][i]
        img, gt = D.read_image(t["path"], hw), D.read_mask(t["mask"], hw)
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(t["cond"], fontsize=9)
        axes[r, 1].imshow(mask_overlay(img, gt))
        for c, m in enumerate(METHODS, start=2):
            if m in M:
                s = float(M[m]["scores"][i])
                axes[r, c].imshow(overlay(img, M[m]["maps"][i].astype(np.float32), *rng[m]))
                axes[r, c].set_xlabel(f"score/ngưỡng = {s / thr.get(m, np.nan):.2f}", fontsize=8)
        for c in range(len(cols)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=10)
    fig.suptitle(f"TN4a — {cat}: cùng một cảnh lỗi ({scene}) dưới các điều kiện ánh sáng")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()


def merge_geo(sy, geo):
    """Dùng kết quả xoay/dịch làm lại đúng cách (tn4_syn_geo2) thay cho bản cũ."""
    if not len(geo):
        print("CHÚ Ý: chưa có kết quả geo2 -> xoay/dịch đang dùng bản cũ (lẫn hiệu ứng viền + nội suy).")
        return sy
    return pd.concat([sy[~sy.kind.isin(["rotate", "translate"])], geo], ignore_index=True)


def plot_shifts(sy, cat, kinds, path):
    """TN4b: I-AUROC, AUPRO, FPR theo mức shift tăng dần, mốc 'gốc' = không shift."""
    base = sy[sy.kind == "none"].set_index(["category", "method"])
    panels = [("i_auroc", "Image AUROC"), ("aupro", "AUPRO@0.3"), ("img_fpr_thr", "FPR @ ngưỡng val")]
    fig, axes = plt.subplots(len(kinds), len(panels), figsize=(4.3 * len(panels), 2.7 * len(kinds)), squeeze=False)
    for r, kind in enumerate(kinds):
        for c, (col, name) in enumerate(panels):
            ax = axes[r, c]
            for meth, g in sorted(sy[(sy.category == cat) & (sy.kind == kind)].groupby("method")):
                g = g.sort_values("level")
                b = base.loc[(cat, meth), col]
                ax.plot([0] + list(range(1, len(g) + 1)), [b] + list(g[col]), marker="o", color=COLORS[meth], label=meth)
                ax.set_xticks(range(len(g) + 1))
                ax.set_xticklabels(["gốc"] + [f"{l:g}" for l in g.level])
            ax.set_title(f"{kind} — {name}", fontsize=9)
            ax.grid(alpha=.3)
            if col != "img_fpr_thr":
                ax.set_ylim(None, 1.01)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"TN4b — {cat}: shift nhân tạo trên ảnh test (mức độ tăng dần sang phải)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.show()


def worst_case(sy):
    """Mức giảm I-AUROC / AUPRO và FPR ở mức shift mạnh nhất (độ sáng: xa 1 nhất) so với không shift."""
    base = sy[sy.kind == "none"].set_index(["category", "method"])
    rows = []
    for (cat, meth, kind), g in sy[sy.kind != "none"].groupby(["category", "method", "kind"]):
        worst = g.loc[g.level.abs().idxmax()] if kind != "brightness" else g.loc[(g.level - 1).abs().idxmax()]
        b = base.loc[(cat, meth)]
        rows.append(dict(category=cat, method=meth, kind=kind, level=worst.level,
                         d_i_auroc=round(worst.i_auroc - b.i_auroc, 3), d_aupro=round(worst.aupro - b.aupro, 3),
                         fpr_thr=round(worst.img_fpr_thr, 2), fpr_thr_goc=round(b.img_fpr_thr, 2)))
    return pd.DataFrame(rows)
