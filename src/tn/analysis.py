"""Bảng tổng hợp và biểu đồ cho phần phân tích của TN1–TN5. Mọi hình lưu vào `fig_dir`."""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.tn import data as D
from src.tn.metrics import f1_max

COLORS = dict(ae="tab:green", padim="tab:blue", patchcore="tab:red")
ALL_METHODS = ["ae", "padim", "patchcore"]
SHOT_ORDER = ["1", "2", "4", "8", "16", "all"]
COND_ORDER = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4"]


def overlay(img, heat, vmin, vmax):
    h = np.clip((heat - vmin) / (vmax - vmin + 1e-8), 0, 1)
    h = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * h).astype(np.uint8)


# ----------------------------------------------------------------- TN1, TN2
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


# ----------------------------------------------------------------- TN2b
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


# ----------------------------------------------------------------- TN5
def load_maps(maps_dir, cat):
    out = {}
    for m in ALL_METHODS:
        p = maps_dir / f"{cat}_{m}.npz"
        if p.exists():
            z = np.load(p)
            out[m] = dict(maps=z["maps"].astype(np.float32), scores=z["scores"])
    return out


def failure_cases(scores, labels, n=3):
    """n ảnh lỗi điểm thấp nhất và n ảnh good điểm cao nhất, gắn nhãn theo ngưỡng F1-max."""
    labels = np.asarray(labels)
    _, thr = f1_max(labels, scores)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    fn = pos[np.argsort(scores[pos])][:n]
    fp = neg[np.argsort(-scores[neg])][:n]
    tag = lambda i: ("FN" if scores[i] < thr else "TP(thấp)") if labels[i] else ("FP" if scores[i] >= thr else "TN(cao)")
    return [(i, tag(i)) for i in fn], [(i, tag(i)) for i in fp], thr


def draw_grid(C, M, rows_idx, title, path):
    """Ảnh | ground truth | AE | PaDiM | PatchCore; thang màu mỗi phương pháp chung cho cả tập test."""
    cols = ["Ảnh", "Ground truth"] + [m.upper() for m in ALL_METHODS]
    rng = {m: np.percentile(d["maps"], [1, 99.5]) for m, d in M.items()}
    fig, axes = plt.subplots(len(rows_idx), len(cols), figsize=(2.6 * len(cols), 2.7 * len(rows_idx)), squeeze=False)
    for r, (i, note) in enumerate(rows_idx):
        t, img, gt = C["data"]["test"][i], C["imgs_te"][i], C["masks"][i]
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{t['dtype']}\n{note}", fontsize=8)
        g = img.copy()
        g[gt] = (0.5 * g[gt] + [127, 0, 0]).astype(np.uint8)
        axes[r, 1].imshow(g)
        for c, m in enumerate(ALL_METHODS, start=2):
            if m in M:
                axes[r, c].imshow(overlay(img, M[m]["maps"][i], *rng[m]))
                axes[r, c].set_xlabel(f"score={M[m]['scores'][i]:.2f}", fontsize=8)
        for c in range(len(cols)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.show()


# ----------------------------------------------------------------- TN3
def plot_matrices(df, cats, title, path):
    """Hàng = nguồn (train), cột = đích (test). Đường chéo = train và test cùng category."""
    meths = [m for m in ALL_METHODS if m in set(df.method)]
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
            for m in ALL_METHODS:
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


# ----------------------------------------------------------------- TN4
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
    M = {m: np.load(maps_dir / f"ad2_{cat}_{m}.npz") for m in ALL_METHODS if (maps_dir / f"ad2_{cat}_{m}.npz").exists()}
    if not M:
        return
    reg = [i for i, t in enumerate(d["test"]) if t["label"] == 1 and t["cond"] == "regular"]
    best = max(reg, key=lambda i: D.read_mask(d["test"][i]["mask"], hw).sum())
    scene = d["test"][best]["scene"]
    rows = sorted([i for i, t in enumerate(d["test"]) if t["scene"] == scene],
                  key=lambda i: COND_ORDER.index(d["test"][i]["cond"]) if d["test"][i]["cond"] in COND_ORDER else 99)
    rng = {m: np.percentile(z["maps"].astype(np.float32), [1, 99.5]) for m, z in M.items()}
    cols = ["Ảnh", "Ground truth"] + [m.upper() for m in ALL_METHODS]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.4 * len(cols) * hw[1] / hw[0], 2.5 * len(rows)),
                             squeeze=False)
    for r, i in enumerate(rows):
        t = d["test"][i]
        img, gt = D.read_image(t["path"], hw), D.read_mask(t["mask"], hw)
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(t["cond"], fontsize=9)
        g = img.copy()
        g[gt] = (0.5 * g[gt] + [127, 0, 0]).astype(np.uint8)
        axes[r, 1].imshow(g)
        for c, m in enumerate(ALL_METHODS, start=2):
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
