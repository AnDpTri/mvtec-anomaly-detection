"""TN5: phân tích ca thất bại từ anomaly map đã lưu ở TN2 (mức 'all')."""
import matplotlib.pyplot as plt
import numpy as np

from src.tn.analysis.common import mask_overlay, overlay
from src.tn.config import METHODS
from src.tn.metrics import f1_max


def load_maps(maps_dir, cat):
    out = {}
    for m in METHODS:
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
    cols = ["Ảnh", "Ground truth"] + [m.upper() for m in METHODS]
    rng = {m: np.percentile(d["maps"], [1, 99.5]) for m, d in M.items()}
    fig, axes = plt.subplots(len(rows_idx), len(cols), figsize=(2.6 * len(cols), 2.7 * len(rows_idx)), squeeze=False)
    for r, (i, note) in enumerate(rows_idx):
        t, img, gt = C["data"]["test"][i], C["imgs_te"][i], C["masks"][i]
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{t['dtype']}\n{note}", fontsize=8)
        axes[r, 1].imshow(mask_overlay(img, gt))
        for c, m in enumerate(METHODS, start=2):
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
