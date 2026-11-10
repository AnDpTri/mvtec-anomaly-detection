"""Metric dùng chung cho cả 3 phương pháp: không cần ngưỡng (AUROC, AP, F1-max, AUPRO) và tại ngưỡng cố định."""
import numpy as np
from scipy.ndimage import label as cc_label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3, n_steps: int = 1000) -> float:
    """AUPRO tới FPR=fpr_limit, chuẩn hóa về [0,1]. Mỗi vùng lỗi liên thông (8-connectivity) có trọng số như nhau."""
    normal = np.sort(maps[~masks].ravel())
    regions = []
    for m, g in zip(maps, masks):
        if g.any():
            lab, n = cc_label(g, structure=np.ones((3, 3)))
            regions += [np.sort(m[lab == r]) for r in range(1, n + 1)]
    if not regions or normal.size == 0:
        return float("nan")
    fpr_grid = np.linspace(0, fpr_limit, n_steps)
    thr = np.quantile(normal, 1 - fpr_grid)                       # ngưỡng giảm dần
    fpr = 1 - np.searchsorted(normal, thr, side="right") / normal.size
    pro = np.mean([1 - np.searchsorted(r, thr, side="right") / r.size for r in regions], axis=0)
    o = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[o], pro[o]
    pro_lim = np.interp(fpr_limit, fpr, pro)
    keep = fpr <= fpr_limit
    x, y = np.r_[fpr[keep], fpr_limit], np.r_[pro[keep], pro_lim]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, x) / fpr_limit)


def image_scores(maps: np.ndarray) -> np.ndarray:
    """Điểm ảnh = giá trị lớn nhất của anomaly map. maps (N,H,W) -> (N,)."""
    return maps.reshape(len(maps), -1).max(1)


def f1_max(y, s) -> tuple[float, float]:
    """-> (F1 lớn nhất trên mọi ngưỡng, ngưỡng đạt F1 đó)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(f1[i]), float(t[i]) if len(t) else float("nan")


def compute_metrics(scores, maps, labels, masks) -> dict:
    """Chỉ số không cần ngưỡng: I-AUROC, I-AP, I-F1max, P-AUROC, P-AP, AUPRO@0.3."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    out = dict(i_auroc=roc_auc_score(labels, scores), i_ap=average_precision_score(labels, scores))
    out["i_f1max"], out["i_f1_thr"] = f1_max(labels, scores)
    flat_m, flat_s = masks.ravel(), maps.ravel()
    out["p_auroc"] = roc_auc_score(flat_m, flat_s)
    out["p_ap"] = average_precision_score(flat_m, flat_s)
    out["aupro"] = aupro(maps, masks)
    return out


def per_defect_auroc(scores, labels, dtypes) -> dict:
    """AUROC của từng loại lỗi: ảnh lỗi loại đó so với toàn bộ ảnh good."""
    scores, labels, dtypes = map(np.asarray, (scores, labels, dtypes))
    good = scores[labels == 0]
    return {d: roc_auc_score(np.r_[np.zeros(len(good)), np.ones((dtypes == d).sum())],
                             np.r_[good, scores[dtypes == d]])
            for d in sorted(set(dtypes[labels == 1]))}


def thresholds_from_val(val_maps: np.ndarray) -> dict:
    """Ngưỡng = trung bình + 3 độ lệch chuẩn trên ảnh good validation (cách của paper MVTec AD 2)."""
    s = image_scores(val_maps)
    return dict(thr_img=float(s.mean() + 3 * s.std()), thr_pix=float(val_maps.mean() + 3 * val_maps.std()))


def metrics_full(scores, maps, labels, masks, thr: dict) -> dict:
    """compute_metrics + AUPRO@0.05 + FPR/TPR/F1 tại ngưỡng cố định `thr` (mức ảnh và mức pixel)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    out = compute_metrics(scores, maps, labels, masks)
    out["aupro_005"] = aupro(maps, masks, fpr_limit=0.05)
    pred = scores >= thr["thr_img"]
    tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
    out["img_fpr_thr"] = fp / max(1, (labels == 0).sum())       # tỷ lệ ảnh good bị báo lỗi
    out["img_tpr_thr"] = tp / max(1, (labels == 1).sum())       # tỷ lệ ảnh lỗi bắt được
    out["img_f1_thr"] = 2 * tp / max(1, 2 * tp + fp + int(((~pred) & (labels == 1)).sum()))
    pp = maps >= thr["thr_pix"]
    ptp, pfp, pfn = int((pp & masks).sum()), int((pp & ~masks).sum()), int((~pp & masks).sum())
    out["pix_f1_thr"] = 2 * ptp / max(1, 2 * ptp + pfp + pfn)   # SegF1 tại ngưỡng validation
    out["pix_fpr_thr"] = pfp / max(1, (~masks).sum())
    return out
