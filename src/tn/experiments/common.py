"""Phần dùng chung của các vòng chạy: cache ảnh/đặc trưng của một category và fit + đặt ngưỡng."""
import time

import numpy as np
import torch

from src.tn import data as D
from src.tn.methods import Detector, extract_features, fit_detector
from src.tn.metrics import thresholds_from_val


class CategoryCache:
    """Ảnh, mask và đặc trưng của MỘT category MVTec AD trong RAM (đổi category thì xóa cái cũ)."""

    def __init__(self, data_root, methods):
        self.data_root, self.methods, self.cur = data_root, methods, {}

    def get(self, cat: str, feats: bool = True) -> dict:
        need = feats and any(m in self.methods for m in ("padim", "patchcore"))
        C = self.cur.get(cat)
        if C is not None and (C["feats"] is not None or not need):
            return C
        t0 = time.time()
        data = D.load_mvtec(self.data_root, cat)
        imgs_tr = D.load_imgs(data["train"])
        imgs_te = D.load_imgs([t["path"] for t in data["test"]])
        C = dict(data=data, imgs_tr=imgs_tr, imgs_te=imgs_te, masks=D.load_masks(data["test"]), feats=None)
        if need:
            pad_tr, pc_tr = extract_features(imgs_tr)
            pad_te, pc_te = extract_features(imgs_te)
            C["feats"] = dict(padim=(pad_tr, pad_te), patchcore=(pc_tr, pc_te))
        self.cur = {cat: C}
        print(f"[{cat}] đã nạp {len(imgs_tr)} train / {len(imgs_te)} test trong {time.time() - t0:.0f}s")
        return C


def gather(C: dict, items, kind: str):
    """items: list ('train', i) | ('test', i) -> mảng ảnh (kind='img') hoặc đặc trưng của phương pháp `kind`."""
    if kind == "img":
        return np.stack([C["imgs_tr"][i] if s == "train" else C["imgs_te"][i] for s, i in items])
    tr, te = C["feats"][kind]
    idx_tr = [i for s, i in items if s == "train"]
    idx_te = [i for s, i in items if s == "test"]
    return torch.cat([tr[idx_tr]] + ([te[idx_te]] if idx_te else []))


def fit_with_threshold(method: str, imgs_fit, imgs_val, seed: int = 0,
                       coreset: float | None = None) -> tuple[Detector, dict]:
    """Fit trên ảnh good rồi đặt ngưỡng (trung bình + 3σ) từ ảnh good validation."""
    model = fit_detector(method, imgs_fit, seed=seed, coreset=coreset)
    return model, thresholds_from_val(model.predict(imgs_val))
