"""Nạp mô hình của tools/build_tn_models.py (results/checkpoints_tn/<loại>/) và chấm một ảnh.

Mỗi loại có ae.pt, padim.pt, patchcore_bank.pt và meta.json (kích thước ảnh, ngưỡng mean + 3σ trên ảnh good
validation, chỉ số trên tập test). Suy luận dùng lại đúng hàm của src/tn nên kết quả trùng với lúc fit.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.common.config import ROOT
from src.tn.config import DEVICE
from src.tn.methods import ConvAE, predict_method

MODEL_ROOT = Path(os.environ.get("TN_CHECKPOINT_ROOT", ROOT / "results" / "checkpoints_tn"))
DATA_ROOT = ROOT / "datasets"
DATASET_NAME = {"mvtec": "MVTec AD", "ad2": "MVTec AD 2", "visa": "VisA"}
FILES = {"ae": "ae.pt", "padim": "padim.pt", "patchcore": "patchcore_bank.pt"}


def categories():
    """-> [(nhãn hiển thị, tên loại)] cho các loại đã có meta.json."""
    out = []
    for d in sorted(MODEL_ROOT.glob("*/meta.json")):
        m = load_meta(d.parent.name)
        out.append((f"{m['category']} ({DATASET_NAME[m['dataset']]})", m["category"]))
    return out


@lru_cache(maxsize=None)
def load_meta(cat):
    p = MODEL_ROOT / cat / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có mô hình cho {cat}: {p}. Xem tools/build_tn_models.py.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)                       # PaDiM sheet_metal chiếm ~650 MB, chỉ giữ một mô hình
def load_model(cat, method):
    f = MODEL_ROOT / cat / FILES[method]
    st = torch.load(f, map_location="cpu", weights_only=True)
    if method == "ae":
        net = ConvAE()
        net.load_state_dict(st)
        return "ae", net.to(DEVICE).eval()
    if method == "padim":
        return "padim", dict(mu=st["mu"].to(DEVICE), inv=st["inv"].to(DEVICE), H=st["H"], W=st["W"])
    return "patchcore", st.to(DEVICE).float()


def to_input(image, hw):
    """Ảnh PIL/numpy -> uint8 (1, H, W, 3), resize như src.tn.data.read_image."""
    img = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return cv2.resize(img[..., :3], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)[None]


def predict(image, cat, method):
    """-> (ảnh đã resize, anomaly map, điểm ảnh, meta)."""
    meta = load_meta(cat)
    x = to_input(image, tuple(meta["hw"]))
    m = predict_method(load_model(cat, method), x)[0]
    return x[0], m, float(m.max()), meta


def examples(per_cat=2):
    """Ảnh test có sẵn trên máy: mỗi loại 1 ảnh tốt + (per_cat - 1) ảnh lỗi, lấy từ test_scores.csv."""
    out = []
    for _, cat in categories():
        df = pd.read_csv(MODEL_ROOT / cat / "test_scores.csv")
        pick = pd.concat([df[df.label == 0].head(1), df[df.label == 1].head(per_cat - 1)])
        out += [[str(DATA_ROOT / p), cat] for p in pick.path if (DATA_ROOT / p).exists()]
    return out
