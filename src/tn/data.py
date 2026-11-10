"""Đọc MVTec AD, MVTec AD 2, VisA (tải dữ liệu: xem download.py).

Mỗi bộ dữ liệu trả về dict(train=[đường dẫn], test=[dict(path, label, dtype, mask, ...)]), ảnh được đọc
thành uint8 (H, W, 3) ở kích thước `hw` = (cao, rộng).
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.tn.config import SQUARE


# ----------------------------------------------------------------- đọc ảnh
def read_image(path, hw=SQUARE):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)   # ảnh xám tự thành 3 kênh
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def read_mask(path, hw=SQUARE, thr=127):
    """Mask nhị phân. MVTec lưu 0/255 (thr=127); VisA lưu 0/1 nên dùng thr=0."""
    if path is None:
        return np.zeros(hw, bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > thr


def load_imgs(paths, hw=SQUARE):
    return np.stack([read_image(p, hw) for p in paths])


def load_masks(items, hw=SQUARE, thr=127):
    return np.stack([read_mask(t["mask"], hw, thr) for t in items])


# ----------------------------------------------------------------- danh sách ảnh
def load_mvtec(root, cat):
    root = Path(root) / cat
    train = sorted(str(p) for p in (root / "train" / "good").glob("*.png"))
    test = []
    for d in sorted((root / "test").iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            if d.name == "good":
                test.append(dict(path=str(p), label=0, dtype="good", mask=None))
            else:
                m = root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                test.append(dict(path=str(p), label=1, dtype=d.name, mask=str(m) if m.exists() else None))
    return dict(train=train, test=test)


def load_ad2(root, cat):
    """MVTec AD 2: train/good, validation/good, test_public (tên file '<cảnh>_<điều kiện>.png')."""
    root = Path(root) / cat
    train = sorted(str(p) for p in (root / "train" / "good").glob("*.png"))
    val = sorted(str(p) for p in (root / "validation" / "good").glob("*.png"))
    test = []
    for lab, sub in [(0, "good"), (1, "bad")]:
        for p in sorted((root / "test_public" / sub).glob("*.png")):
            scene, cond = re.match(r"(\d+)_(.+)\.png$", p.name).groups()
            m = root / "test_public" / "ground_truth" / "bad" / f"{p.stem}_mask.png"
            test.append(dict(path=str(p), label=lab, dtype=sub, cond=cond, scene=f"{sub}_{scene}",
                             mask=str(m) if lab and m.exists() else None))
    return dict(train=train, val=val, test=test)


def load_visa(root, split, cat):
    """VisA theo split 1cls chính thức (`split` = bảng split_csv/1cls.csv)."""
    root = Path(root)
    d = split[split.object == cat]
    anno = pd.read_csv(root / cat / "image_anno.csv").set_index("image")["label"].to_dict()
    train = [str(root / p) for p in d[d.split == "train"].image]
    test = []
    for r in d[d.split == "test"].itertuples():
        lab = int(r.label == "anomaly")
        test.append(dict(path=str(root / r.image), label=lab,
                         dtype=str(anno.get(r.image, "anomaly")) if lab else "good",
                         mask=str(root / r.mask) if lab and isinstance(r.mask, str) else None))
    return dict(train=train, test=test)
