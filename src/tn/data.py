"""Đọc MVTec AD, MVTec AD 2, VisA; tải và giải nén dữ liệu trên Colab.

Mỗi bộ dữ liệu trả về dict(train=[đường dẫn], test=[dict(path, label, dtype, mask, ...)]), ảnh được đọc
thành uint8 (H, W, 3) ở kích thước `hw` = (cao, rộng).
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.tn.config import SQUARE

MVTEC_URL = ("https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"
             "420938113-1629952094/mvtec_anomaly_detection.tar.xz")
AD2_URLS = {
    "vial": "https://www.mydrive.ch/shares/163176/f0fb9a2d28793346815191991a807916/download/466654885-1743167660/vial.tar.gz",
    "sheet_metal": "https://www.mydrive.ch/shares/163660/945bbe557279ee9cdc6728374fac93ca/download/466654829-1743167110/sheet_metal.tar.gz",
}
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


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


# ----------------------------------------------------------------- tải dữ liệu (Colab)
def prepare_mvtec(data_root, cats, zip_on_drive):
    """Giải nén zip các category từ Drive; không có zip thì tải bản đầy đủ rồi lưu zip lên Drive."""
    data_root, zip_on_drive = Path(data_root), Path(zip_on_drive)
    ok = lambda: all((data_root / c / "train" / "good").is_dir() for c in cats)
    if not ok():
        data_root.mkdir(parents=True, exist_ok=True)
        if zip_on_drive.exists():
            print("Giải nén từ Drive ...")
            with zipfile.ZipFile(zip_on_drive) as z:
                z.extractall(data_root)
        else:
            print("Không thấy zip trên Drive -> tải bản đầy đủ từ MVTec (vài phút) ...")
            subprocess.run(["wget", "-q", "-O", "/content/mvtec.tar.xz", MVTEC_URL], check=True)
            pats = [f"*{c}/*" for c in cats]
            subprocess.run(["tar", "-xJf", "/content/mvtec.tar.xz", "-C", str(data_root), "--wildcards", *pats],
                           check=True)
            Path("/content/mvtec.tar.xz").unlink()
            print("Lưu zip các category lên Drive cho lần sau ...")
            shutil.make_archive(str(zip_on_drive.with_suffix("")), "zip", data_root, ".")
    for c in cats:                                   # zip có thêm 1 cấp thư mục (vd. MVTec_AD/screw) -> kéo lên
        if not (data_root / c).is_dir():
            hits = list(data_root.glob(f"*/{c}"))
            if hits:
                shutil.move(str(hits[0]), str(data_root / c))
    assert ok(), f"Thiếu dữ liệu trong {data_root}: cần {cats}/train/good"
    for c in cats:
        n_tr = len(list((data_root / c / "train" / "good").glob("*.png")))
        n_te = len(list((data_root / c / "test").glob("*/*.png")))
        print(f"{c:12s} train={n_tr:4d}  test={n_te:4d}")


def prepare_ad2(ad2_root, cats, zip_on_drive=None):
    """Tải MVTec AD 2 từ trang MVTec (chỉ giải nén train / validation / test_public)."""
    ad2_root = Path(ad2_root)
    ok = lambda c: all((ad2_root / c / s).is_dir() for s in ("train", "validation", "test_public"))
    ad2_root.mkdir(parents=True, exist_ok=True)
    if zip_on_drive is not None and Path(zip_on_drive).exists() and not all(ok(c) for c in cats):
        print("Giải nén", Path(zip_on_drive).name, "...")
        with zipfile.ZipFile(zip_on_drive) as z:
            z.extractall(ad2_root)
    for c in cats:
        if not ok(c):
            tar = Path(f"/content/{c}.tar.gz")
            print(f"Tải {c} từ MVTec ...")
            subprocess.run(["wget", "-q", "-O", str(tar), AD2_URLS[c]], check=True)
            subprocess.run(["tar", "-xzf", str(tar), "-C", str(ad2_root), "--wildcards",
                            "*/train/*", "*/validation/*", "*/test_public/*"], check=True)
            tar.unlink()
        if not (ad2_root / c).is_dir():              # bị bọc thêm 1 cấp thư mục
            hits = [p for p in ad2_root.glob(f"**/{c}") if p.is_dir()]
            if hits:
                shutil.move(str(hits[0]), str(ad2_root / c))
        assert ok(c), f"Thiếu dữ liệu {c} trong {ad2_root}"
        n = {s: len(list((ad2_root / c / s).rglob("*.png"))) for s in ("train", "validation", "test_public")}
        print(f"{c:12s}", n)


def prepare_visa(visa_dir, cats):
    """Tải VisA từ AWS. Trả về (thư mục gốc chứa split_csv/, bảng split 1cls)."""
    visa_dir = Path(visa_dir)

    def find_root():
        hits = [p.parent.parent for p in visa_dir.glob("**/split_csv/1cls.csv")]
        return hits[0] if hits else None

    root = find_root()
    if root is None or not all((root / c).is_dir() for c in cats):
        visa_dir.mkdir(parents=True, exist_ok=True)
        tar = Path("/content/VisA.tar")
        print("Tải VisA (~1,8 GB) ...")
        subprocess.run(["wget", "-q", "-O", str(tar), VISA_URL], check=True)
        pats = [f"*{c}/*" for c in cats] + ["*split_csv/*"]
        subprocess.run(["tar", "-xf", str(tar), "-C", str(visa_dir), "--wildcards", *pats], check=True)
        tar.unlink()
        root = find_root()
    assert root is not None, f"Không tìm thấy split_csv/1cls.csv trong {visa_dir}"
    split = pd.read_csv(root / "split_csv" / "1cls.csv")
    print("VisA ở", root)
    print(split[split.object.isin(cats)].groupby(["object", "split", "label"]).size().unstack(["split", "label"]))
    return root, split
