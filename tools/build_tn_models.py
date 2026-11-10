"""Fit 3 phương pháp cho 9 loại của thí nghiệm TN1–TN5 và lưu mô hình cho demo.

Cấu hình giống hệt src/tn (cùng cách chia fit/val, kích thước ảnh, coreset như TN3/TN4):
- MVTec AD  grid, screw, transistor  256×256, coreset 10%, fit 85% train/good, ngưỡng trên 15% còn lại
- MVTec AD 2 vial (256×192), sheet_metal (256×1024), coreset 1%, fit train/good, ngưỡng trên validation/good
- VisA      pcb1–pcb4                256×320, coreset 1%, fit 85% train (split 1cls), ngưỡng trên 15% còn lại

Mỗi loại ghi vào results/checkpoints_tn/<loại>/:
  ae.pt (state_dict ConvAE), padim.pt (mu, inv, H, W), patchcore_bank.pt (fp16), meta.json, test_scores.csv

Chạy:  python tools/build_tn_models.py [loại ...] [--data DIR] [--out DIR]
       (bỏ trống loại = cả 9; loại đã có meta.json thì bỏ qua, nên chạy lại sau khi Colab ngắt vẫn tiếp được)
       DIR dữ liệu chứa MVTecAD/, MVTecAD2/, VisA/ (VisA/split_csv/1cls.csv).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.tn import data as D                                         # noqa: E402
from src.tn.config import DEVICE, METHODS, free_gpu                  # noqa: E402
from src.tn.methods import fit_detector                              # noqa: E402
from src.tn.metrics import image_scores, metrics_full, thresholds_from_val   # noqa: E402
from src.tn.splits import split_fit_val                              # noqa: E402

DATA = ROOT / "datasets"
OUT = ROOT / "results" / "checkpoints_tn"

CATS = {
    "grid": ("mvtec", (256, 256), 0.10),
    "screw": ("mvtec", (256, 256), 0.10),
    "transistor": ("mvtec", (256, 256), 0.10),
    "vial": ("ad2", (256, 192), 0.01),
    "sheet_metal": ("ad2", (256, 1024), 0.01),
    "pcb1": ("visa", (256, 320), 0.01),
    "pcb2": ("visa", (256, 320), 0.01),
    "pcb3": ("visa", (256, 320), 0.01),
    "pcb4": ("visa", (256, 320), 0.01),
}


def load_split(cat, source, data_dir):
    """-> (ảnh fit, ảnh val, danh sách test, mask thr). Đường dẫn giữ dạng tương đối với datasets/."""
    if source == "mvtec":
        d = D.load_mvtec(data_dir / "MVTecAD", cat)
        fit_p, val_p = split_fit_val(d["train"])
        return fit_p, val_p, d["test"], 127
    if source == "ad2":
        d = D.load_ad2(data_dir / "MVTecAD2", cat)
        return d["train"], d["val"], d["test"], 127
    root = data_dir / "VisA"
    d = D.load_visa(root, pd.read_csv(root / "split_csv" / "1cls.csv"), cat)
    fit_p, val_p = split_fit_val(d["train"])
    return fit_p, val_p, d["test"], 0


def build(cat, data_dir=DATA, out_dir=OUT):
    source, hw, coreset = CATS[cat]
    out = out_dir / cat
    if (out / "meta.json").exists():
        print(f"[{cat}] đã có, bỏ qua")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fit_p, val_p, test, mthr = load_split(cat, source, data_dir)
    imgs_fit, imgs_val = D.load_imgs(fit_p, hw), D.load_imgs(val_p, hw)
    imgs_te = D.load_imgs([t["path"] for t in test], hw)
    masks = D.load_masks(test, hw, thr=mthr)
    labels = np.array([t["label"] for t in test])
    print(f"[{cat}] {hw[0]}x{hw[1]} fit={len(fit_p)} val={len(val_p)} test={len(test)} "
          f"({int(labels.sum())} lỗi) nạp trong {time.time() - t0:.0f}s")

    rel = lambda p: Path(p).relative_to(data_dir).as_posix() if p else ""
    scores_df = pd.DataFrame(dict(path=[rel(t["path"]) for t in test], dtype=[t["dtype"] for t in test],
                                  label=labels, mask=[rel(t["mask"]) for t in test]))
    if source == "ad2":
        scores_df["cond"] = [t["cond"] for t in test]
    meta = dict(category=cat, dataset=source, hw=list(hw), coreset=coreset, n_fit=len(fit_p),
                val_paths=[rel(p) for p in val_p], thr={}, metrics={})

    for method in METHODS:
        t1 = time.time()
        model = fit_detector(method, imgs_fit, coreset=coreset)
        val_maps = model.predict(imgs_val)
        thr = thresholds_from_val(val_maps)
        maps = model.predict(imgs_te)
        scores = image_scores(maps)
        met = metrics_full(scores, maps, labels, masks, thr)
        model.save(out)
        meta["thr"][method] = dict(**thr, lo=float(val_maps.mean()))
        meta["metrics"][method] = {k: round(float(met[k]), 4) for k in
                                   ("i_auroc", "p_auroc", "aupro", "img_fpr_thr", "img_tpr_thr")}
        scores_df[f"score_{method}"] = scores
        print(f"  {method:9s} I-AUROC={met['i_auroc']:.3f} P-AUROC={met['p_auroc']:.3f} AUPRO={met['aupro']:.3f} "
              f"FPR@ngưỡng={met['img_fpr_thr']:.2f} TPR@ngưỡng={met['img_tpr_thr']:.2f} ({time.time() - t1:.0f}s)")
        del model, val_maps, maps
        free_gpu()

    meta["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    scores_df.to_csv(out / "test_scores.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{cat}] xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cats", nargs="*", help=", ".join(CATS))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if set(a.cats) - set(CATS):
        ap.error(f"loại không hợp lệ: {sorted(set(a.cats) - set(CATS))}")
    print("thiết bị:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
    for c in a.cats or list(CATS):
        build(c, a.data, a.out)
