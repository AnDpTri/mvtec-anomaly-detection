"""Số liệu và hình cho Chương 3 từ mô hình trong results/checkpoints_tn (không fit lại).

- results/tn_cost.csv: thời gian suy luận (ms/ảnh, trên GPU đang có) và dung lượng mô hình mỗi loại × phương pháp
- bao_cao/images/ket_qua/tn_<loại>_<kiểu lỗi>.png: ảnh, mặt nạ, bản đồ của 3 phương pháp trên cùng một ảnh lỗi;
  thang màu mỗi phương pháp đi từ trung bình pixel (0) tới ngưỡng pixel (1) của ảnh good validation

Chạy:  python tools/tn_report_figures.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.demo import tn_models  # noqa: E402
from src.tn import data as D  # noqa: E402
from src.tn.config import DEVICE  # noqa: E402
from src.tn.methods import free_gpu, predict_method  # noqa: E402

METHODS = ["ae", "padim", "patchcore"]
N_TIME = 32                                      # số ảnh test dùng để đo thời gian suy luận
NAME = {"ae": "Autoencoder", "padim": "PaDiM", "patchcore": "PatchCore"}
FIG_DIR = ROOT / "bao_cao" / "images" / "ket_qua"
# (loại, kiểu lỗi, chỉ số ảnh trong test_scores.csv của kiểu lỗi đó)
CASES = [("grid", "bent", 0), ("screw", "thread_side", 0), ("transistor", "misplaced", 0),
         ("pcb3", "bent", 0), ("vial", "bad", 0), ("sheet_metal", "bad", 0)]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def cost():
    rows = []
    for _, cat in tn_models.categories():
        meta = tn_models.load_meta(cat)
        df = pd.read_csv(tn_models.MODEL_ROOT / cat / "test_scores.csv")
        imgs = D.load_imgs([str(tn_models.DATA_ROOT / p) for p in df.path[:N_TIME]], tuple(meta["hw"]))
        for m in METHODS:
            model = tn_models.load_model(cat, m)
            predict_method(model, imgs[:1])                           # khởi động GPU
            sync()
            t0 = time.perf_counter()
            for i in range(N_TIME):                                   # từng ảnh một, như khi dùng demo
                predict_method(model, imgs[i:i + 1])
            sync()
            ms = (time.perf_counter() - t0) / N_TIME * 1000
            mb = (tn_models.MODEL_ROOT / cat / tn_models.FILES[m]).stat().st_size / 2**20
            rows.append(dict(loai=cat, pp=m, hw=f"{meta['hw'][0]}x{meta['hw'][1]}", ms_per_img=round(ms, 1),
                             mb=round(mb, 1)))
            print(f"{cat:11s} {m:9s} {ms:6.1f} ms/ảnh  {mb:6.1f} MB")
            free_gpu()
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tn_cost.csv", index=False)
    print(out.pivot_table(index="pp", values=["ms_per_img", "mb"], aggfunc=["mean", "min", "max"]).round(1))


def figure(cat, dtype, k):
    meta = tn_models.load_meta(cat)
    hw = tuple(meta["hw"])
    df = pd.read_csv(tn_models.MODEL_ROOT / cat / "test_scores.csv")
    r = df[df.dtype == dtype].iloc[k]
    img = D.load_imgs([str(tn_models.DATA_ROOT / r.path)], hw)
    mask = D.load_masks([dict(mask=str(tn_models.DATA_ROOT / r["mask"]))], hw, thr=0 if meta["dataset"] == "visa" else 127)[0]
    wide = hw[1] / hw[0] > 2
    fig, ax = plt.subplots(5 if wide else 1, 1 if wide else 5, figsize=(9, 6) if wide else (11, 2.6))
    ax = np.ravel(ax)
    sep = ": " if wide else "\n"                     # ảnh dài xếp dọc: tiêu đề một dòng cho gọn
    ax[0].imshow(img[0]); ax[0].set_title("Ảnh" + ("" if wide else "\n"), fontsize=10)
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Mặt nạ lỗi" + ("" if wide else "\n"), fontsize=10)
    for a, m in zip(ax[2:], METHODS):
        amap = predict_method(tn_models.load_model(cat, m), img)[0]
        t = meta["thr"][m]
        norm = np.clip((amap - t["lo"]) / max(t["thr_pix"] - t["lo"], 1e-8), 0, 1)
        a.imshow(img[0]); a.imshow(norm, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        verdict = "lỗi" if amap.max() >= t["thr_img"] else "tốt"
        a.set_title(f"{NAME[m]}{sep}{amap.max() / t['thr_img']:.2f} × ngưỡng, kết luận {verdict}", fontsize=10)
        free_gpu()
    for a in ax:
        a.set_axis_off()
    fig.tight_layout(h_pad=0.3 if wide else None)
    out = FIG_DIR / f"tn_{cat}_{dtype}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("đã vẽ", out.relative_to(ROOT), "|", r.path)


if __name__ == "__main__":
    print("thiết bị:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if "--chi-do" not in sys.argv:
        for c in CASES:
            figure(*c)
    if "--chi-hinh" not in sys.argv:
        cost()
