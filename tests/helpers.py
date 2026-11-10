"""Đồ dùng chung cho các kiểm thử: backbone giả, dữ liệu giả, dựng thư mục MVTec AD giả, ép chạy CPU.

Backbone WideResNet được thay bằng mạng nhỏ (cùng số kênh, cùng độ phân giải đầu ra) để không phải tải
trọng số. Mọi kiểm thử chạy trên CPU với dữ liệu nhỏ, không cần datasets/ hay mô hình thật.
"""
from pathlib import Path
from unittest.mock import patch

import warnings

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")   # plt.show() trong hàm vẽ
torch.set_num_threads(2)


class FakeBackbone(nn.Module):
    """Thay cho WideResNet-50-2: trả về đặc trưng 256/512/1024 kênh ở lưới H/4, H/8, H/16."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c1 = nn.Conv2d(3, 256, 4, 4)
        self.c2 = nn.Conv2d(256, 512, 2, 2)
        self.c3 = nn.Conv2d(512, 1024, 2, 2)

    @torch.no_grad()
    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        return f1, f2, self.c3(f2)


def use_fake_backbone_on_cpu(case, ae_steps=3, ae_batch=4):
    """Trong một TestCase: backbone giả, AE train vài bước, ép CPU (máy có GPU vẫn chạy kiểm thử trên CPU)."""
    for p in (patch("src.tn.methods.backbone.get_backbone", lambda: FakeBackbone()),
              patch("src.tn.config.DEVICE", "cpu"),
              patch("src.tn.config.AE_STEPS", ae_steps),
              patch("src.tn.config.AE_BATCH", ae_batch)):
        p.start()
        case.addCleanup(p.stop)


def fake_data(n_test=6, n_good=8, size=64, seed=0):
    """Ảnh nền nhiễu nhẹ; ảnh lỗi có một ô sáng, mask đúng ô đó."""
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 110, (n_good, size, size, 3), dtype=np.uint8)
    test = rng.integers(90, 110, (n_test, size, size, 3), dtype=np.uint8)
    masks = np.zeros((n_test, size, size), bool)
    labels = np.array([0, 1] * (n_test // 2))
    for i in np.where(labels == 1)[0]:
        test[i, 20:36, 20:36] = 250
        masks[i, 20:36, 20:36] = True
    return base, test, labels, masks


def write_fake_mvtec(root, cat, n_train=8, n_good=4, n_bad=4, size=256, seed=0):
    """Ghi thư mục theo bố cục MVTec AD: <root>/<cat>/{train/good, test/good, test/bad, ground_truth/bad}."""
    base, good, _, _ = fake_data(n_test=2 * n_good, n_good=n_train, size=size, seed=seed)
    _, bad, _, masks = fake_data(n_test=2 * n_bad, n_good=1, size=size, seed=seed + 1)
    d = Path(root) / cat
    for sub in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    save = lambda p, a: cv2.imwrite(str(p), np.ascontiguousarray(a))
    for i, im in enumerate(base):
        save(d / "train/good" / f"{i:03d}.png", im)
    for i in range(n_good):
        save(d / "test/good" / f"{i:03d}.png", good[i])
    for j, i in enumerate(np.where(masks.any((1, 2)))[0][:n_bad]):
        save(d / "test/bad" / f"{j:03d}.png", bad[i])
        save(d / "ground_truth/bad" / f"{j:03d}_mask.png", masks[i].astype(np.uint8) * 255)
    return d


def silence_stdout(case):
    """Các vòng chạy in tiến độ tiếng Việt; trong kiểm thử bỏ đi (cũng tránh lỗi mã hóa của console Windows)."""
    import contextlib
    import io
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    case.addCleanup(stack.close)
