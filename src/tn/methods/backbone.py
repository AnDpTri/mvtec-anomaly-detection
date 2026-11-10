"""Backbone chung cho PaDiM và PatchCore: WideResNet-50-2 pretrained ImageNet; đặc trưng lưu fp16 trên CPU.

- PaDiM: layer1–3 ở lưới H/4×W/4, chọn ngẫu nhiên PADIM_DIM chiều từ 256 + 512 + 1024.
- PatchCore: layer2–3, gộp lân cận 3×3, đưa về PC_DIM chiều ở lưới H/8×W/8.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.tn import config as cfg
from src.tn.methods.ops import to_tensor

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PADIM_IDX = torch.from_numpy(
    np.random.default_rng(cfg.PADIM_IDX_SEED).choice(256 + 512 + 1024, cfg.PADIM_DIM, replace=False))


class Backbone(nn.Module):
    """WideResNet-50-2 cắt ở layer3, trả về đặc trưng layer1, layer2, layer3."""

    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3 = m.layer1, m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        return f1, f2, self.l3(f2)


_BACKBONE = None


def get_backbone() -> Backbone:
    """Backbone dùng chung, tạo một lần ở lần gọi đầu."""
    global _BACKBONE
    if _BACKBONE is None:
        _BACKBONE = Backbone().eval().to(cfg.DEVICE)
    return _BACKBONE


@torch.no_grad()
def extract_features(imgs: np.ndarray, bs: int = 32, which=("padim", "patchcore")):
    """imgs uint8 (N,H,W,3) -> padim (N,100,H/4,W/4), patchcore (N,H/8,W/8,1024), fp16 trên CPU.
    Phần không có trong `which` trả về None."""
    net = get_backbone()
    device = cfg.DEVICE
    pad, pc = [], []
    bs = max(1, int(bs * (256 * 256) / (imgs.shape[1] * imgs.shape[2])))      # ảnh lớn -> batch nhỏ hơn
    for i in range(0, len(imgs), bs):
        x = ((to_tensor(imgs[i:i + bs]) - MEAN) / STD).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f1, f2, f3 = net(x)
        f1, f2, f3 = f1.float(), f2.float(), f3.float()
        if "padim" in which:
            s = f1.shape[-2:]
            e = torch.cat([f1, F.interpolate(f2, size=s, mode="nearest"), F.interpolate(f3, size=s, mode="nearest")], 1)
            pad.append(e[:, PADIM_IDX.to(e.device)].half().cpu())
        if "patchcore" in which:
            p2, p3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)
            p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            p = torch.cat([p2, p3], 1)
            B, C, H, W = p.shape
            p = F.adaptive_avg_pool1d(p.permute(0, 2, 3, 1).reshape(-1, 1, C), cfg.PC_DIM).reshape(B, H, W, cfg.PC_DIM)
            pc.append(p.half().cpu())
    return (torch.cat(pad) if pad else None), (torch.cat(pc) if pc else None)


def extract_one(imgs: np.ndarray, kind: str) -> torch.Tensor:
    """Đặc trưng của riêng một phương pháp (`kind` = 'padim' hoặc 'patchcore')."""
    pad, pc = extract_features(imgs, which=(kind,))
    return pad if kind == "padim" else pc
