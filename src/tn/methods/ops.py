"""Phép tensor chung của ba phương pháp: đổi ảnh uint8 sang tensor và làm mượt anomaly map.

Anomaly map của mọi phương pháp được nội suy về kích thước ảnh rồi làm mượt Gauss σ = BLUR_SIGMA.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.tn import config as cfg
from src.tn.config import SQUARE

KSIZE = 2 * int(4 * cfg.BLUR_SIGMA + 0.5) + 1


def to_tensor(imgs_uint8: np.ndarray) -> torch.Tensor:
    """Ảnh uint8 (N,H,W,3) -> float (N,3,H,W) trong [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(imgs_uint8)).permute(0, 3, 1, 2).float().div(255)


def postprocess(raw: torch.Tensor, hw=SQUARE) -> np.ndarray:
    """raw (B,h,w) -> anomaly map (B,H,W) đã làm mượt, numpy float32."""
    m = F.interpolate(raw[:, None].float(), size=tuple(hw), mode="bilinear", align_corners=False)
    m = TF.gaussian_blur(m, kernel_size=KSIZE, sigma=cfg.BLUR_SIGMA)
    return m[:, 0].cpu().numpy().astype(np.float32)
