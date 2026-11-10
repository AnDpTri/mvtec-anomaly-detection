"""PatchCore: kho patch của ảnh good (greedy coreset), điểm = khoảng cách tới patch gần nhất."""
import math

import numpy as np
import torch

from src.tn import config as cfg
from src.tn.config import SQUARE
from src.tn.methods.base import FeatureDetector
from src.tn.methods.ops import postprocess


@torch.no_grad()
def greedy_coreset(feats_cpu: torch.Tensor, n_select: int, seed: int = 0, proj_dim: int = 128,
                   chunk: int = 200_000) -> torch.Tensor:
    """feats_cpu (M,C) fp16 CPU -> chỉ số được chọn. Chiếu ngẫu nhiên xuống 128 chiều rồi chọn greedy trên GPU."""
    device = cfg.DEVICE
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(feats_cpu.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)).to(device)
    z = torch.cat([feats_cpu[j:j + chunk].to(device).float() @ proj for j in range(0, len(feats_cpu), chunk)])
    idx = int(torch.randint(len(z), (1,), generator=g))
    sel = [idx]
    min_d = torch.linalg.norm(z - z[idx], dim=1)
    for _ in range(n_select - 1):
        idx = int(torch.argmax(min_d))
        sel.append(idx)
        min_d = torch.minimum(min_d, torch.linalg.norm(z - z[idx], dim=1))
    return torch.tensor(sel)


class PatchCoreDetector(FeatureDetector):
    name = "patchcore"
    filename = "patchcore_bank.pt"

    def __init__(self, bank: torch.Tensor):
        self.bank = bank                                                # (M,PC_DIM) float32 trên thiết bị

    @classmethod
    @torch.no_grad()
    def fit_features(cls, feat, seed=0, coreset=None):
        ratio = cfg.PC_CORESET if coreset is None else coreset
        flat = feat.reshape(-1, cfg.PC_DIM)                              # CPU fp16
        if len(flat) > cfg.PC_CORESET_MIN_BANK:
            flat = flat[greedy_coreset(flat, max(1, int(ratio * len(flat))), seed=seed)]
        return cls(flat.to(cfg.DEVICE).float())

    @torch.no_grad()
    def predict_features(self, feat, hw=SQUARE, bs=8, q_chunk=4096):
        maps = []
        for i in range(0, len(feat), bs):
            T = feat[i:i + bs].to(cfg.DEVICE).float()
            B, H, W, C = T.shape
            q = T.reshape(-1, C)
            d = torch.cat([torch.cdist(q[j:j + q_chunk], self.bank).min(1).values for j in range(0, len(q), q_chunk)])
            maps.append(postprocess(d.reshape(B, H, W), hw))
        return np.concatenate(maps)

    def state(self):
        return self.bank.half().cpu()

    @classmethod
    def from_state(cls, state):
        return cls(state.to(cfg.DEVICE).float())
