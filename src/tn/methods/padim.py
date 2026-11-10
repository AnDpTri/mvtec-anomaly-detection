"""PaDiM: Gaussian đa biến ở mỗi vị trí của lưới đặc trưng (+0,01·I), điểm = khoảng cách Mahalanobis."""
import numpy as np
import torch

from src.tn import config as cfg
from src.tn.config import SQUARE
from src.tn.methods.base import FeatureDetector
from src.tn.methods.ops import postprocess


class PaDiMDetector(FeatureDetector):
    name = "padim"
    filename = "padim.pt"

    def __init__(self, mu: torch.Tensor, inv: torch.Tensor, H: int, W: int):
        self.mu, self.inv, self.H, self.W = mu, inv, H, W      # mu (H·W,D), inv (H·W,D,D) trên thiết bị

    @classmethod
    @torch.no_grad()
    def fit_features(cls, feat, seed=0, coreset=None, chunk=4096):
        device = cfg.DEVICE
        N, D, H, W = feat.shape
        X = feat.permute(2, 3, 0, 1).reshape(H * W, N, D)                       # CPU fp16
        mu = torch.empty(H * W, D, device=device)
        inv = torch.empty(H * W, D, D, device=device)
        eye = 0.01 * torch.eye(D, device=device)
        for j in range(0, H * W, chunk):                                         # theo khối vị trí để tiết kiệm VRAM
            x = X[j:j + chunk].to(device).float()
            m = x.mean(1)
            xc = x - m[:, None]
            cov = xc.transpose(1, 2) @ xc / max(N - 1, 1) + eye
            mu[j:j + chunk] = m
            inv[j:j + chunk] = torch.linalg.inv(cov.double().cpu()).float().to(device)   # nghịch đảo trên CPU (tránh lỗi CUDA)
        return cls(mu, inv, H, W)

    @torch.no_grad()
    def predict_features(self, feat, hw=SQUARE, bs=16):
        maps = []
        for i in range(0, len(feat), bs):
            T = feat[i:i + bs].to(cfg.DEVICE).float()
            B, D = T.shape[:2]
            d = T.permute(2, 3, 0, 1).reshape(self.H * self.W, B, D) - self.mu[:, None]
            m = (torch.bmm(d, self.inv) * d).sum(-1).clamp_min(0).sqrt()
            maps.append(postprocess(m.T.reshape(B, self.H, self.W), hw))
        return np.concatenate(maps)

    def state(self):
        return dict(mu=self.mu.cpu(), inv=self.inv.cpu(), H=self.H, W=self.W)

    @classmethod
    def from_state(cls, state):
        return cls(state["mu"].to(cfg.DEVICE), state["inv"].to(cfg.DEVICE), state["H"], state["W"])
