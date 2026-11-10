"""Autoencoder tích chập: bottleneck fully-connected 256 chiều, MSE, số bước cố định; đầu vào luôn 256×256.

Anomaly map = bình phương sai khác giữa ảnh và ảnh tái tạo, trung bình theo kênh.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tn import config as cfg
from src.tn.config import SQUARE
from src.tn.methods.base import Detector
from src.tn.methods.ops import postprocess, to_tensor


class ConvAE(nn.Module):
    """Đầu vào cố định 256×256 (ảnh kích thước khác được resize trước khi vào AE).

    Tên lớp con (enc, fc1, fc2, dec) quyết định khóa state_dict trong ae.pt; đổi tên là hỏng mô hình đã lưu.
    """

    def __init__(self, latent: int = 256):
        super().__init__()
        ch = [3, 32, 64, 128, 256, 256]
        enc = []
        for a, b in zip(ch[:-1], ch[1:]):
            enc += [nn.Conv2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.LeakyReLU(0.2, True)]
        self.enc = nn.Sequential(*enc)                                       # 256 -> 8
        self.fc1 = nn.Linear(256 * 8 * 8, latent)
        self.fc2 = nn.Linear(latent, 256 * 8 * 8)
        dec, rch = [], ch[::-1]
        for a, b in zip(rch[:-2], rch[1:-1]):
            dec += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(True)]
        dec += [nn.ConvTranspose2d(rch[-2], 3, 4, 2, 1), nn.Sigmoid()]
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        z = self.fc1(self.enc(x).flatten(1))
        return self.dec(self.fc2(z).view(-1, 256, 8, 8))


def _to_input_size(x: torch.Tensor) -> torch.Tensor:
    return x if tuple(x.shape[-2:]) == SQUARE else F.interpolate(x, size=SQUARE, mode="bilinear", align_corners=False)


class AEDetector(Detector):
    name = "ae"
    filename = "ae.pt"

    def __init__(self, net: ConvAE):
        self.net = net

    @classmethod
    def fit(cls, imgs, seed=0, coreset=None):
        device = cfg.DEVICE
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ConvAE().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.AE_LR)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        Xtr = to_tensor(imgs)
        g = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(cfg.AE_STEPS):
            idx = torch.randint(len(Xtr), (cfg.AE_BATCH,), generator=g)      # có hoàn lại -> chạy được cả khi k=1
            x = _to_input_size(Xtr[idx].to(device))
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = F.mse_loss(model(x).float(), x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        return cls(model.eval())

    @torch.no_grad()
    def predict(self, imgs, bs=32):
        hw = imgs.shape[1:3]
        maps = []
        for i in range(0, len(imgs), bs):
            x = _to_input_size(to_tensor(imgs[i:i + bs]).to(cfg.DEVICE))
            maps.append(postprocess(((self.net(x) - x) ** 2).mean(1), hw))
        return np.concatenate(maps)

    def state(self):
        return self.net.state_dict()

    @classmethod
    def from_state(cls, state):
        net = ConvAE()
        net.load_state_dict(state)
        return cls(net.to(cfg.DEVICE).eval())
