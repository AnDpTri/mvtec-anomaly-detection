"""Ba phương pháp, mỗi cái tách thành fit / predict để train một lần rồi chấm trên nhiều tập test.

- Backbone chung cho PaDiM và PatchCore: WideResNet-50-2 pretrained ImageNet; đặc trưng lưu fp16 trên CPU.
- PaDiM: layer1–3 ở lưới H/4×W/4, chọn ngẫu nhiên 100 chiều, Gaussian mỗi vị trí (+0,01·I), Mahalanobis.
- PatchCore: layer2–3, gộp lân cận 3×3, đưa về 1024 chiều ở lưới H/8×W/8, greedy coreset (chiếu ngẫu nhiên
  xuống 128 chiều), khoảng cách tới patch gần nhất.
- Autoencoder: conv AE, bottleneck fully-connected 256 chiều, MSE, số bước cố định; đầu vào luôn 256×256.

Anomaly map của mọi phương pháp được nội suy về kích thước ảnh rồi làm mượt Gauss σ = BLUR_SIGMA.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.tn.config import (AE_BATCH, AE_LR, AE_STEPS, BLUR_SIGMA, DEVICE, PADIM_DIM, PADIM_IDX_SEED, PC_CORESET,
                           PC_CORESET_MIN_BANK, PC_DIM, SQUARE)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KSIZE = 2 * int(4 * BLUR_SIGMA + 0.5) + 1
PADIM_IDX = torch.from_numpy(np.random.default_rng(PADIM_IDX_SEED).choice(256 + 512 + 1024, PADIM_DIM, replace=False))


def postprocess(raw, hw=SQUARE):
    """raw (B,h,w) -> anomaly map (B,H,W) đã làm mượt, numpy float32."""
    m = F.interpolate(raw[:, None].float(), size=tuple(hw), mode="bilinear", align_corners=False)
    m = TF.gaussian_blur(m, kernel_size=KSIZE, sigma=BLUR_SIGMA)
    return m[:, 0].cpu().numpy().astype(np.float32)


def to_tensor(imgs_uint8):
    return torch.from_numpy(np.ascontiguousarray(imgs_uint8)).permute(0, 3, 1, 2).float().div(255)


# ----------------------------------------------------------------- backbone + đặc trưng
class Backbone(nn.Module):
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


def get_backbone():
    global _BACKBONE
    if _BACKBONE is None:
        _BACKBONE = Backbone().eval().to(DEVICE)
    return _BACKBONE


@torch.no_grad()
def extract_features(imgs, bs=32, which=("padim", "patchcore")):
    """imgs uint8 (N,H,W,3) -> padim (N,100,H/4,W/4), patchcore (N,H/8,W/8,1024), fp16 trên CPU.
    Phần không có trong `which` trả về None."""
    net = get_backbone()
    pad, pc = [], []
    bs = max(1, int(bs * (256 * 256) / (imgs.shape[1] * imgs.shape[2])))      # ảnh lớn -> batch nhỏ hơn
    for i in range(0, len(imgs), bs):
        x = ((to_tensor(imgs[i:i + bs]) - MEAN) / STD).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
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
            p = F.adaptive_avg_pool1d(p.permute(0, 2, 3, 1).reshape(-1, 1, C), PC_DIM).reshape(B, H, W, PC_DIM)
            pc.append(p.half().cpu())
    return (torch.cat(pad) if pad else None), (torch.cat(pc) if pc else None)


# ----------------------------------------------------------------- PaDiM
@torch.no_grad()
def padim_fit(feat_train, chunk=4096):
    N, D, H, W = feat_train.shape
    X = feat_train.permute(2, 3, 0, 1).reshape(H * W, N, D)                 # CPU fp16
    mu = torch.empty(H * W, D, device=DEVICE)
    inv = torch.empty(H * W, D, D, device=DEVICE)
    eye = 0.01 * torch.eye(D, device=DEVICE)
    for j in range(0, H * W, chunk):                                         # theo khối vị trí để tiết kiệm VRAM
        x = X[j:j + chunk].to(DEVICE).float()
        m = x.mean(1)
        xc = x - m[:, None]
        cov = xc.transpose(1, 2) @ xc / max(N - 1, 1) + eye
        mu[j:j + chunk] = m
        inv[j:j + chunk] = torch.linalg.inv(cov.double().cpu()).float().to(DEVICE)   # nghịch đảo trên CPU (tránh lỗi CUDA)
    return dict(mu=mu, inv=inv, H=H, W=W)


@torch.no_grad()
def padim_predict(st, feat_test, hw=SQUARE, bs=16):
    H, W, mu, inv = st["H"], st["W"], st["mu"], st["inv"]
    maps = []
    for i in range(0, len(feat_test), bs):
        T = feat_test[i:i + bs].to(DEVICE).float()
        B, D = T.shape[:2]
        d = T.permute(2, 3, 0, 1).reshape(H * W, B, D) - mu[:, None]
        m = (torch.bmm(d, inv) * d).sum(-1).clamp_min(0).sqrt()
        maps.append(postprocess(m.T.reshape(B, H, W), hw))
    return np.concatenate(maps)


# ----------------------------------------------------------------- PatchCore
@torch.no_grad()
def greedy_coreset(feats_cpu, n_select, seed=0, proj_dim=128, chunk=200_000):
    """feats_cpu (M,C) fp16 CPU -> chỉ số được chọn. Chiếu ngẫu nhiên xuống 128 chiều rồi chọn greedy trên GPU."""
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(feats_cpu.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)).to(DEVICE)
    z = torch.cat([feats_cpu[j:j + chunk].to(DEVICE).float() @ proj for j in range(0, len(feats_cpu), chunk)])
    idx = int(torch.randint(len(z), (1,), generator=g))
    sel = [idx]
    min_d = torch.linalg.norm(z - z[idx], dim=1)
    for _ in range(n_select - 1):
        idx = int(torch.argmax(min_d))
        sel.append(idx)
        min_d = torch.minimum(min_d, torch.linalg.norm(z - z[idx], dim=1))
    return torch.tensor(sel)


@torch.no_grad()
def patchcore_fit(feat_train, seed=0, ratio=None):
    ratio = PC_CORESET if ratio is None else ratio
    flat = feat_train.reshape(-1, PC_DIM)                                    # CPU fp16
    if len(flat) > PC_CORESET_MIN_BANK:
        flat = flat[greedy_coreset(flat, max(1, int(ratio * len(flat))), seed=seed)]
    return flat.to(DEVICE).float()


@torch.no_grad()
def patchcore_predict(bank, feat_test, hw=SQUARE, bs=8, q_chunk=4096):
    maps = []
    for i in range(0, len(feat_test), bs):
        T = feat_test[i:i + bs].to(DEVICE).float()
        B, H, W, C = T.shape
        q = T.reshape(-1, C)
        d = torch.cat([torch.cdist(q[j:j + q_chunk], bank).min(1).values for j in range(0, len(q), q_chunk)])
        maps.append(postprocess(d.reshape(B, H, W), hw))
    return np.concatenate(maps)


# ----------------------------------------------------------------- Autoencoder
class ConvAE(nn.Module):
    """Đầu vào cố định 256×256 (ảnh kích thước khác được resize trước khi vào AE)."""

    def __init__(self, latent=256):
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


def _ae_in(x):
    return x if tuple(x.shape[-2:]) == SQUARE else F.interpolate(x, size=SQUARE, mode="bilinear", align_corners=False)


def ae_fit(imgs_train, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    Xtr = to_tensor(imgs_train)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(AE_STEPS):
        idx = torch.randint(len(Xtr), (AE_BATCH,), generator=g)              # có hoàn lại -> chạy được cả khi k=1
        x = _ae_in(Xtr[idx].to(DEVICE))
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            loss = F.mse_loss(model(x).float(), x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return model.eval()


@torch.no_grad()
def ae_predict(model, imgs, bs=32):
    hw = imgs.shape[1:3]
    maps = []
    for i in range(0, len(imgs), bs):
        x = _ae_in(to_tensor(imgs[i:i + bs]).to(DEVICE))
        maps.append(postprocess(((model(x) - x) ** 2).mean(1), hw))
    return np.concatenate(maps)


# ----------------------------------------------------------------- giao diện chung
def fit_method(method, imgs_train, seed=0, coreset=None):
    """-> (tên phương pháp, trạng thái đã fit)."""
    if method == "ae":
        return "ae", ae_fit(imgs_train, seed=seed)
    pad, pc = extract_features(imgs_train, which=(method,))
    if method == "padim":
        return "padim", padim_fit(pad)
    return "patchcore", patchcore_fit(pc, seed=seed, ratio=coreset)


def predict_method(model, imgs):
    """Anomaly map cùng kích thước với `imgs`."""
    kind, st = model
    if kind == "ae":
        return ae_predict(st, imgs)
    pad, pc = extract_features(imgs, which=(kind,))
    hw = imgs.shape[1:3]
    return padim_predict(st, pad, hw) if kind == "padim" else patchcore_predict(st, pc, hw)


# ----------------------------------------------------------------- lưu / nạp mô hình đã fit
MODEL_FILES = {"ae": "ae.pt", "padim": "padim.pt", "patchcore": "patchcore_bank.pt"}


def save_model(model, folder):
    """Ghi trạng thái đã fit vào folder/<tên file> (định dạng mà demo nạp lại, xem MODEL_FILES)."""
    kind, st = model
    if kind == "ae":
        torch.save(st.state_dict(), Path(folder) / MODEL_FILES[kind])
    elif kind == "padim":
        torch.save(dict(mu=st["mu"].cpu(), inv=st["inv"].cpu(), H=st["H"], W=st["W"]), Path(folder) / MODEL_FILES[kind])
    else:
        torch.save(st.half().cpu(), Path(folder) / MODEL_FILES[kind])


def load_model(folder, kind):
    """Nạp mô hình đã lưu bằng save_model -> (tên phương pháp, trạng thái) dùng được với predict_method."""
    st = torch.load(Path(folder) / MODEL_FILES[kind], map_location="cpu", weights_only=True)
    if kind == "ae":
        net = ConvAE()
        net.load_state_dict(st)
        return "ae", net.to(DEVICE).eval()
    if kind == "padim":
        return "padim", dict(mu=st["mu"].to(DEVICE), inv=st["inv"].to(DEVICE), H=st["H"], W=st["W"])
    return "patchcore", st.to(DEVICE).float()


def run_on_features(method, feat_train, feat_test, seed=0):
    """PaDiM/PatchCore trên đặc trưng đã trích sẵn (TN1, TN2, TN2b trích một lần cho mỗi category)."""
    if method == "padim":
        return padim_predict(padim_fit(feat_train), feat_test)
    return patchcore_predict(patchcore_fit(feat_train, seed=seed), feat_test)


def free_gpu():
    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
