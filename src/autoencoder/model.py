"""[An] Autoencoder (baseline chính thức).

Mục tiêu: học nén rồi vẽ lại ảnh tốt. Khi gặp ảnh lỗi, mạng vẽ lại thành phiên bản
không lỗi, nên chỗ khác biệt |ảnh gốc − ảnh vẽ lại| chính là vùng lỗi.

Lý do dùng CNN thay vì fully connected: CNN biết pixel nào nằm cạnh pixel nào và dùng chung
bộ lọc khắp ảnh, nên vẽ lại sắc nét hơn (so sánh với bản NumPy trong src/autoencoder/numpy_ae/).

Thiết kế (theo hướng bài gốc MVTec AD, các tham số đều đổi được):
- Encoder: 5 lớp Conv stride 2 thu ảnh 3×256×256 về 256×8×8, rồi một lớp Linear nén thành vector
  `latent_dim` chiều. Cổ chai dạng vector (không phải bản đồ 8×8×C) để mạng khó chép nguyên lỗi.
- Decoder: Linear bung lại 256×8×8, rồi 5 lớp ConvTranspose phóng về 3×256×256, sigmoid ở đầu ra
  (ảnh đầu vào ở thang [0, 1], không chuẩn hóa ImageNet).
- Anomaly map: lỗi tái tạo theo pixel, trung bình theo kênh (loss "mse"), hoặc 1 − SSIM cục bộ
  (loss "ssim").

Tài liệu: docs/phuong-phap/autoencoder.md. Đầu ra đưa vào src/evaluation: docs/giao-dien-chung.md (mục 3).
"""
import torch
import torch.nn.functional as F
from torch import nn


def _down(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))


def _up(c_in, c_out):
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=256, channels=(32, 64, 128, 256, 256)):
        super().__init__()
        self.feat_size = image_size // 2 ** len(channels)
        self.feat_ch = channels[-1]
        flat = self.feat_ch * self.feat_size ** 2

        enc, c_prev = [], 3
        for c in channels:
            enc.append(_down(c_prev, c))
            c_prev = c
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)

        dec, rev = [], list(channels[::-1])
        for c_in, c_out in zip(rev, rev[1:]):
            dec.append(_up(c_in, c_out))
        dec += [nn.ConvTranspose2d(rev[-1], 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z):
        h = self.from_latent(z).view(-1, self.feat_ch, self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))


def ssim_map(x, y, window=11):
    """SSIM cục bộ theo pixel (trung bình theo kênh), ảnh trong [0, 1] → B×1×H×W."""
    c1, c2, pad = 0.01 ** 2, 0.03 ** 2, window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    var_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x ** 2
    var_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y ** 2
    cov = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return s.mean(dim=1, keepdim=True)


def reconstruction_loss(x, recon, loss="mse"):
    if loss == "mse":
        return F.mse_loss(recon, x)
    if loss == "ssim":
        return (1 - ssim_map(x, recon)).mean()
    raise ValueError(f"loss không hỗ trợ: {loss}")


@torch.no_grad()
def anomaly_map(model, images, loss="mse"):
    """Bản đồ bất thường B×1×H×W từ lỗi tái tạo (cùng loại với loss đã dùng khi huấn luyện)."""
    recon = model(images)
    if loss == "ssim":
        return 1 - ssim_map(images, recon)
    return ((images - recon) ** 2).mean(dim=1, keepdim=True)
