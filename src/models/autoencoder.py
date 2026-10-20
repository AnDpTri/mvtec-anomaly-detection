"""[An] Convolutional autoencoder bằng PyTorch (baseline chính thức).

Mục tiêu: học nén rồi vẽ lại ảnh tốt. Khi gặp ảnh lỗi, mạng vẽ lại thành phiên bản
không lỗi, nên chỗ khác biệt |ảnh gốc − ảnh vẽ lại| chính là vùng lỗi.

Lý do dùng CNN thay vì fully connected: CNN biết pixel nào nằm cạnh pixel nào và dùng chung
bộ lọc khắp ảnh, nên vẽ lại sắc nét hơn (so sánh với bản NumPy trong numpy_ae/).

Thiết kế (thẻ Trello "Học PyTorch, viết khung autoencoder"):
- Encoder: các lớp Conv stride 2 thu nhỏ 3×256×256 xuống một "cổ chai" nhỏ.
- Decoder: các lớp ConvTranspose phóng ngược về 3×256×256.

Tài liệu: docs/phuong-phap/autoencoder.md. Đầu ra đưa vào evaluate.py: docs/giao-dien-chung.md (mục 3).
"""
import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_channels=64):
        super().__init__()
        # TODO: encoder: Conv2d(stride=2) + BatchNorm + ReLU, lặp vài lần
        # TODO: decoder: ConvTranspose2d(stride=2) + BatchNorm + ReLU, lớp cuối không ReLU
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def anomaly_map(model, images):
    """Bản đồ bất thường: lỗi tái tạo theo pixel, trung bình theo kênh → B×1×H×W.

    Sau đó evaluate.py làm mượt Gaussian; điểm ảnh = max của bản đồ.
    """
    with torch.no_grad():
        recon = model(images)
    return ((images - recon) ** 2).mean(dim=1, keepdim=True)
