"""[An] Autoencoder (baseline chính thức).

Mục tiêu: học nén rồi vẽ lại ảnh tốt. Khi gặp ảnh lỗi, mạng vẽ lại thành phiên bản
không lỗi, nên chỗ khác biệt |ảnh gốc − ảnh vẽ lại| chính là vùng lỗi.

Lý do dùng CNN thay vì fully connected: CNN biết pixel nào nằm cạnh pixel nào và dùng chung
bộ lọc khắp ảnh, nên vẽ lại sắc nét hơn (so sánh với bản NumPy trong src/autoencoder/numpy_ae/).

Thiết kế: tự do (kiến trúc, cỡ cổ chai, loss, thư viện). Điểm xuất phát gợi ý theo bài gốc MVTec AD:
encoder gồm các lớp Conv stride 2 thu nhỏ ảnh xuống một "cổ chai", decoder gồm ConvTranspose phóng
ngược lại; loss L2 hoặc SSIM. Yêu cầu duy nhất: anomaly map đưa vào đánh giá có kích thước 256×256.

Tài liệu: docs/phuong-phap/autoencoder.md. Đầu ra đưa vào src/evaluation: docs/giao-dien-chung.md (mục 3).
"""
import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_channels=64):
        super().__init__()
        # TODO: encoder + decoder, kiến trúc tự chọn (gợi ý: Conv stride 2 / ConvTranspose)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def anomaly_map(model, images):
    """Bản đồ bất thường: lỗi tái tạo theo pixel, trung bình theo kênh → B×1×H×W.

    Sau đó src.evaluation.metrics làm mượt Gaussian; điểm ảnh = max của bản đồ.
    """
    with torch.no_grad():
        recon = model(images)
    return ((images - recon) ** 2).mean(dim=1, keepdim=True)
