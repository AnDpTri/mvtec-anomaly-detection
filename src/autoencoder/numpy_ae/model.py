"""[An] Autoencoder viết hoàn toàn bằng NumPy (tùy chọn, để hiểu tận gốc).

Mỗi tầng: h = f(W·x + b). Học bằng: loss → lan truyền ngược (chain rule) → bộ tối ưu.
Kiến trúc, kích thước ảnh, loss, bộ tối ưu: tự chọn; giá trị mặc định dưới đây chỉ là gợi ý.
Xem src/autoencoder/numpy_ae/README.md cho mục tiêu và thiết kế.
"""
import numpy as np


def load_images(folder, size=128):
    """Đọc ảnh, chuyển xám, resize size×size, trải phẳng thành (N, size*size), giá trị [0, 1]."""
    raise NotImplementedError


class NumpyAutoencoder:
    def __init__(self, sizes=(16384, 512, 64, 512, 16384), seed=42):
        # TODO: khởi tạo W (He init) và b cho từng tầng; trạng thái Adam (m, v) cho từng tham số
        raise NotImplementedError

    def forward(self, x):
        """Trả về ảnh vẽ lại, đồng thời lưu giá trị trung gian để lan truyền ngược."""
        raise NotImplementedError

    def backward(self, x, x_hat):
        """Tính gradient của loss theo mọi W, b bằng chain rule."""
        raise NotImplementedError

    def step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Một bước cập nhật Adam."""
        raise NotImplementedError


def gradient_check(model, x, n_checks=10, h=1e-5):
    """So gradient lan truyền ngược với sai phân hữu hạn, để chắc chắn backward đúng."""
    raise NotImplementedError


def auroc(labels, scores):
    """Tự tính AUROC = xác suất một ảnh lỗi ngẫu nhiên có điểm cao hơn một ảnh tốt ngẫu nhiên."""
    raise NotImplementedError
