"""[Dưỡng] Độ đo chung cho cả 3 phương pháp.

Mục tiêu: một thước đo thống nhất. Mọi phương pháp chỉ cần đưa vào anomaly map (N×H×W)
và điểm ảnh; mọi con số trong báo cáo đều tính qua module này.

- image-AUROC: phân biệt ảnh tốt / ảnh lỗi, dựa trên điểm ảnh.
- pixel-AUROC: so anomaly map với mask, từng pixel.
- Tính riêng từng loại rồi mới lấy trung bình (không gộp 15 loại).
- Kiểm tra trên một trường hợp biết trước kết quả trước khi dùng thật.

Định dạng đầu vào: docs/giao-dien-chung.md (mục 3). Giải thích độ đo: docs/danh-gia.md.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    """labels: (N,) 0/1; scores: (N,) điểm bất thường của từng ảnh."""
    return roc_auc_score(np.asarray(labels), np.asarray(scores))


def pixel_auroc(masks, maps):
    """masks: (N, H, W) 0/1; maps: (N, H, W) anomaly map cùng kích thước."""
    return roc_auc_score(np.asarray(masks).ravel().astype(np.uint8), np.asarray(maps).ravel())


def gaussian_smooth(anomaly_map, sigma=4):
    """Làm mượt anomaly map trước khi lấy điểm ảnh."""
    raise NotImplementedError


def image_score(anomaly_map, sigma=4):
    """Điểm của cả ảnh = max của anomaly map đã làm mượt (mặc định chung cho cả 3 phương pháp)."""
    return float(np.max(gaussian_smooth(anomaly_map, sigma)))
