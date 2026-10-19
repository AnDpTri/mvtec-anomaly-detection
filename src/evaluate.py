"""[Dưỡng] Đánh giá chung cho cả 3 phương pháp.

Mục tiêu: một thước đo thống nhất. Mọi phương pháp chỉ cần đưa vào anomaly map (H×W)
và điểm ảnh; mọi con số trong báo cáo đều tính qua file này.

- image-AUROC: phân biệt ảnh tốt / ảnh lỗi, dựa trên điểm ảnh.
- pixel-AUROC: so anomaly map với mask, từng pixel.
- Tính riêng từng loại rồi mới lấy trung bình (không gộp 15 loại).
- Kiểm tra trên một trường hợp biết trước kết quả trước khi dùng thật.
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
    """Làm mượt anomaly map trước khi lấy điểm ảnh = max."""
    raise NotImplementedError


def save_figure(image, mask, anomaly_map, out_path, vmin=None, vmax=None):
    """Vẽ: ảnh gốc | mask | heatmap chồng ảnh, cùng một thang màu cho mọi phương pháp."""
    raise NotImplementedError


def append_result(csv_path, row):
    """Ghi một dòng kết quả vào CSV chung.

    Cột: category, type, method, image_auroc, pixel_auroc, train_time, infer_time_per_img, memory
    """
    raise NotImplementedError
