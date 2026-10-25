"""[Dưỡng] Độ đo chung cho cả 3 phương pháp.

Mục tiêu: một thước đo thống nhất. Mọi phương pháp chỉ cần đưa vào anomaly map (N×H×W)
và điểm ảnh; mọi con số trong báo cáo đều tính qua module này.

- image-AUROC: phân biệt ảnh tốt / ảnh lỗi, dựa trên điểm ảnh.
- pixel-AUROC: so anomaly map với mask, từng pixel.
- Tính riêng từng loại rồi mới lấy trung bình (không gộp 15 loại).

Định dạng đầu vào: docs/giao-dien-chung.md (mục 3). Giải thích độ đo: docs/danh-gia.md.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    """labels: (N,) 0/1; scores: (N,) điểm bất thường của từng ảnh."""
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def pixel_auroc(masks, maps):
    """masks: (N, H, W) 0/1; maps: (N, H, W) anomaly map cùng kích thước."""
    masks, maps = np.asarray(masks), np.asarray(maps)
    if masks.shape != maps.shape:
        raise ValueError(f"mask {masks.shape} và anomaly map {maps.shape} phải cùng kích thước")
    return float(roc_auc_score(masks.ravel().astype(np.uint8), maps.ravel()))


def gaussian_smooth(anomaly_map, sigma=4):
    """Làm mượt một anomaly map H×W (hoặc N×H×W, làm mượt từng ảnh)."""
    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.ndim == 3:
        return np.stack([gaussian_filter(x, sigma=sigma) for x in m])
    return gaussian_filter(m, sigma=sigma)


def image_score(anomaly_map, sigma=4):
    """Cách tính điểm ảnh mặc định: max của anomaly map đã làm mượt.

    Chỉ là mặc định cho phương pháp nào không có cách riêng; phương pháp có cách tính điểm riêng
    (ví dụ PatchCore hiệu chỉnh theo mật độ láng giềng) thì tự đưa image_scores vào.
    """
    return float(np.max(gaussian_smooth(anomaly_map, sigma)))


def evaluate(labels, scores, masks, maps):
    """Trả về dict {image_auroc, pixel_auroc} cho một loại sản phẩm."""
    return {"image_auroc": image_auroc(labels, scores), "pixel_auroc": pixel_auroc(masks, maps)}
