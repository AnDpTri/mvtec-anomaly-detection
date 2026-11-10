"""Màu, thứ tự nhãn và hàm vẽ chồng ảnh dùng chung cho các biểu đồ."""
import cv2
import numpy as np

COLORS = dict(ae="tab:green", padim="tab:blue", patchcore="tab:red")
SHOT_ORDER = ["1", "2", "4", "8", "16", "all"]
COND_ORDER = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4"]


def overlay(img, heat, vmin, vmax):
    """Chồng bản đồ nhiệt (thang vmin..vmax, JET) lên ảnh uint8."""
    h = np.clip((heat - vmin) / (vmax - vmin + 1e-8), 0, 1)
    h = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * h).astype(np.uint8)


def mask_overlay(img: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Tô đỏ vùng lỗi (ground truth bool) lên ảnh uint8."""
    g = img.copy()
    g[gt] = (0.5 * g[gt] + [127, 0, 0]).astype(np.uint8)
    return g
