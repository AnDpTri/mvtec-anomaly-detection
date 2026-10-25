"""[Dưỡng] Vẽ hình kết quả: ảnh gốc | mask | heatmap chồng ảnh.

Dùng cùng một thang màu (vmin, vmax) cho mọi ảnh của một loại để hình so sánh được.
Hình lưu ở results/figures/<method>/<category>/ (không đưa lên git).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def to_display(image, mean=None, std=None):
    """Tensor/mảng 3×H×W (có thể đã chuẩn hóa) → mảng H×W×3 trong [0, 1] để vẽ."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    if mean is not None and std is not None:
        img = img * np.asarray(std) + np.asarray(mean)
    return np.clip(img, 0, 1)


def save_figure(image, mask, anomaly_map, out_path, vmin=None, vmax=None, title=None):
    """image: H×W×3 trong [0, 1]; mask, anomaly_map: H×W."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("Ảnh")
    axes[1].imshow(np.asarray(mask).squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask lỗi")
    axes[2].imshow(image)
    axes[2].imshow(np.asarray(anomaly_map).squeeze(), cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    axes[2].set_title("Heatmap")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_examples(dataset, maps, out_dir, n_per_type=1, mean=None, std=None):
    """Lưu n ảnh đầu tiên của mỗi kiểu lỗi (kể cả good) với cùng thang màu."""
    maps = np.asarray(maps)
    vmin, vmax = float(np.percentile(maps, 1)), float(np.percentile(maps, 99.5))
    seen = {}
    for i, (path, defect_type, _) in enumerate(dataset.samples):
        if seen.get(defect_type, 0) >= n_per_type:
            continue
        seen[defect_type] = seen.get(defect_type, 0) + 1
        item = dataset[i]
        save_figure(to_display(item["image"], mean, std), item["mask"], maps[i],
                    Path(out_dir) / f"{defect_type}_{Path(path).stem}.png",
                    vmin=vmin, vmax=vmax, title=f"{dataset.category} / {defect_type}")
