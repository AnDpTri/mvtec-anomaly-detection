"""[Dưỡng] Đọc dữ liệu MVTec AD cho PyTorch.

Mục tiêu: một Dataset chung cho mọi phương pháp tự viết, trả về (ảnh, mask, nhãn) đúng định dạng.

Xử lý (xem docs/du-lieu.md):
- Ảnh xám (grid, screw, zipper) → chuyển sang 3 kênh.
- Ảnh good không có mask → mask toàn 0.
- Resize mask bằng nearest rồi nhị phân hóa lại (> 0.5 → 1).
- Kích thước và chuẩn hóa lấy từ configs/common.yaml; mean/std = None thì giữ giá trị [0, 1].

Định dạng trả về là giao diện chung cho cả nhóm: docs/giao-dien-chung.md (mục 2).
"""
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from src.common.config import ROOT

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def resolve_root(root):
    """Đường dẫn tương đối được tính từ thư mục gốc dự án."""
    root = Path(root)
    return root if root.is_absolute() else ROOT / root


class MVTecDataset(Dataset):
    """Một loại sản phẩm, một split.

    split="train": chỉ ảnh train/good, nhãn 0.
    split="test":  ảnh test/<mọi kiểu>, nhãn 0 (good) hoặc 1 (lỗi), kèm mask.

    __getitem__ trả về dict:
        image: tensor 3×H×W (đã chuẩn hóa nếu có mean/std)
        mask:  tensor 1×H×W, giá trị 0/1
        label: 0 hoặc 1
        defect_type: tên kiểu lỗi ("good" nếu tốt)
        path:  đường dẫn ảnh gốc
    """

    def __init__(self, root, category, split, image_size=256, mean=None, std=None):
        if split not in ("train", "test"):
            raise ValueError(f"split phải là 'train' hoặc 'test', nhận được {split!r}")
        self.root = resolve_root(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.mean, self.std = mean, std

        split_dir = self.root / category / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Không thấy {split_dir}. Tải dữ liệu: python tools/download_mvtec.py")

        defect_dirs = [split_dir / "good"] if split == "train" else sorted(p for p in split_dir.iterdir() if p.is_dir())
        self.samples = []
        for d in defect_dirs:
            for img_path in sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                if d.name == "good":
                    mask_path = None
                else:
                    mask_path = self.root / category / "ground_truth" / d.name / f"{img_path.stem}_mask.png"
                self.samples.append((img_path, d.name, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, defect_type, mask_path = self.samples[idx]
        size = [self.image_size, self.image_size]

        image = Image.open(img_path).convert("RGB")  # ảnh xám → 3 kênh
        image = TF.resize(image, size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        image = TF.to_tensor(image)
        if self.mean is not None and self.std is not None:
            image = TF.normalize(image, self.mean, self.std)

        if mask_path is None:
            mask = torch.zeros(1, *size)
        else:
            mask = Image.open(mask_path).convert("L")
            mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)
            mask = (TF.to_tensor(mask) > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "label": int(defect_type != "good"),
            "defect_type": defect_type,
            "path": str(img_path),
        }


def dataset_from_config(cfg, category, split, normalize=True):
    """Tạo MVTecDataset theo configs/common.yaml."""
    data = cfg["data"]
    mean = data["normalize_mean"] if normalize else None
    std = data["normalize_std"] if normalize else None
    return MVTecDataset(data["root"], category, split, data["image_size"], mean, std)
