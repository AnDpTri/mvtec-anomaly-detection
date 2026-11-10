"""Giao diện chung của ba phương pháp: fit một lần, predict nhiều lần, save/load một file.

Mỗi phương pháp là một lớp con của `Detector`:

    det = PatchCoreDetector.fit(imgs_train, seed=0, coreset=0.1)
    maps = det.predict(imgs_test)                 # (N,H,W) float32, cùng kích thước ảnh
    det.save(folder); PatchCoreDetector.load(folder)

PaDiM và PatchCore dùng đặc trưng backbone nên là `FeatureDetector`: ngoài ra còn fit/predict thẳng trên
đặc trưng đã trích sẵn (TN1, TN2, TN2b trích một lần cho mỗi category rồi fit nhiều lần).
"""
from pathlib import Path

import numpy as np
import torch

from src.tn.config import SQUARE
from src.tn.methods.backbone import extract_one


class Detector:
    """Giao diện chung: lớp con đặt `name`, `filename` và cài fit / predict / state / from_state."""

    name: str = ""            # khóa trong config.METHODS
    filename: str = ""        # tên file mô hình trong thư mục của một loại sản phẩm

    @classmethod
    def fit(cls, imgs: np.ndarray, seed: int = 0, coreset: float | None = None) -> "Detector":
        """imgs uint8 (N,H,W,3) -> mô hình đã fit. `coreset` chỉ có nghĩa với PatchCore."""
        raise NotImplementedError

    def predict(self, imgs: np.ndarray) -> np.ndarray:
        """imgs uint8 (N,H,W,3) -> anomaly map (N,H,W) float32."""
        raise NotImplementedError

    def state(self) -> object:
        """Nội dung ghi vào file (tensor / dict / state_dict), chỉ gồm kiểu mà torch.load(weights_only=True) đọc được."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state) -> "Detector":
        raise NotImplementedError

    def save(self, folder) -> Path:
        """Ghi vào folder/<filename> (định dạng mà demo nạp lại)."""
        path = Path(folder) / self.filename
        torch.save(self.state(), path)
        return path

    @classmethod
    def load(cls, folder) -> "Detector":
        state = torch.load(Path(folder) / cls.filename, map_location="cpu", weights_only=True)
        return cls.from_state(state)


class FeatureDetector(Detector):
    """Phương pháp chạy trên đặc trưng backbone (PaDiM, PatchCore)."""

    @classmethod
    def fit(cls, imgs, seed=0, coreset=None):
        return cls.fit_features(extract_one(imgs, cls.name), seed=seed, coreset=coreset)

    def predict(self, imgs):
        return self.predict_features(extract_one(imgs, self.name), imgs.shape[1:3])

    @classmethod
    def fit_features(cls, feat: torch.Tensor, seed: int = 0, coreset: float | None = None) -> "FeatureDetector":
        raise NotImplementedError

    def predict_features(self, feat: torch.Tensor, hw=SQUARE) -> np.ndarray:
        raise NotImplementedError
