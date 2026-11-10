"""Ba phương pháp phát hiện bất thường, mỗi phương pháp một file, chung giao diện `Detector`.

    ae          Autoencoder tích chập
    padim       PaDiM (Gaussian + Mahalanobis trên đặc trưng WideResNet-50-2)
    patchcore   PatchCore (kho patch + greedy coreset)
    backbone    WideResNet-50-2 và trích đặc trưng dùng chung cho PaDiM / PatchCore
    base        Detector, FeatureDetector
    ops         đổi ảnh sang tensor, làm mượt anomaly map

Dùng:  det = fit_detector("padim", imgs); maps = det.predict(imgs_test); det.save(folder)
       det = load_detector(folder, "padim")
"""
from src.tn.methods.ae import AEDetector
from src.tn.methods.backbone import extract_features
from src.tn.methods.base import Detector, FeatureDetector
from src.tn.methods.padim import PaDiMDetector
from src.tn.methods.patchcore import PatchCoreDetector

DETECTORS: dict[str, type[Detector]] = {d.name: d for d in (AEDetector, PaDiMDetector, PatchCoreDetector)}


def fit_detector(method: str, imgs, seed: int = 0, coreset: float | None = None) -> Detector:
    """Fit phương pháp `method` ('ae', 'padim', 'patchcore') trên ảnh train uint8 (N,H,W,3)."""
    return DETECTORS[method].fit(imgs, seed=seed, coreset=coreset)


def load_detector(folder, method: str) -> Detector:
    """Nạp mô hình đã lưu bằng `Detector.save` trong `folder`."""
    return DETECTORS[method].load(folder)


def run_on_features(method: str, feat_train, feat_test, seed: int = 0):
    """PaDiM/PatchCore trên đặc trưng đã trích sẵn -> anomaly map (N,256,256)."""
    return DETECTORS[method].fit_features(feat_train, seed=seed).predict_features(feat_test)


__all__ = ["DETECTORS", "AEDetector", "PaDiMDetector", "PatchCoreDetector", "Detector", "FeatureDetector",
           "extract_features", "fit_detector", "load_detector", "run_on_features"]
