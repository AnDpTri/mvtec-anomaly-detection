"""[Chung] Chạy PaDiM / PatchCore của Anomalib (mô hình PyTorch) trên một loại MVTec AD.

Dùng trực tiếp lớp `PadimModel` / `PatchcoreModel` của Anomalib 2.7.0, không qua Engine/datamodule,
để ảnh vào là MVTecDataset của dự án (resize 256×256, chuẩn hóa ImageNet, không center crop) và
anomaly map 256×256 khớp từng pixel với mask.

Quy trình cho mỗi loại:
    train/good → forward ở chế độ train (gom embedding) → PaDiM: fit Gaussian; PatchCore: coreset
    → test → anomaly map + điểm ảnh (pred_score của mô hình) → src.evaluation (AUROC)

PatchCore trên GPU 4 GB: embedding gom về CPU, chỉ đưa các đặc trưng đã chiếu ngẫu nhiên lên GPU để
chạy k-center-greedy (cùng thuật toán với Anomalib, chỉ khác nơi lưu), rồi memory bank sau coreset mới
nằm trên GPU.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import ROOT
from src.data.mvtec import MVTecDataset
from src.evaluation.metrics import evaluate
from src.evaluation.visualize import save_examples

DEFAULT_LAYERS = {
    "padim": ["layer1", "layer2", "layer3"],
    "patchcore": ["layer2", "layer3"],
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _coreset(embeddings, ratio, device):
    """k-center-greedy trên `embeddings` (CPU); tính khoảng cách trên `device`. Trả về coreset (CPU)."""
    from anomalib.models.components import KCenterGreedy

    sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=ratio)
    sampler.model.fit(embeddings)
    sampler.features = sampler.model.transform(embeddings).to(device)
    sampler.reset_distances()
    idx = torch.randint(high=sampler.n_observations, size=(1,), device=device).squeeze()
    chosen = [idx]
    for _ in range(sampler.coreset_size - 1):
        sampler.update_distances(cluster_center=idx)
        idx = sampler.get_new_idx()
        sampler.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
        chosen.append(idx)
    chosen = torch.stack(chosen).cpu()
    del sampler
    return embeddings[chosen]


def _build_model(method, backbone, layers, n_features, num_neighbors, pre_trained=True):
    if method == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        return PatchcoreModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
    if method == "padim":
        from anomalib.models.image.padim.torch_model import PadimModel

        return PadimModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
    raise ValueError(f"method phải là 'padim' hoặc 'patchcore', nhận được: {method}")


def run_category(
    method,
    category,
    cfg,
    device,
    backbone,
    layers=None,
    batch_size=16,
    workers=2,
    data_root=None,
    coreset_ratio=0.1,
    n_features=None,
    num_neighbors=9,
    figures=True,
    checkpoint_dir=None,
    calibration_samples=64,
):
    """Huấn luyện + đánh giá `method` ('padim' | 'patchcore') trên một loại.

    Trả về dict: result ({image_auroc, pixel_auroc}), train_time (s), infer_ms (ms/ảnh), memory_mb.
    """
    layers = list(layers or DEFAULT_LAYERS[method])
    data = cfg["data"]
    root = data_root or data["root"]
    size, mean, std = data["image_size"], data["normalize_mean"], data["normalize_std"]
    train_ds = MVTecDataset(root, category, "train", size, mean, std)
    test_ds = MVTecDataset(root, category, "test", size, mean, std)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    model = _build_model(method, backbone, layers, n_features, num_neighbors).to(device)

    # Huấn luyện: không có gradient; backbone luôn ở chế độ eval (giữ nguyên thống kê BatchNorm)
    _sync(device)
    t0 = time.time()
    model.train()
    model.feature_extractor.eval()
    with torch.no_grad():
        for batch in train_dl:
            model(batch["image"].to(device))
            # đưa embedding về CPU để không tràn VRAM
            store = (
                model.embedding_store if method == "patchcore" else model.memory_bank
            )
            store[-1] = store[-1].cpu()
    if method == "patchcore":
        embeddings = torch.vstack(model.embedding_store)
        model.embedding_store.clear()
        model.memory_bank = _coreset(embeddings, coreset_ratio, device).to(device)
        del embeddings
    else:
        # Gaussian (hiệp phương sai + nghịch đảo) tính trên CPU: torch.linalg.inv theo lô trên GPU
        # (MAGMA bản CUDA 13 trên Colab T4) lỗi "illegal memory access"; xong thì đưa mean/inv_cov lên GPU
        model.fit()
        model.to(device)
    _sync(device)
    train_time = time.time() - t0

    # Suy luận
    model.eval()
    # Checkpoint và ngưỡng tham khảo chỉ dùng train/good, không nhìn nhãn test.
    from src.demo.artifacts import infer_tensor, save_artifact

    config = {
        "image_size": size,
        "mean": list(mean),
        "std": list(std),
        "backbone": backbone,
        "layers": layers,
        "n_features": n_features,
        "num_neighbors": num_neighbors,
        "coreset_ratio": coreset_ratio,
        "seed": cfg["seed"],
        "threshold": None,
        "calibration": "train_normal_quantile_0.995_in_sample",
        "calibration_warning": "Ngưỡng tham khảo từ train/good, không bảo đảm FPR ngoài mẫu.",
    }
    reference_maps, reference_scores = [], []
    indices = np.linspace(
        0, len(train_ds) - 1, min(calibration_samples, len(train_ds)), dtype=int
    )
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x = torch.stack(
                [train_ds[int(i)]["image"] for i in indices[start : start + batch_size]]
            ).to(device)
            rm, rs = infer_tensor(model, x, method, config)
            reference_maps.append(rm)
            reference_scores.append(rs)
    config["threshold"] = float(np.quantile(np.concatenate(reference_scores), 0.995))
    config["map_scale"] = max(
        float(np.quantile(np.concatenate(reference_maps), 0.995)), 1e-8
    )
    config["calibration_samples"] = len(indices)
    ckpt_root = checkpoint_dir or ROOT / "results" / "checkpoints"
    save_artifact(
        model, method, category, config, Path(ckpt_root) / method / f"{category}.pt"
    )
    maps, scores, labels, masks = [], [], [], []
    infer_time = 0.0
    with torch.no_grad():
        for batch in test_dl:
            x = batch["image"].to(device)
            _sync(device)
            t1 = time.time()
            out = model(x)
            _sync(device)
            infer_time += time.time() - t1
            maps.append(out.anomaly_map.squeeze(1).cpu().numpy())
            scores.append(out.pred_score.reshape(-1).cpu().numpy())
            labels.append(batch["label"].numpy())
            masks.append(batch["mask"].squeeze(1).numpy())
    infer_ms = infer_time / len(test_ds) * 1000

    maps = np.concatenate(maps)
    result = evaluate(
        np.concatenate(labels), np.concatenate(scores), np.concatenate(masks), maps
    )

    if method == "patchcore":
        memory_mb = model.memory_bank.numel() * model.memory_bank.element_size() / 2**20
    else:
        g = model.gaussian
        memory_mb = (
            g.mean.numel() * g.mean.element_size()
            + g.inv_covariance.numel() * g.inv_covariance.element_size()
        ) / 2**20

    if figures:
        save_examples(
            test_ds,
            maps,
            ROOT / "results" / "figures" / method / category,
            mean=mean,
            std=std,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "result": result,
        "train_time": train_time,
        "infer_ms": infer_ms,
        "memory_mb": memory_mb,
    }
