"""Checkpoint tự mô tả, nạp weights_only và suy luận không fit lại."""

from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from src.common.config import ROOT, load_config
from src.evaluation.metrics import gaussian_smooth


def checkpoint_path(method, category, root=None):
    cfg = load_config()
    categories = cfg["categories"]["texture"] + cfg["categories"]["object"]
    if method not in {"ae", "padim", "patchcore"} or category not in categories:
        raise ValueError("Loại sản phẩm hoặc phương pháp không có trong cấu hình.")
    return (
        Path(
            root
            or os.environ.get("MVTEC_CHECKPOINT_ROOT", ROOT / "results/checkpoints")
        )
        / method
        / f"{category}.pt"
    )


def save_artifact(model, method, category, config, path=None):
    dest = Path(path) if path else checkpoint_path(method, category)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".pt.tmp")
    state = {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "method": method,
            "category": category,
            "config": config,
            "state_dict": state,
        },
        temp,
    )
    temp.replace(dest)
    dest.with_suffix(".json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dest


def prepare_image(image, config, method, device):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = image.convert("RGB")
    size = config["image_size"]
    x = TF.to_tensor(
        TF.resize(
            image,
            [size, size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    if method != "ae":
        x = TF.normalize(x, config["mean"], config["std"])
    return x.unsqueeze(0).to(device)


@lru_cache(maxsize=2)
def load_artifact(path, device_name="cpu"):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Chưa có mô hình: {p}. Xem tools/build_demo_models.py."
        )
    data = torch.load(p, map_location="cpu", weights_only=True)
    if "format_version" not in data:
        from src.autoencoder.model import ConvAutoencoder

        config = {
            "image_size": 256,
            "latent_dim": int(data["to_latent.weight"].shape[0]),
            "loss": "mse",
            "sigma": 4,
            "threshold": None,
            "calibration": "not_calibrated",
        }
        method = "ae"
        model = ConvAutoencoder(config["latent_dim"], config["image_size"])
        model.load_state_dict(data)
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            config.update(json.loads(sidecar.read_text(encoding="utf-8")))
    else:
        method = data["method"]
        config = data["config"]
        if method == "ae":
            from src.autoencoder.model import ConvAutoencoder

            model = ConvAutoencoder(config["latent_dim"], config["image_size"])
        else:
            from src.common.anomalib_runner import _build_model

            model = _build_model(
                method,
                config["backbone"],
                config["layers"],
                config.get("n_features"),
                config.get("num_neighbors", 9),
                pre_trained=False,
            )
        model.load_state_dict(data["state_dict"], strict=True)
    return model.to(torch.device(device_name)).eval(), config, method


@torch.no_grad()
def infer_tensor(model, images, method, config):
    if method == "ae":
        from src.autoencoder.model import anomaly_map

        maps = (
            anomaly_map(model, images, config.get("loss", "mse"))
            .squeeze(1)
            .cpu()
            .numpy()
        )
        maps = gaussian_smooth(maps, sigma=config.get("sigma", 4))
        return maps, maps.reshape(len(maps), -1).max(axis=1)
    out = model(images)
    return (
        out.anomaly_map.squeeze(1).cpu().numpy(),
        out.pred_score.reshape(-1).cpu().numpy(),
    )


def predict_file(image, category, method, root=None, device="cpu"):
    path = checkpoint_path(method, category, root)
    model, config, stored_method = load_artifact(str(path), device)
    if stored_method != method:
        raise ValueError("Phương pháp và checkpoint không khớp.")
    maps, scores = infer_tensor(
        model, prepare_image(image, config, method, device), method, config
    )
    return maps[0], float(scores[0]), config
