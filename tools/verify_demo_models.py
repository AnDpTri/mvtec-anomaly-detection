"""Kiểm tra mọi checkpoint bằng suy luận thực, không tải mô hình qua mạng."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.common.config import ROOT, all_categories, load_config
from src.demo.artifacts import checkpoint_path, load_artifact, predict_file


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "docs/demo_validation.json")
    args = p.parse_args()
    torch.set_num_threads(4)
    records = []
    with patch(
        "socket.socket.connect", side_effect=RuntimeError("Offline verification")
    ):
        for method in ["ae", "padim", "patchcore"]:
            for category in all_categories(load_config()):
                path = checkpoint_path(method, category)
                payload = torch.load(path, weights_only=True, mmap=True)
                assert payload["category"] == category and payload["method"] == method
                assert payload["format_version"] == 1
                del payload
                image_path = next(
                    iter(
                        sorted((args.data_root / category / "test/good").glob("*.png"))
                    )
                )
                with Image.open(image_path) as source:
                    anomaly, score, config = predict_file(source, category, method)
                assert anomaly.shape == (256, 256) and np.isfinite(anomaly).all()
                assert np.isfinite(score) and np.isfinite(config["threshold"])
                assert config["calibration_samples"] > 0
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                records.append(
                    {
                        "method": method,
                        "category": category,
                        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "smoke_score_cpu": score,
                        "map_shape": list(anomaly.shape),
                        "threshold": config["threshold"],
                        "calibration": config["calibration"],
                    }
                )
                load_artifact.cache_clear()
                gc.collect()
                print("VERIFIED", method, category, flush=True)
    report = {
        "count": len(records),
        "network_blocked": True,
        "device": "cpu",
        "validation": "reload + finite score/map on one good test image per model; not an accuracy estimate",
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchvision",
                "anomalib",
                "numpy",
                "scipy",
                "scikit-learn",
                "gradio",
                "timm",
            ]
        },
        "checkpoints": records,
    }
    assert len(records) == 45
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print("All 45 checkpoints verified", flush=True)


if __name__ == "__main__":
    main()
