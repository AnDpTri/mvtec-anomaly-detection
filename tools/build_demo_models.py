"""Lưu mô hình để demo; CSV kết quả cũ được giữ nguyên. Có resume từng loại."""

from pathlib import Path
import argparse
import shutil
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import ROOT, load_config, all_categories, category_type
from src.common.seed import set_seed
from src.common.anomalib_runner import run_category, DEFAULT_LAYERS
from src.data.mvtec import MVTecDataset
from src.demo.artifacts import load_artifact, infer_tensor, save_artifact
from src.evaluation.results import append_result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root")
    p.add_argument("--categories", nargs="+")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["ae", "padim", "patchcore"],
        default=["ae", "padim", "patchcore"],
    )
    p.add_argument(
        "--ae-source",
        type=Path,
        help="Thư mục AE đã train, chỉ đọc và sao chép vào worktree",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    cfg = load_config()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    categories = args.categories or all_categories(cfg)
    if any(c not in all_categories(cfg) for c in categories):
        p.error("Loại không có trong common.yaml")
    for method in args.methods:
        for category in categories:
            dest = ROOT / "results/checkpoints" / method / f"{category}.pt"
            if dest.exists() and dest.with_suffix(".json").exists():
                print("RESUME skip", method, category, flush=True)
                continue
            set_seed(cfg["seed"])
            print("BUILD", method, category, flush=True)
            if method == "ae":
                source = (
                    args.ae_source or ROOT / "results/checkpoints/ae"
                ) / f"{category}.pt"
                if not source.is_file():
                    raise FileNotFoundError(f"AE chưa train: {source}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != dest.resolve():
                    shutil.copy2(source, dest)
                model, config, _ = load_artifact(str(dest), str(device))
                ds = MVTecDataset(
                    args.data_root or cfg["data"]["root"],
                    category,
                    "train",
                    cfg["data"]["image_size"],
                )
                indices = np.linspace(0, len(ds) - 1, min(64, len(ds)), dtype=int)
                maps = []
                scores = []
                for start in range(0, len(indices), args.batch_size):
                    images = torch.stack(
                        [
                            ds[int(i)]["image"]
                            for i in indices[start : start + args.batch_size]
                        ]
                    ).to(device)
                    m, s = infer_tensor(model, images, "ae", config)
                    maps.append(m)
                    scores.append(s)
                config.update(
                    threshold=float(np.quantile(np.concatenate(scores), 0.995)),
                    map_scale=max(
                        float(np.quantile(np.concatenate(maps), 0.995)), 1e-8
                    ),
                    calibration="train_normal_quantile_0.995_in_sample",
                    calibration_samples=len(indices),
                    calibration_warning="Ngưỡng tham khảo từ train/good, không bảo đảm FPR ngoài mẫu.",
                )
                save_artifact(model, "ae", category, config, dest)
                load_artifact.cache_clear()
                del model
            else:
                backbone = "resnet18" if method == "padim" else "wide_resnet50_2"
                result = run_category(
                    method,
                    category,
                    cfg,
                    device,
                    backbone,
                    layers=DEFAULT_LAYERS[method],
                    batch_size=args.batch_size,
                    workers=args.workers,
                    data_root=args.data_root,
                    figures=False,
                    checkpoint_dir=ROOT / "results/checkpoints",
                )
                append_result(
                    ROOT / "results/demo_models/results.csv",
                    dict(
                        category=category,
                        type=category_type(cfg, category),
                        method=method,
                        implementation="anomalib",
                        **result["result"],
                        train_time_s=result["train_time"],
                        infer_time_ms=result["infer_ms"],
                        memory_mb=result["memory_mb"],
                        backbone=backbone,
                        note=f"demo_backfill; batch={args.batch_size}; seed=42; original_results_unchanged",
                    ),
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("SAVED", dest, flush=True)


if __name__ == "__main__":
    main()
