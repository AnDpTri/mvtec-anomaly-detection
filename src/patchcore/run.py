"""[Chung] Chạy PatchCore (Anomalib) trên các loại MVTec AD và ghi kết quả.

Cách dùng (chạy từ thư mục gốc dự án):
    # An: 8 loại
    python -m src.patchcore.run --categories bottle cable capsule carpet grid hazelnut leather metal_nut
    # Dưỡng: 7 loại
    python -m src.patchcore.run --categories pill screw tile toothbrush transistor wood zipper

Triển khai `anomalib` dùng mô hình PyTorch của Anomalib qua src/common/anomalib_runner.py: ảnh 256×256
chuẩn hóa ImageNet, huấn luyện chỉ trên train/good, anomaly map 256×256 đưa vào src/evaluation.
Mỗi loại ghi một dòng vào CSV (method = patchcore) kèm cấu hình đầy đủ trong cột note; chạy lại thì bỏ
qua loại đã có cùng cấu hình. Bản tự cài (--impl tu-cai) chưa có.

Hai người chia loại nên **phải dùng chung một cấu hình với nhau**. Embedding gom về CPU, coreset chạy
trên GPU (vừa VRAM 4 GB). Nếu tràn RAM: giảm --coreset-ratio, ghi rõ trong cột note.

Tài liệu: docs/phuong-phap/patchcore.md.
"""
import argparse

import torch

from src.common.anomalib_runner import DEFAULT_LAYERS, run_category
from src.common.config import all_categories, category_type, load_config
from src.common.seed import set_seed
from src.evaluation.results import append_result, done_categories


def main():
    parser = argparse.ArgumentParser(description="PatchCore cho MVTec AD")
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--impl", default="anomalib", help="cách triển khai: anomalib (tu-cai chưa có)")
    parser.add_argument("--backbone", default="wide_resnet50_2", help="backbone timm")
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS["patchcore"], help="các tầng lấy đặc trưng")
    parser.add_argument("--coreset-ratio", type=float, default=0.1, help="tỷ lệ coreset giữ lại")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", help="thư mục MVTecAD (mặc định theo configs/common.yaml)")
    parser.add_argument("--csv", default=None, help="mặc định: results_csv trong cấu hình")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="chạy lại cả loại đã có kết quả")
    args = parser.parse_args()

    if args.impl != "anomalib":
        raise NotImplementedError("Bản tự cài (--impl tu-cai) chưa có; hiện chỉ hỗ trợ --impl anomalib.")

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.csv or cfg["results_csv"]
    note = (f"backbone={args.backbone}; layers={'+'.join(args.layers)}; coreset={args.coreset_ratio}; neighbors=9; "
            f"batch={args.batch_size}; size={cfg['data']['image_size']}; seed={cfg['seed']}")
    done = set() if args.rerun else done_categories(csv_path, "patchcore", note)
    print(f"Thiết bị: {device}. Cấu hình: {note}")

    for category in args.categories or all_categories(cfg):
        if category in done:
            print(f"Bỏ qua {category} (đã có kết quả)")
            continue
        print(f"== {category}", flush=True)
        set_seed(cfg["seed"])
        out = run_category("patchcore", category, cfg, device, args.backbone, layers=args.layers,
                           batch_size=args.batch_size, workers=args.workers, data_root=args.data_root,
                           coreset_ratio=args.coreset_ratio, figures=not args.no_figures)
        result = out["result"]
        print(f"  image-AUROC {result['image_auroc']:.4f}  pixel-AUROC {result['pixel_auroc']:.4f}  "
              f"train {out['train_time']:.1f}s  infer {out['infer_ms']:.1f}ms/ảnh", flush=True)
        append_result(csv_path, {
            "category": category, "type": category_type(cfg, category), "method": "patchcore",
            "implementation": "anomalib", **{k: f"{v:.4f}" for k, v in result.items()},
            "train_time_s": f"{out['train_time']:.1f}", "infer_time_ms": f"{out['infer_ms']:.2f}",
            "memory_mb": f"{out['memory_mb']:.1f}", "backbone": args.backbone, "note": note,
        })


if __name__ == "__main__":
    main()
