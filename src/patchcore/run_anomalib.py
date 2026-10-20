"""[Chung] Chạy PatchCore bằng Anomalib trên các loại MVTec AD.

Cách dùng (dự kiến, chạy từ thư mục gốc dự án):
    # An: 8 loại
    python -m src.patchcore.run_anomalib --categories bottle cable capsule carpet grid hazelnut leather metal_nut
    # Dưỡng: 7 loại
    python -m src.patchcore.run_anomalib --categories pill screw tile toothbrush transistor wood zipper

Yêu cầu:
- Hai người dùng cùng cấu hình (backbone, tầng, tỷ lệ coreset, seed) và cùng phiên bản Anomalib.
- Kết quả ghi vào results/results.csv (method = patchcore); ghi thời gian suy luận và bộ nhớ.
- Nếu tràn RAM: giảm --coreset-ratio, ghi rõ trong cột note.

Quy trình chạy dùng chung với PaDiM: src/common/anomalib_runner.py.
Tài liệu: docs/phuong-phap/patchcore.md.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    args = parser.parse_args()
    # TODO: model = anomalib.models.Patchcore(backbone=args.backbone,
    #                                         coreset_sampling_ratio=args.coreset_ratio, ...)
    #       src.common.anomalib_runner.run(model, "patchcore", categories, note=...)
    raise NotImplementedError


if __name__ == "__main__":
    main()
