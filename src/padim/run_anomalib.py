"""[Dưỡng] Chạy PaDiM bằng Anomalib trên các loại MVTec AD.

Cách dùng (dự kiến, chạy từ thư mục gốc dự án):
    python -m src.padim.run_anomalib --categories bottle carpet screw   # chạy thử
    python -m src.padim.run_anomalib                                   # đủ 15 loại

Yêu cầu:
- Cùng cấu hình chung cho cả 15 loại; kết quả ghi vào results/results.csv (method = padim).
- Đối chiếu với bản tự cài src/padim/model.py; nếu lệch nhiều thì tìm nguyên nhân
  (backbone, tầng đặc trưng, số chiều chọn ngẫu nhiên, seed).
- Ghi thời gian suy luận mỗi ảnh và bộ nhớ (ma trận hiệp phương sai).

Quy trình chạy dùng chung với PatchCore: src/common/anomalib_runner.py.
Tài liệu: docs/phuong-phap/padim.md.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--backbone", default="resnet18")
    args = parser.parse_args()
    # TODO: model = anomalib.models.Padim(backbone=args.backbone, ...)
    #       src.common.anomalib_runner.run(model, "padim", categories)
    raise NotImplementedError


if __name__ == "__main__":
    main()
