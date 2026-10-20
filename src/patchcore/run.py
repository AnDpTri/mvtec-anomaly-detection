"""[Chung] Chạy PatchCore trên các loại MVTec AD và ghi kết quả.

Cách dùng (dự kiến, chạy từ thư mục gốc dự án):
    # An: 8 loại
    python -m src.patchcore.run --categories bottle cable capsule carpet grid hazelnut leather metal_nut
    # Dưỡng: 7 loại
    python -m src.patchcore.run --categories pill screw tile toothbrush transistor wood zipper

Triển khai tự do: Anomalib (src/common/anomalib_runner.py), tự cài, hoặc cách khác. Backbone, tầng,
tỷ lệ coreset… tùy chọn; giá trị mặc định dưới đây chỉ là điểm xuất phát theo bài gốc.

Giữ cố định (docs/giao-dien-chung.md mục 0): huấn luyện chỉ trên ảnh good, anomaly map 256×256,
AUROC tính qua src/evaluation, ghi results/results.csv (method = patchcore) kèm cấu hình đã dùng.
Vì hai người chia loại, **hai người phải dùng chung một cấu hình với nhau** (cấu hình nào thì nhóm chọn).
Nếu tràn RAM: giảm --coreset-ratio, ghi rõ trong cột note.

Tài liệu: docs/phuong-phap/patchcore.md.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--impl", default="anomalib", help="cách triển khai, ví dụ: anomalib, tu-cai")
    parser.add_argument("--backbone", default="wide_resnet50_2", help="điểm xuất phát theo bài gốc, đổi tùy ý")
    parser.add_argument("--coreset-ratio", type=float, default=0.1, help="điểm xuất phát theo bài gốc")
    args = parser.parse_args()
    # TODO: chọn triển khai theo args.impl, chạy từng loại, đưa anomaly map vào src.evaluation,
    #       ghi CSV kèm cấu hình đã dùng.
    raise NotImplementedError


if __name__ == "__main__":
    main()
