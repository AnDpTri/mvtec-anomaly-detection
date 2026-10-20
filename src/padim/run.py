"""[Dưỡng] Chạy PaDiM trên các loại MVTec AD và ghi kết quả.

Cách dùng (dự kiến, chạy từ thư mục gốc dự án):
    python -m src.padim.run --categories bottle carpet screw   # chạy thử
    python -m src.padim.run                                   # đủ 15 loại

Triển khai tự do: dùng Anomalib (src/common/anomalib_runner.py), bản tự cài (src/padim/model.py),
hoặc cách khác. Backbone, tầng, số chiều… tùy chọn; giá trị mặc định dưới đây chỉ là điểm xuất phát
theo bài gốc.

Giữ cố định (docs/giao-dien-chung.md mục 0): huấn luyện chỉ trên ảnh good, anomaly map 256×256,
AUROC tính qua src/evaluation, ghi results/results.csv (method = padim) kèm cấu hình đã dùng
(cột implementation, backbone, note), thời gian suy luận và bộ nhớ.

Tài liệu: docs/phuong-phap/padim.md.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--impl", default="anomalib", help="cách triển khai, ví dụ: anomalib, tu-cai")
    parser.add_argument("--backbone", default="resnet18", help="điểm xuất phát theo bài gốc, đổi tùy ý")
    args = parser.parse_args()
    # TODO: chọn triển khai theo args.impl, chạy từng loại, đưa anomaly map vào src.evaluation,
    #       ghi CSV kèm cấu hình đã dùng.
    raise NotImplementedError


if __name__ == "__main__":
    main()
