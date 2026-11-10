"""[Dưỡng] Thí nghiệm chuyên sâu TN1–TN5 (AE, PaDiM, PatchCore tự cài, backbone WRN-50-2).

Dữ liệu -> phương pháp -> đánh giá -> thí nghiệm -> phân tích, mỗi khâu một module hoặc gói:

    config       siêu tham số chung, thiết bị, kiểm tra GPU
    data         đọc MVTec AD / MVTec AD 2 / VisA
    download     tải dữ liệu trên Colab
    splits       chia tập TN1 (trộn ảnh lỗi), TN2 (few-shot), tách fit/val
    methods/     ba phương pháp chung giao diện Detector: ae, padim, patchcore (+ backbone dùng chung)
    metrics      AUROC, AP, F1-max, AUPRO, ngưỡng từ ảnh good validation
    experiments/ vòng chạy TN1, TN2, TN2b, TN3, TN4 và ghi CSV từng run
    analysis/    bảng tổng hợp và biểu đồ cho phần phân tích

Notebook trong notebooks/tn/ chỉ còn cấu hình, vòng chạy và phân tích; demo nằm ở src/demo.
"""
