"""[Dưỡng] Thí nghiệm chuyên sâu TN1–TN5 (AE, PaDiM, PatchCore tự cài, backbone WRN-50-2).

Trước đây toàn bộ code nằm lặp lại trong 3 notebook Colab; gói này giữ phần dùng chung, notebook trong
notebooks/tn/ chỉ còn cấu hình, vòng chạy và phân tích.

    config       siêu tham số chung, thiết bị
    data         đọc MVTec AD / MVTec AD 2 / VisA, tải dữ liệu trên Colab
    splits       chia tập TN1 (trộn ảnh lỗi), TN2 (few-shot), tách fit/val
    methods      backbone, PaDiM, PatchCore, Autoencoder (fit / predict)
    metrics      AUROC, AP, F1-max, AUPRO, ngưỡng từ ảnh good validation
    shifts       shift nhân tạo cho TN4b
    screw        đo và căn góc con vít cho TN2b
    results      ghi CSV theo từng run, bỏ qua run đã xong
    experiments  vòng chạy TN1, TN2, TN2b, TN3, TN4
    analysis     bảng tổng hợp và biểu đồ cho phần phân tích
"""
