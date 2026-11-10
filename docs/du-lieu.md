# Dữ liệu: MVTec AD, MVTec AD 2, VisA

Cách tải: [../datasets/README.md](../datasets/README.md). Đề tài dùng 9 loại từ ba bộ:

| Bộ | Loại | Nguồn |
|---|---|---|
| MVTec AD | grid, screw, transistor | Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. CC BY-NC-SA 4.0 |
| MVTec AD 2 | vial, sheet_metal | Heckler-Kram et al., *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection* |
| VisA | pcb1, pcb2, pcb3, pcb4 | Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (VisA), ECCV 2022 |

Số ảnh fit, validation, test của từng loại: [ket-qua.md](ket-qua.md) mục 2.

## Vì sao dùng các bộ này
- Trong công nghiệp, lỗi **nhỏ, tinh tế**, nằm trên sản phẩm gần như giống hệt hàng tốt; ảnh lỗi hiếm và
  không biết trước kiểu lỗi. MVTec AD mô phỏng đúng tình huống đó, kèm **mask từng pixel** để đánh giá cả định vị.
- MVTec AD 2 có ánh sáng thay đổi giữa train và test (`cond` trong tên file) nên dùng cho thí nghiệm đổi điều kiện chụp.
- VisA có nhiều ảnh train hơn (khoảng 770 ảnh mỗi loại pcb), dùng để so với các loại ít ảnh.

## MVTec AD
- Train chỉ gồm ảnh tốt; test gồm ảnh tốt và ảnh lỗi kèm mask. Ảnh cắt về 700×700 đến 1024×1024.
- **grid, screw là ảnh xám 1 kênh** → `read_image` nhân thành 3 kênh cho backbone ImageNet.
- Thống kê ba loại đang dùng:

| Loại | Train | Test tốt | Test lỗi | Kiểu lỗi | Ghi chú |
|---|---|---|---|---|---|
| grid | 264 | 21 | 57 | 5 | Xám; texture đều |
| screw | 320 | 41 | 119 | 5 | Xám, **xoay ngẫu nhiên**, lỗi rất nhỏ |
| transistor | 213 | 60 | 40 | 4 | Có lỗi thiếu/lệch linh kiện |

MVTec AD không có tập validation: `split_fit_val` tách 15% `train/good` (seed 0) để đặt ngưỡng.

## MVTec AD 2
Dùng `train/good`, `validation/good` và `test_public`; tên file test có dạng `<cảnh>_<điều kiện>.png`
(`regular`, `overexposed`, `underexposed`, `shift_1`…). Cỡ ảnh đưa về 256×192 (vial) và 256×1024 (sheet_metal).

## VisA
Theo split `split_csv/1cls.csv` chính thức. Mask lưu giá trị 0/1 (đọc với `thr=0`). Cỡ ảnh 256×320.

## Lưu ý khi xử lý
- Ảnh good không có file mask → mask toàn 0.
- Mask: resize bằng **nearest** rồi nhị phân hóa lại; ảnh resize bằng `INTER_AREA`.
- Mask MVTec ở `ground_truth/<kiểu lỗi>/<tên ảnh>_mask.png`.
- Mỗi loại một mô hình riêng; kết quả tính riêng từng loại.

## Điều bài gốc MVTec AD phát hiện (dùng cho phần phân tích)
- Không phương pháp nào thắng ở mọi loại.
- Phương pháp dùng đặc trưng CNN nhưng **bỏ qua vị trí** làm tốt trên texture, kém trên object.
- Autoencoder vẽ lại **mờ** ở texture nhiều chi tiết → báo động giả.
- ROC AUC cao **không** có nghĩa là định vị tốt ở một ngưỡng cụ thể: chọn ngưỡng chỉ từ ảnh tốt rất khó.
