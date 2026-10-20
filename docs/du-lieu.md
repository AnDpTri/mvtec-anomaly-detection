# Dữ liệu: MVTec AD

Nguồn: Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*,
CVPR 2019 (bảng 1 và mục 3 của bài). Giấy phép CC BY-NC-SA 4.0. Cách tải: [../datasets/README.md](../datasets/README.md).

## Vì sao dùng bộ này
- Trước MVTec AD, "phát hiện bất thường" thường được thử bằng cách coi vài lớp của MNIST/CIFAR là "lạ";
  ảnh lạ khác hẳn ảnh thường, không giống bài toán thật.
- Trong công nghiệp, lỗi **nhỏ, tinh tế**, nằm trên sản phẩm gần như giống hệt hàng tốt; ảnh lỗi hiếm và
  không biết trước kiểu lỗi. MVTec AD mô phỏng đúng tình huống đó, kèm **mask từng pixel** để đánh giá cả định vị.

## Tổng quan
- **5.354 ảnh, 15 loại** (5 texture, 10 object).
- **Train: 3.629 ảnh, toàn ảnh tốt.** Test: 1.725 ảnh = 467 tốt + 1.258 lỗi.
- **73 kiểu lỗi** (trung bình ~5 kiểu mỗi loại), **1.888 vùng lỗi** vẽ tay.
- Chụp bằng cảm biến công nghiệp 2048×2048, cắt về **700×700 đến 1024×1024**.
- **grid, screw, zipper là ảnh xám 1 kênh** → phải nhân thành 3 kênh cho backbone ImageNet.
- Ánh sáng được kiểm soát; một số loại cố ý đổi ánh sáng để tăng độ khó. Lỗi được tạo thủ công cho giống lỗi thật.

## Thống kê 15 loại
| Loại | Nhóm | Train | Test tốt | Test lỗi | Kiểu lỗi | Vùng lỗi | Cạnh ảnh | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| carpet | texture | 280 | 28 | 89 | 5 | 97 | 1024 | Texture đều |
| grid | texture | 264 | 21 | 57 | 5 | 170 | 1024 | Xám; texture đều |
| leather | texture | 245 | 32 | 92 | 5 | 99 | 1024 | Texture ngẫu nhiên |
| tile | texture | 230 | 33 | 84 | 5 | 86 | 840 | Texture ngẫu nhiên |
| wood | texture | 247 | 19 | 60 | 5 | 168 | 1024 | Texture ngẫu nhiên |
| bottle | object | 209 | 20 | 63 | 3 | 68 | 900 | Cứng, thẳng hàng |
| cable | object | 224 | 58 | 92 | 8 | 151 | 1024 | Mềm, biến dạng; có lỗi kiểu logic |
| capsule | object | 219 | 23 | 109 | 5 | 114 | 1000 | Gần thẳng hàng |
| hazelnut | object | 391 | 40 | 70 | 4 | 136 | 1024 | Biến thiên tự nhiên, **xoay ngẫu nhiên** |
| metal_nut | object | 220 | 22 | 93 | 4 | 132 | 700 | Cứng, **xoay ngẫu nhiên** |
| pill | object | 267 | 26 | 141 | 7 | 245 | 800 | Gần thẳng hàng |
| screw | object | 320 | 41 | 119 | 5 | 135 | 1024 | Xám, **xoay ngẫu nhiên**, lỗi rất nhỏ |
| toothbrush | object | 60 | 12 | 30 | 1 | 66 | 1024 | Tập train nhỏ nhất |
| transistor | object | 213 | 60 | 40 | 4 | 44 | 1024 | Có lỗi thiếu/lệch linh kiện |
| zipper | object | 240 | 32 | 119 | 7 | 177 | 1024 | Xám |
| **Tổng** | | **3.629** | **467** | **1.258** | **73** | **1.888** | | |

## Lưu ý khi xử lý (áp dụng cho cả 3 phương pháp)
- Ảnh xám (grid, screw, zipper) → nhân thành 3 kênh.
- Ảnh good không có file mask → tạo mask toàn 0.
- Mask: resize bằng **nearest** rồi nhị phân hóa lại (giá trị > 0.5 → 1); crop ảnh thì crop mask y hệt.
- Mask nằm ở `ground_truth/<kiểu lỗi>/<tên ảnh>_mask.png`.
- Mỗi loại một mô hình riêng; kết quả tính riêng từng loại rồi mới lấy trung bình.

## Điều bài gốc phát hiện (dùng cho phần phân tích)
- Không phương pháp nào (AnoGAN, autoencoder L2/SSIM, CNN feature dictionary, GMM, variation model) thắng ở mọi loại.
- Phương pháp dùng đặc trưng CNN nhưng **bỏ qua vị trí** làm tốt trên texture, kém trên object.
- Autoencoder vẽ lại **mờ** ở texture nhiều chi tiết (tile, zipper) → báo động giả.
- ROC AUC cao **không** có nghĩa là định vị tốt ở một ngưỡng cụ thể: chọn ngưỡng chỉ từ ảnh tốt rất khó.
- Bài gốc **có** dùng xoay/dịch/lật ảnh để tăng dữ liệu cho mô hình học sâu (10.000 mẫu mỗi loại).
  Nhóm đang chọn **không** xoay/lật cho AE; cần ghi rõ lý do trong báo cáo.
