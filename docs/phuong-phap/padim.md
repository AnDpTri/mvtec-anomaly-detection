# PaDiM · Phụ trách: Dưỡng

Cài đặt: `PaDiMDetector.fit_features`, `predict_features` trong `src/tn/methods/padim.py`; trích đặc trưng ở `src/tn/methods/backbone.py`.
Bài gốc: Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020.

## Mục tiêu
Lập một **"bảng tiêu chuẩn" cho từng vị trí trên ảnh** từ ảnh tốt, rồi đo mỗi mảnh của ảnh mới lệch khỏi
tiêu chuẩn của đúng vị trí đó bao xa.

## Lý do
Khắc phục điểm yếu của autoencoder: so sánh ở mức **đặc trưng** (CNN đã học trên ImageNet) thay vì pixel,
nên bền hơn với ánh sáng, nhiễu; và **không cần huấn luyện mạng**, chỉ tính thống kê, nên nhanh.

## Cách hoạt động
```
Huấn luyện (chỉ ảnh tốt, không cập nhật trọng số mạng):
  ảnh → CNN pretrained (bài gốc: ResNet18 hoặc WideResNet50) → đặc trưng nhiều tầng (bài gốc: layer1–3)
      → ghép lại → giảm chiều (bài gốc: chọn ngẫu nhiên d chiều)
  tại MỖI vị trí (i, j): trung bình μᵢⱼ và hiệp phương sai Σᵢⱼ (+ εI để khả nghịch)
Kiểm tra:
  điểm tại (i, j) = khoảng cách Mahalanobis √((x − μ)ᵀ Σ⁻¹ (x − μ))
  → phóng về 256×256 → làm mượt Gaussian → anomaly map; điểm ảnh = max
```
- **Mahalanobis thay Euclid:** tính đến việc mỗi chiều đặc trưng dao động khác nhau và có tương quan với
  nhau; một thay đổi nhỏ ở chiều vốn rất ổn định vẫn bị phát hiện.

## Ý nghĩa
Cho thấy **"con mắt" học sẵn (CNN ImageNet) + thống kê đơn giản** đã vượt xa một mạng tự huấn luyện từ đầu.

## Điểm yếu đã biết (đưa vào phân tích)
- Bảng tiêu chuẩn **gắn với vị trí**: vật xoay ngẫu nhiên (screw, metal_nut, hazelnut) làm cùng một vị trí
  lúc là ren, lúc là nền → phân phối bị nhòe. Bottle luôn thẳng hàng nên tốt.
- Bộ nhớ lớn: một ma trận hiệp phương sai d×d cho mỗi vị trí.

## Cấu hình đang dùng
WideResNet-50-2 pretrained ImageNet, layer1–3 ở lưới H/4×W/4, chọn ngẫu nhiên 100 chiều (seed 1234),
Gaussian mỗi vị trí cộng 0,01·I, Mahalanobis, làm mượt Gauss σ = 4. Hằng số ở `src/tn/config.py`.

