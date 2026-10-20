# Autoencoder (baseline) · Phụ trách: An

File: `src/models/autoencoder.py`, `scripts/train_ae.py` (PyTorch, bảng chính) ·
`numpy_ae/autoencoder_numpy.py` (thuần NumPy, phụ lục)

## Mục tiêu
Tạo ra **phiên bản bình thường của chính ảnh đang kiểm tra**, rồi so với ảnh gốc: chỗ khác nhau là lỗi.

## Lý do
Cần một ảnh chuẩn để so, và ảnh chuẩn đó phải cùng vị trí, cùng góc, cùng ánh sáng với ảnh đang xét,
chỉ khác là không có lỗi. Ảnh tốt có sẵn trong tập train không đáp ứng được (vật xoay, lệch); mạng tự
vẽ lại thì đáp ứng được.

## Cách hoạt động
```
Ảnh x ──[Encoder: nén]──> cổ chai nhỏ ──[Decoder: bung]──> ảnh vẽ lại x̂
Huấn luyện: chỉ ảnh tốt, giảm loss ‖x − x̂‖²
Kiểm tra:   anomaly map = (x − x̂)² theo pixel → làm mượt → điểm ảnh = max
```
- **Cổ chai** là chìa khóa: không có thì mạng chép nguyên ảnh, kể cả vết xước. Có cổ chai thì mạng buộc
  phải học "ảnh tốt trông thế nào"; thứ không có trong "từ vựng" ảnh tốt (vết xước) bị rơi mất khi nén.
- **Bản PyTorch:** conv autoencoder (Conv stride 2 / ConvTranspose), ảnh 3×256×256, đúng cấu hình chung.
- **Bản NumPy:** fully connected 16.384 → 512 → 64 → 512 → 16.384 trên ảnh xám 128×128; tự viết
  xuôi, ngược, Adam, AUROC. Không so sánh công bằng với bảng chính, chỉ để hiểu tận gốc.

## Ý nghĩa
Mốc so sánh cho hai phương pháp chính, và là phương pháp **duy nhất trong ba phương pháp thật sự huấn luyện
mạng neuron**.

## Điểm yếu đã biết (đưa vào phân tích)
- Mạng tổng quát quá tốt → vẽ lại được cả vùng lỗi → bỏ sót.
- Vẽ lại mờ ở texture nhiều chi tiết (tile, zipper) → báo động giả khắp ảnh (bài gốc MVTec AD).
- So sánh ở mức pixel → nhạy với ánh sáng, nhiễu.
- Vật xoay ngẫu nhiên (screw, metal_nut, hazelnut) → cổ chai phải tốn chỗ để nhớ góc xoay.

## Quyết định cần ghi vào báo cáo
- Không xoay/lật ảnh khi tăng dữ liệu (tránh biến lỗi thành "bình thường"); bài gốc thì có xoay/lật.
- Cùng siêu tham số cho cả 15 loại, không chỉnh theo tập test.
