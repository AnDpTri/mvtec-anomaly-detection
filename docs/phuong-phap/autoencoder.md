# Autoencoder (baseline) · Phụ trách: An

Module: `src/autoencoder/` · `model.py`, `train.py` (bản chính) · `numpy_ae/` (bản tự viết bằng NumPy, phụ lục)

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
- **Gợi ý điểm xuất phát (bài gốc MVTec AD):** conv autoencoder ảnh 256×256, loss L2 hoặc SSIM. Kiến trúc,
  cỡ cổ chai, loss đều đổi được (ví dụ thêm skip, dùng SSIM, VAE…).
- **Bản NumPy (tùy chọn, để học):** tự viết xuôi, ngược, tối ưu, AUROC bằng NumPy; kiến trúc và kích thước
  tự chọn. Không so sánh công bằng với bảng chính nếu khác giao thức.

## Ý nghĩa
Mốc so sánh cho hai phương pháp chính, và là phương pháp **duy nhất trong ba phương pháp thật sự huấn luyện
mạng neuron**.

## Điểm yếu đã biết (đưa vào phân tích)
- Mạng tổng quát quá tốt → vẽ lại được cả vùng lỗi → bỏ sót.
- Vẽ lại mờ ở texture nhiều chi tiết (tile, zipper) → báo động giả khắp ảnh (bài gốc MVTec AD).
- So sánh ở mức pixel → nhạy với ánh sáng, nhiễu.
- Vật xoay ngẫu nhiên (screw, metal_nut, hazelnut) → cổ chai phải tốn chỗ để nhớ góc xoay.

## Lựa chọn cần ghi vào báo cáo
- Có xoay/lật ảnh khi tăng dữ liệu hay không, và vì sao (bài gốc có; xoay/lật có thể biến lỗi thành "bình thường").
- Kiến trúc, loss, siêu tham số đã dùng. Không chỉnh siêu tham số dựa trên tập test.

## Triển khai: tự do
Thư viện, kiến trúc, backbone, tầng, loss, siêu tham số, tăng cường dữ liệu đều **tự chọn**; chỉ cần giữ
giao thức đánh giá chung và ghi cấu hình đã dùng vào CSV (xem [giao-dien-chung.md](../giao-dien-chung.md) mục 0).
Các con số trong phần "Cách hoạt động" là **điểm xuất phát theo bài gốc**, không bắt buộc.

