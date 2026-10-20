# Đánh giá: image-AUROC và pixel-AUROC

**Mục tiêu:** một thước đo thống nhất cho cả 3 phương pháp, cài đặt duy nhất ở module `src/evaluation/` (Dưỡng).
**Lý do chọn AUROC:** mô hình chỉ cho ra điểm liên tục; AUROC đánh giá trên **mọi ngưỡng**, nên so sánh
được các phương pháp mà không phụ thuộc vào cách chọn ngưỡng (bài gốc MVTec AD cho thấy chọn ngưỡng chỉ
từ ảnh tốt rất khó).

## Image-AUROC (phát hiện)
- Đầu vào: `labels` (0/1) và `image_scores` của mọi ảnh test một loại.
- Ý nghĩa: xác suất một ảnh lỗi chọn ngẫu nhiên có điểm **cao hơn** một ảnh tốt chọn ngẫu nhiên.
- 1.0 = tách hoàn hảo; 0.5 = không hơn đoán mò; dưới 0.5 = ngược (thường là bug).

## Pixel-AUROC (định vị)
- Đầu vào: `masks` và `anomaly_maps` của mọi ảnh test một loại, trải phẳng thành từng pixel.
- Ý nghĩa: như trên nhưng ở mức pixel: pixel lỗi có điểm cao hơn pixel tốt không.
- **Cạm bẫy:** pixel tốt chiếm đa số (vùng lỗi thường < 5% ảnh), nên pixel-AUROC dễ cao dù heatmap
  còn nhiều báo động giả. Nếu kịp, tính thêm **PRO** (per-region overlap) để mỗi vùng lỗi được tính
  ngang nhau, bất kể to hay nhỏ.

## Điểm ảnh từ anomaly map
Mặc định: làm mượt anomaly map bằng Gaussian (sigma = 4) rồi lấy **max**. Ghi rõ nếu đổi cách khác.

## Quy tắc
- Tính **riêng từng loại**, rồi mới lấy trung bình (trung bình texture, object và chung 15 loại).
  Không gộp ảnh của 15 loại vào một lần tính.
- Cả 3 phương pháp dùng chung `src/evaluation/`; với Anomalib, lấy anomaly map ra rồi tính lại bằng module
  này, đối chiếu với số Anomalib tự báo.
- Trước khi dùng thật: kiểm tra trên trường hợp biết trước kết quả (ví dụ điểm ảnh = nhãn → AUROC = 1.0;
  điểm ngẫu nhiên → khoảng 0.5).

## Cài đặt
`sklearn.metrics.roc_auc_score`. Riêng bản NumPy (`src/autoencoder/numpy_ae/`) tự viết AUROC để hiểu cách tính, rồi đối
chiếu với sklearn.
