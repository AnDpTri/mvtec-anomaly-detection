# Đánh giá: các chỉ số trong `src/tn/metrics.py`

**Mục tiêu:** một cách tính thống nhất cho cả 3 phương pháp, cài đặt duy nhất ở `src/tn/metrics.py`.
**Lý do chọn AUROC:** mô hình chỉ cho ra điểm liên tục; AUROC đánh giá trên **mọi ngưỡng**, nên so sánh
được các phương pháp mà không phụ thuộc vào cách chọn ngưỡng (bài gốc MVTec AD cho thấy chọn ngưỡng chỉ
từ ảnh tốt rất khó). Demo vẫn cần một ngưỡng cụ thể nên có thêm các chỉ số tại ngưỡng.

## Image-AUROC (phát hiện)
- Đầu vào: nhãn 0/1 và điểm ảnh của mọi ảnh test một loại.
- Ý nghĩa: xác suất một ảnh lỗi chọn ngẫu nhiên có điểm **cao hơn** một ảnh tốt chọn ngẫu nhiên.
- 1.0 = tách hoàn hảo; 0.5 = không hơn đoán mò; dưới 0.5 = ngược (thường là bug).
- Kèm `i_ap` (average precision) và `i_f1max` (F1 lớn nhất trên mọi ngưỡng).

## Pixel-AUROC (định vị)
- Đầu vào: mask và anomaly map của mọi ảnh test một loại, trải phẳng thành từng pixel.
- **Cạm bẫy:** pixel tốt chiếm đa số (vùng lỗi thường < 5% ảnh), nên pixel-AUROC dễ cao dù heatmap
  còn nhiều báo động giả. Vì vậy có thêm `p_ap` và AUPRO.

## AUPRO
Per-region overlap: mỗi vùng lỗi liên thông (8-connectivity) có trọng số như nhau, bất kể to hay nhỏ.
Tính diện tích dưới đường PRO tới FPR = 0,3 (`aupro`) và 0,05 (`aupro_005`), chuẩn hóa về [0, 1].

## Điểm ảnh từ anomaly map
`image_scores(maps)`: lấy **max** của anomaly map đã làm mượt Gauss (σ = 4).

## Chỉ số tại ngưỡng
`thresholds_from_val(val_maps)`: ngưỡng ảnh = trung bình + 3 độ lệch chuẩn của điểm trên ảnh good validation;
ngưỡng pixel tính cùng cách trên từng điểm ảnh. `metrics_full` trả thêm tỷ lệ báo nhầm ảnh tốt (`img_fpr_thr`),
tỷ lệ bắt được ảnh lỗi (`img_tpr_thr`), F1 mức ảnh và mức pixel tại ngưỡng đó.

## Quy tắc
- Tính **riêng từng loại**, không gộp ảnh của nhiều loại vào một lần tính.
- Ngưỡng chỉ lấy từ ảnh good validation, không dùng ảnh fit và không dùng nhãn test.
- Kiểm tra trên trường hợp biết trước kết quả: map trùng mask → AUROC = 1.0; map ngẫu nhiên → khoảng 0.5.
  Các kiểm tra này nằm trong `tests/test_metrics.py`.
