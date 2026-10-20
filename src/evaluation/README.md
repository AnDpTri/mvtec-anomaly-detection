# evaluation · Phụ trách: Dưỡng

**Mục tiêu:** một thước đo duy nhất cho cả 3 phương pháp: AUROC, hình heatmap, file CSV kết quả.
**Lý do:** mọi con số trong báo cáo phải tính cùng một cách thì so sánh mới công bằng.
**Ý nghĩa:** autoencoder và Anomalib đều đưa anomaly map vào đây, kể cả khi Anomalib tự báo số
(để đối chiếu).

| File | Nội dung |
|---|---|
| `metrics.py` | `image_auroc`, `pixel_auroc`, `gaussian_smooth`, `image_score` (max của map đã làm mượt) |
| `visualize.py` | `save_figure`: ảnh gốc \| mask \| heatmap, cùng thang màu |
| `results.py` | `append_result` ghi `results/results.csv`; `done_categories` để chạy tiếp khi bị ngắt |

## Đầu vào
`anomaly_maps` (N×256×256), `image_scores` (N), `labels` (N), `masks` (N×256×256), cùng thứ tự ảnh.
Cột CSV: [docs/giao-dien-chung.md](../../docs/giao-dien-chung.md) mục 3–4. Giải thích độ đo: [docs/danh-gia.md](../../docs/danh-gia.md).

Thẻ Trello: "[Dưỡng] Viết evaluate.py (image/pixel-AUROC, heatmap)".
