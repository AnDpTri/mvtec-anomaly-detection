# padim · Phụ trách: Dưỡng

**Mục tiêu:** lập "bảng tiêu chuẩn" (trung bình + hiệp phương sai) cho từng vị trí từ ảnh tốt; mảnh nào
lệch xa tiêu chuẩn của đúng vị trí đó (khoảng cách Mahalanobis) là lỗi.
**Lý do:** so ở mức đặc trưng CNN thay vì pixel, và không cần huấn luyện mạng.
**Ý nghĩa:** cho thấy đặc trưng học sẵn + thống kê đơn giản vượt xa autoencoder; lộ điểm yếu khi vật xoay.

| File | Nội dung |
|---|---|
| `model.py` | PaDiM tự cài (tùy chọn: để hiểu, hoặc làm bản chính) |
| `run.py` | Chạy đủ 15 loại, ghi CSV (method = `padim`); triển khai bằng Anomalib hoặc bản tự cài (`--impl`) |

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.padim.run --categories bottle carpet screw   # chạy thử
python -m src.padim.run                                   # đủ 15 loại
```
Triển khai tự do (thư viện, backbone, tầng, số chiều); chỉ cần giữ giao thức đánh giá và ghi cấu hình
([docs/giao-dien-chung.md](../../docs/giao-dien-chung.md) mục 0). Nếu dùng Anomalib: `src/common/anomalib_runner.py`.

Lý thuyết và điểm yếu: [docs/phuong-phap/padim.md](../../docs/phuong-phap/padim.md).
Thẻ Trello: "[Dưỡng] Đọc bài PaDiM, tự cài…", "[Dưỡng] Chạy thử PaDiM trên 3 loại…",
"[Dưỡng] Chạy PaDiM đủ 15 loại".
