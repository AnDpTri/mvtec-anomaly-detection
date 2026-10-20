# padim · Phụ trách: Dưỡng

**Mục tiêu:** lập "bảng tiêu chuẩn" (trung bình + hiệp phương sai) cho từng vị trí từ ảnh tốt; mảnh nào
lệch xa tiêu chuẩn của đúng vị trí đó (khoảng cách Mahalanobis) là lỗi.
**Lý do:** so ở mức đặc trưng CNN thay vì pixel, và không cần huấn luyện mạng.
**Ý nghĩa:** cho thấy đặc trưng học sẵn + thống kê đơn giản vượt xa autoencoder; lộ điểm yếu khi vật xoay.

| File | Nội dung |
|---|---|
| `model.py` | PaDiM tự cài (~100 dòng) để hiểu và đối chiếu |
| `run_anomalib.py` | PaDiM bằng Anomalib trên đủ 15 loại, ghi CSV (method = `padim`) |

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.padim.run_anomalib --categories bottle carpet screw   # chạy thử
python -m src.padim.run_anomalib                                   # đủ 15 loại
```
Quy trình chung với PatchCore: `src/common/anomalib_runner.py`.

Lý thuyết và điểm yếu: [docs/phuong-phap/padim.md](../../docs/phuong-phap/padim.md).
Thẻ Trello: "[Dưỡng] Đọc bài PaDiM, tự cài…", "[Dưỡng] Chạy PaDiM bằng Anomalib trên 3 loại…",
"[Dưỡng] Chạy PaDiM đủ 15 loại".
