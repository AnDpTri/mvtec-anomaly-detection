# Autoencoder thuần NumPy (bài học, không thuộc bảng kết quả chính) · Phụ trách: An

**Mục tiêu:** tự viết toàn bộ một mạng neuron bằng NumPy (lan truyền xuôi, lan truyền ngược,
Adam, AUROC) để hiểu PyTorch đang làm gì bên dưới. Chạy trên **đủ 15 loại** MVTec AD.

**Lý do:** baseline chính thức là conv AE bằng PyTorch (`src/autoencoder/model.py`), cùng cấu hình
256×256 với PaDiM/PatchCore. Bản NumPy nhằm hiểu tận gốc; nếu chạy khác giao thức (ảnh nhỏ hơn, ảnh xám…)
thì không so sánh công bằng với bảng chính được.

**Thiết kế: tự chọn.** Một điểm xuất phát gợi ý (không bắt buộc):
- Ảnh xám 128×128 → vector 16.384, chuẩn hóa về [0, 1]
- Mạng fully connected 16.384 → 512 → 64 → 512 → 16.384 (ReLU ở tầng ẩn, sigmoid ở đầu ra)
- Loss MSE, tối ưu Adam viết tay
- Heatmap = |ảnh gốc − ảnh vẽ lại|, điểm ảnh = max heatmap
- Gỡ lỗi trên bottle trước, rồi chạy đủ 15 loại với cùng siêu tham số (ghi cấu hình vào cột note)
- Chỉ cần `numpy` và `pillow`, chạy CPU; kết quả ghi vào `results/results.csv` với `method = ae_numpy`

**Ý nghĩa:** thấy tận mắt vì sao mạng fully connected vẽ lại mờ, kém trên texture nhiều chi tiết và
vật xoay (screw, metal_nut, hazelnut); đưa vào phụ lục như thí nghiệm "vì sao cần CNN".

Lý thuyết: [docs/phuong-phap/autoencoder.md](../../../docs/phuong-phap/autoencoder.md).
Checklist: thẻ Trello "[An] Autoencoder thuần NumPy trên đủ 15 loại".
