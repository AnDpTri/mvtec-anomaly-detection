# Autoencoder thuần NumPy (bài học, không thuộc bảng kết quả chính) · Phụ trách: An

**Mục tiêu:** tự viết toàn bộ một mạng neuron bằng NumPy (lan truyền xuôi, lan truyền ngược,
Adam, AUROC) để hiểu PyTorch đang làm gì bên dưới. Chạy trên **đủ 15 loại** MVTec AD.

**Lý do:** baseline chính thức là conv AE bằng PyTorch (`src/models/autoencoder.py`), cùng cấu hình
256×256 với PaDiM/PatchCore. Bản NumPy dùng ảnh xám 128×128 và mạng fully connected, nên không
so sánh công bằng được, nhưng cho thấy rõ vì sao cần CNN.

**Thiết kế:**
- Ảnh xám 128×128 → vector 16.384, chuẩn hóa về [0, 1]
- Mạng: 16.384 → 512 → 64 → 512 → 16.384 (ReLU ở tầng ẩn, sigmoid ở đầu ra)
- Loss MSE, tối ưu Adam viết tay
- Heatmap = |ảnh gốc − ảnh vẽ lại|, điểm ảnh = max heatmap
- Gỡ lỗi trên bottle trước, rồi chạy đủ 15 loại với cùng siêu tham số
- Chỉ cần `numpy` và `pillow`, chạy CPU; kết quả ghi vào `results/results.csv` với `method = ae_numpy`

**Ý nghĩa:** thấy tận mắt vì sao mạng fully connected vẽ lại mờ, kém trên texture nhiều chi tiết và
vật xoay (screw, metal_nut, hazelnut); đưa vào phụ lục như thí nghiệm "vì sao cần CNN".

Lý thuyết: [../docs/phuong-phap/autoencoder.md](../docs/phuong-phap/autoencoder.md).
Checklist: thẻ Trello "[An] Autoencoder thuần NumPy trên đủ 15 loại".
