# Autoencoder thuần NumPy (bài học, không thuộc bảng kết quả chính)

**Mục tiêu:** tự viết toàn bộ một mạng neuron bằng NumPy (lan truyền xuôi, lan truyền ngược,
Adam, AUROC) để hiểu PyTorch đang làm gì bên dưới.

**Lý do:** baseline chính thức là conv AE bằng PyTorch (`src/models/autoencoder.py`), cùng cấu hình
256×256 với PaDiM/PatchCore. Bản NumPy dùng ảnh xám 128×128 và mạng fully connected, nên không
so sánh công bằng được, nhưng cho thấy rõ vì sao cần CNN.

**Thiết kế:**
- Ảnh xám 128×128 → vector 16.384, chuẩn hóa về [0, 1]
- Mạng: 16.384 → 512 → 64 → 512 → 16.384 (ReLU ở tầng ẩn, sigmoid ở đầu ra)
- Loss MSE, tối ưu Adam viết tay
- Heatmap = |ảnh gốc − ảnh vẽ lại|, điểm ảnh = max heatmap
- Chỉ cần `numpy` và `pillow`, chạy CPU

**Ý nghĩa:** thấy tận mắt vì sao mạng fully connected vẽ lại mờ và kém trên ốc vít xoay;
có thể đưa vào phụ lục như thí nghiệm "vì sao cần CNN".

Checklist: xem thẻ Trello "[An] Autoencoder thuần NumPy trên bottle, screw".
