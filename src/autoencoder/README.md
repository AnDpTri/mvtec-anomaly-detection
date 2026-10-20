# autoencoder · Phụ trách: An

**Mục tiêu:** baseline tự xây: mạng học nén rồi vẽ lại ảnh tốt; chỗ ảnh vẽ lại khác ảnh gốc là lỗi.
**Lý do:** cần một mốc so sánh đơn giản, và là phương pháp duy nhất trong ba phương pháp thật sự huấn
luyện mạng neuron. **Ý nghĩa:** cho thấy giới hạn của so sánh pixel → động lực cho PaDiM, PatchCore.

| File | Nội dung |
|---|---|
| `model.py` | `ConvAutoencoder` (PyTorch, 3×256×256), `anomaly_map(model, images)` |
| `train.py` | Huấn luyện + đánh giá trên đủ 15 loại, ghi CSV (method = `ae`) |
| `numpy_ae/` | Bản fully connected viết tay bằng NumPy, đủ 15 loại (bài học, phụ lục) |

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.autoencoder.train --categories bottle carpet screw   # chạy thử
python -m src.autoencoder.train                                   # đủ 15 loại
```

## Đầu ra
Anomaly map N×256×256 và điểm ảnh, đưa vào `src/evaluation`
([docs/giao-dien-chung.md](../../docs/giao-dien-chung.md) mục 3).

Lý thuyết và điểm yếu: [docs/phuong-phap/autoencoder.md](../../docs/phuong-phap/autoencoder.md).
Thẻ Trello: "[An] Học PyTorch, viết khung autoencoder", "[An] Huấn luyện autoencoder thử…",
"[An] Cố định siêu tham số AE…", "[An] Huấn luyện và đánh giá autoencoder đủ 15 loại".
