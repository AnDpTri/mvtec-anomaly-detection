# data · Phụ trách: Dưỡng

**Mục tiêu:** đọc MVTec AD thành dữ liệu PyTorch đúng một định dạng chung cho mọi phương pháp.
**Lý do:** autoencoder và PaDiM tự cài đều cần cùng một cách đọc ảnh, mask, nhãn; viết một chỗ thì
không ai xử lý lệch nhau. **Ý nghĩa:** đây là phần cả nhóm phụ thuộc, nên làm sớm nhất (hạn 22/10).

## Đầu vào → Đầu ra
`MVTecDataset(root, category, split, image_size, mean, std)` trong `mvtec.py`, mỗi phần tử là dict:
`image` (3×256×256), `mask` (1×256×256, 0/1), `label` (0/1), `defect_type`, `path`.
Chi tiết: [docs/giao-dien-chung.md](../../docs/giao-dien-chung.md) mục 2.

## Lưu ý bắt buộc
- grid, screw, zipper là ảnh xám → nhân thành 3 kênh.
- Ảnh good không có mask → mask toàn 0.
- Mask resize bằng nearest rồi nhị phân hóa lại; crop ảnh thì crop mask y hệt.

Đặc điểm 15 loại: [docs/du-lieu.md](../../docs/du-lieu.md). Tải dữ liệu: [datasets/README.md](../../datasets/README.md).
Thẻ Trello: "[Dưỡng] Tải MVTec AD, viết DataLoader, thống kê và vẽ dữ liệu".
