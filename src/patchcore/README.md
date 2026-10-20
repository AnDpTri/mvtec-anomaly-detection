# patchcore · Phụ trách: Chung (An 8 loại, Dưỡng 7 loại)

**Mục tiêu:** gom đặc trưng của mọi mảnh tốt vào một kho (memory bank); mảnh mới nào cách mảnh tốt gần
nhất quá xa là lỗi.
**Lý do:** bỏ ràng buộc vị trí của PaDiM, nên vật xoay thế nào cũng có mảnh tương ứng để so.
**Ý nghĩa:** phương pháp chính xác nhất trong ba; thí nghiệm coreset cho thấy đánh đổi chính xác – bộ nhớ – tốc độ.

| File | Nội dung |
|---|---|
| `run_anomalib.py` | PatchCore bằng Anomalib, ghi CSV (method = `patchcore`) |

## Chia loại
| Người | Loại |
|---|---|
| An | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut |
| Dưỡng | pill, screw, tile, toothbrush, transistor, wood, zipper |

Hai người dùng cùng phiên bản Anomalib, backbone, tầng, tỷ lệ coreset, seed.

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.patchcore.run_anomalib --categories bottle cable capsule carpet grid hazelnut leather metal_nut
python -m src.patchcore.run_anomalib --categories pill screw tile toothbrush transistor wood zipper
```
Quy trình chung với PaDiM: `src/common/anomalib_runner.py`.

Lý thuyết và điểm yếu: [docs/phuong-phap/patchcore.md](../../docs/phuong-phap/patchcore.md).
Thẻ Trello: "[Chung·PatchCore] Đọc bài PatchCore, chạy thử…", "[Chung·PatchCore] Chạy PatchCore đủ 15 loại".
