# Phân công theo module

**Mục tiêu:** mỗi module trong `src/` có một người chịu trách nhiệm chính, để hai người làm song song
trong thư mục của mình mà không sửa đè lên nhau. Kế hoạch theo ngày và checklist: board Trello của nhóm.

## Thành viên
| Người | GitHub | Nhánh làm việc | Phụ trách |
|---|---|---|---|
| Đặng Hoàng An | AnDpTri | `an/dev` | `src/autoencoder/` (PyTorch + NumPy), `src/demo/`, README |
| Trần Văn Dưỡng | NeoFortran | `duong/dev` | `src/data/`, `src/evaluation/`, `src/padim/`, thí nghiệm phụ |
| Cả hai | | qua PR vào `main` | `src/common/`, `src/patchcore/` (An 8 loại, Dưỡng 7 loại), `configs/`, báo cáo |

## Module
```
src/
├── common/        [Chung]  config.py, seed.py, anomalib_runner.py
├── data/          [Dưỡng]  mvtec.py
├── evaluation/    [Dưỡng]  metrics.py, visualize.py, results.py
├── autoencoder/   [An]     model.py, train.py, numpy_ae/
├── padim/         [Dưỡng]  model.py (tự cài), run.py
├── patchcore/     [Chung]  run.py
└── demo/          [An]     app.py
```
Mỗi module có `README.md` riêng: người phụ trách, mục tiêu, đầu vào/đầu ra, cách chạy, thẻ Trello.

## File, người, hạn
| File | Người | Việc | Hạn |
|---|---|---|---|
| `src/data/mvtec.py` | Dưỡng | DataLoader MVTec AD | 22/10 |
| `src/evaluation/*` | Dưỡng | AUROC, vẽ heatmap, ghi CSV | 26/10 |
| `src/padim/model.py` | Dưỡng | PaDiM tự cài | 29/10 |
| `src/padim/run.py` | Dưỡng | Chạy PaDiM đủ 15 loại (Anomalib hoặc tự cài) | 4/11 → 8/11 |
| `src/autoencoder/model.py` | An | Mô hình autoencoder (kiến trúc tự chọn) | 22/10 |
| `src/autoencoder/numpy_ae/model.py` | An | AE thuần NumPy, 15 loại (bài học) | 26/10 |
| `src/autoencoder/train.py` | An | Huấn luyện + đánh giá AE, 15 loại | 4/11 → 11/11 |
| `src/patchcore/run.py` | Cả hai | Chạy PatchCore đủ 15 loại (Anomalib hoặc tự cài) | 1/11 → 10/11 |
| `src/common/*` | Cả hai | Cấu hình, seed, tiện ích chạy Anomalib (tùy chọn) | Đổi phải báo nhau |
| `src/demo/app.py` | An | Demo Gradio | 15/11 |

PatchCore chia loại: **An** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut ·
**Dưỡng** pill, screw, tile, toothbrush, transistor, wood, zipper.

## Phụ thuộc giữa các module
```
common  ──> mọi module
data (Dưỡng) ───────┬──> autoencoder (An)
                    └──> padim/model.py (Dưỡng)
evaluation (Dưỡng) ─┬──> autoencoder (An)
                    └──> common/anomalib_runner ──> padim, patchcore
```
`data` và `evaluation` là hai module cả nhóm phụ thuộc, nên Dưỡng làm trước (tuần 1–2).

**Để An không phải chờ:** viết `autoencoder` theo đúng [giao-dien-chung.md](giao-dien-chung.md). Trong lúc
chờ `data`, có thể tạm đọc ảnh bằng `torchvision.datasets.ImageFolder` trên `train/` của một loại, và tạm
tính AUROC bằng `sklearn.metrics.roc_auc_score`; khi module của Dưỡng xong thì thay vào.

**`src/autoencoder/numpy_ae/` không phụ thuộc module nào**, làm được ngay.
