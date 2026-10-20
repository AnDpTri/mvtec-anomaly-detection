# Phân công theo mã nguồn

**Mục tiêu:** mỗi file có một người chịu trách nhiệm chính, để hai người làm song song mà không sửa đè lên nhau.
Kế hoạch theo ngày và checklist chi tiết: board Trello của nhóm.

## Thành viên
| Người | GitHub | Phụ trách |
|---|---|---|
| Đặng Hoàng An | AnDpTri | Autoencoder (PyTorch + NumPy), demo Gradio, README |
| Trần Văn Dưỡng | NeoFortran | Dữ liệu, đánh giá, PaDiM, thí nghiệm phụ |
| Cả hai | | PatchCore (An 8 loại, Dưỡng 7 loại), cấu hình chung, báo cáo |

## File và người phụ trách
| File | Người | Việc (thẻ Trello) | Hạn |
|---|---|---|---|
| `src/data/mvtec.py` | Dưỡng | DataLoader MVTec AD | 22/10 |
| `src/evaluate.py` | Dưỡng | image/pixel-AUROC, vẽ heatmap, ghi CSV | 26/10 |
| `src/models/padim.py` | Dưỡng | PaDiM tự cài | 29/10 |
| `src/models/autoencoder.py` | An | Conv autoencoder PyTorch | 22/10 |
| `numpy_ae/autoencoder_numpy.py` | An | AE thuần NumPy, 15 loại (bài học) | 26/10 |
| `scripts/train_ae.py` | An | Huấn luyện + đánh giá AE, 15 loại | 4/11 → 11/11 |
| `scripts/run_anomalib.py` | Dưỡng (PaDiM), cả hai (PatchCore) | PaDiM, PatchCore bằng Anomalib, 15 loại | 4/11 → 10/11 |
| `demo/app.py` | An | Demo Gradio | 15/11 |
| `configs/common.yaml` | Cả hai | Cấu hình chung | Đổi phải báo nhau |
| `src/utils.py` | Cả hai | Đọc cấu hình, seed | |

PatchCore chia loại: **An** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut ·
**Dưỡng** pill, screw, tile, toothbrush, transistor, wood, zipper.

## Phụ thuộc giữa các phần
```
src/data/mvtec.py (Dưỡng) ──┬──> scripts/train_ae.py (An)
                            └──> src/models/padim.py (Dưỡng)
src/evaluate.py (Dưỡng) ────┬──> scripts/train_ae.py (An)
                            └──> scripts/run_anomalib.py (chung)
```
DataLoader và evaluate.py là hai thứ cả nhóm phụ thuộc, nên Dưỡng làm trước (tuần 1–2).

**Để An không phải chờ:** An viết AE theo đúng [giao-dien-chung.md](giao-dien-chung.md). Trong lúc chờ
DataLoader, có thể tạm đọc ảnh bằng `torchvision.datasets.ImageFolder` trên `train/` của một loại,
và tạm tính AUROC bằng `sklearn.metrics.roc_auc_score`. Khi phần của Dưỡng xong thì thay vào.

**Bản NumPy (`numpy_ae/`) không phụ thuộc ai**, làm được ngay.
