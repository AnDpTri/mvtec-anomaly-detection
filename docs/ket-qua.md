# Kết quả thực nghiệm: Autoencoder, PaDiM, PatchCore trên MVTec AD

Chạy ngày 25/10/2026 trên Google Colab (GPU Tesla T4), đủ 15 loại. Số liệu gốc: `results/results.csv`
(CSV được lưu trong Git; checkpoint và ảnh kết quả đầy đủ được lưu riêng trên máy và Google Drive).

## 1. Giao thức đánh giá (giống nhau cho cả 3 phương pháp)

- Chỉ huấn luyện/fit trên `train/good`; tập test gồm ảnh tốt và ảnh lỗi.
- Ảnh resize 256×256 (không center crop); anomaly map 256×256 khớp từng pixel với mask.
- Chỉ số: **image-AUROC** (phân loại ảnh tốt/lỗi) và **pixel-AUROC** (khoanh vùng lỗi), tính từng loại
  bằng `src/evaluation`, rồi lấy trung bình.

| | Autoencoder | PaDiM | PatchCore |
|---|---|---|---|
| Cài đặt | PyTorch tự viết | Anomalib 2.7.0 | Anomalib 2.7.0 |
| Backbone | không (học từ đầu) | resnet18 (ImageNet) | wide_resnet50_2 (ImageNet) |
| Cấu hình | Conv AE, latent 256, MSE, 100 epoch, lr 1e-3, batch 16, σ = 4 | layer1–3, n_features mặc định | layer2+3, coreset 10%, 9 láng giềng |
| Điểm ảnh | max của map đã làm mượt | pred_score của mô hình | pred_score của mô hình |
| Commit code | `37fb3e5` | `a839dd0` | `a839dd0` |

## 2. Tổng hợp

| Trung bình | Autoencoder | PaDiM | PatchCore |
|---|---|---|---|
| image-AUROC, texture (5 loại) | 0.784 | 0.945 | **0.993** |
| image-AUROC, object (10 loại) | 0.741 | 0.863 | **0.977** |
| **image-AUROC, 15 loại** | 0.756 | 0.890 | **0.983** |
| pixel-AUROC, texture | 0.781 | 0.944 | **0.970** |
| pixel-AUROC, object | 0.875 | 0.968 | **0.985** |
| **pixel-AUROC, 15 loại** | 0.843 | 0.960 | **0.980** |
| Tổng thời gian fit/train 15 loại | ~22 phút | **~6 phút** | ~30 phút |
| Suy luận trung bình (ms/ảnh) | 44.8* | **5.5** | 35.5 |
| Dung lượng mô hình mỗi loại | **45 MB** | 158 MB | 36–235 MB (TB 145 MB) |

\* Thời gian suy luận của AE có tính cả thời gian đọc ảnh từ Drive (PaDiM/PatchCore chỉ đo lượt chạy mô hình),
nên con số này cao hơn thực tế; chưa so sánh trực tiếp được.

Dung lượng: AE là trọng số mạng; PaDiM là mean + nghịch đảo hiệp phương sai; PatchCore là memory bank
sau coreset (chưa tính backbone wide_resnet50 ~276 MB dùng chung).

## 3. Chi tiết từng loại

### image-AUROC

| Loại | Nhóm | AE | PaDiM | PatchCore |
|---|---|---|---|---|
| carpet | texture | 0.553 | **0.988** | 0.988 |
| grid | texture | 0.857 | 0.829 | **0.990** |
| leather | texture | 0.807 | 0.999 | **1.000** |
| tile | texture | 0.766 | 0.943 | **1.000** |
| wood | texture | 0.938 | 0.966 | **0.987** |
| bottle | object | 0.926 | 0.998 | **1.000** |
| cable | object | 0.579 | 0.838 | **0.988** |
| capsule | object | 0.689 | 0.903 | **0.994** |
| hazelnut | object | 0.923 | 0.720 | **1.000** |
| metal_nut | object | 0.574 | 0.957 | **0.997** |
| pill | object | 0.603 | 0.824 | **0.942** |
| screw | object | 0.579 | 0.752 | **0.964** |
| toothbrush | object | **0.958** | 0.772 | 0.917 |
| transistor | object | 0.807 | 0.943 | **0.995** |
| zipper | object | 0.776 | 0.927 | **0.977** |

### pixel-AUROC

| Loại | Nhóm | AE | PaDiM | PatchCore |
|---|---|---|---|---|
| carpet | texture | 0.805 | 0.989 | **0.991** |
| grid | texture | 0.844 | 0.916 | **0.981** |
| leather | texture | 0.916 | 0.987 | **0.992** |
| tile | texture | 0.559 | 0.906 | **0.955** |
| wood | texture | 0.780 | 0.922 | **0.930** |
| bottle | object | 0.935 | 0.978 | **0.986** |
| cable | object | 0.877 | 0.953 | **0.985** |
| capsule | object | 0.531 | 0.983 | **0.990** |
| hazelnut | object | 0.968 | 0.972 | **0.988** |
| metal_nut | object | 0.887 | 0.941 | **0.987** |
| pill | object | 0.887 | 0.944 | **0.981** |
| screw | object | 0.965 | 0.974 | **0.989** |
| toothbrush | object | 0.932 | 0.985 | **0.989** |
| transistor | object | 0.899 | **0.976** | 0.973 |
| zipper | object | 0.865 | 0.970 | **0.981** |

### Thời gian fit/train (giây) và suy luận (ms/ảnh)

| Loại | AE train | PaDiM fit | PatchCore fit | AE infer | PaDiM infer | PatchCore infer |
|---|---|---|---|---|---|---|
| carpet | 98.2 | 22.8 | 157.7 | 58.6 | 6.2 | 44.4 |
| grid | 97.0 | 17.7 | 124.4 | 33.7 | 5.4 | 37.1 |
| leather | 91.0 | 22.6 | 120.5 | 52.9 | 5.4 | 35.4 |
| tile | 85.3 | 20.6 | 99.4 | 41.9 | 5.5 | 33.5 |
| wood | 91.3 | 29.8 | 117.8 | 61.8 | 5.4 | 36.3 |
| bottle | 79.1 | 20.4 | 82.4 | 38.1 | 5.6 | 31.1 |
| cable | 85.3 | 22.7 | 95.2 | 55.7 | 5.5 | 33.4 |
| capsule | 79.0 | 22.2 | 91.8 | 50.0 | 5.4 | 32.9 |
| hazelnut | 145.6 | 37.1 | 299.2 | 50.3 | 5.3 | 51.6 |
| metal_nut | 79.2 | 15.9 | 87.3 | 28.8 | 5.5 | 32.5 |
| pill | 97.0 | 20.8 | 131.4 | 36.4 | 5.5 | 38.0 |
| screw | 121.5 | 24.8 | 184.4 | 28.3 | 5.5 | 43.5 |
| toothbrush | 18.1 | 18.4 | 10.1 | 55.9 | 5.3 | 15.3 |
| transistor | 79.4 | 22.8 | 89.7 | 51.7 | 5.6 | 32.8 |
| zipper | 91.3 | 21.6 | 111.4 | 27.8 | 5.4 | 35.0 |

## 4. Nhận xét

1. **Thứ hạng rõ ràng: PatchCore > PaDiM > Autoencoder** ở cả hai chỉ số và cả hai nhóm loại. Hai phương pháp
   dùng đặc trưng pretrained ImageNet vượt xa AE học từ đầu (image-AUROC 0.983 và 0.890 so với 0.756).
2. **Autoencoder** thất bại ở nhiều loại: image-AUROC của carpet, cable, metal_nut, screw chỉ ~0.55–0.58
   (gần mức đoán ngẫu nhiên 0.5); pixel-AUROC của tile (0.559) và capsule (0.531) cũng gần ngẫu nhiên.
   Ảnh vẽ lại bị mờ nên sai số xuất hiện khắp ảnh chứ không chỉ ở vùng lỗi; với texture, mạng có thể vẽ lại
   được cả chỗ lỗi.
3. **AE vẫn có chỗ mạnh:** toothbrush (0.958, cao nhất trong 3 phương pháp) và hazelnut (0.923, hơn PaDiM 0.720) —
   các loại có nền đơn giản và lỗi lớn, dễ thấy.
4. **PaDiM** nhanh nhất (fit ~20 s/loại, suy luận ~5.5 ms/ảnh), pixel-AUROC tốt (0.960) nhưng image-AUROC yếu ở
   hazelnut, screw, toothbrush, grid, pill, cable (0.72–0.84). Bài gốc dùng backbone lớn hơn (wide_resnet50)
   nên cho số cao hơn; ở đây dùng resnet18.
5. **PatchCore** chính xác nhất, sát số công bố trong bài gốc (coreset 10%: image ~0.990, pixel ~0.981; ở đây
   0.983 / 0.980). Chênh lệch nhỏ có thể do giao thức ở đây không center crop. Đổi lại, fit chậm hơn PaDiM ~5 lần
   (bước chọn coreset tuần tự, thời gian tăng theo bình phương số ảnh: hazelnut 391 ảnh mất 299 s) và suy luận
   chậm hơn ~6 lần.
6. **Loại khó chung:** pill và toothbrush là hai loại PatchCore thấp nhất (0.942, 0.917); toothbrush chỉ có 60 ảnh
   train.

## 5. Lưu ý về độ lặp lại

- PaDiM và PatchCore không huấn luyện mạng: chạy lại cùng code, dữ liệu và seed cho kết quả gần như trùng.
  Ví dụ bottle: PaDiM 0.9976 / 0.9776 trên cả RTX 3050 và T4; PatchCore 1.000 / 0.986 và 1.000 / 0.9855.
- Autoencoder có yếu tố ngẫu nhiên khi huấn luyện trên GPU: bottle đạt 0.902 trên RTX 3050 và 0.926 trên T4 với
  cùng code. Trọng số đã lưu ở `results/checkpoints/ae/<loại>.pt`.
- Khi chạy thử AE trên bottle, loss tăng vọt ở epoch 80 (0.0008 → 0.015) rồi hồi lại 0.002: lr 1e-3 có thể hơi
  cao. Chưa điều chỉnh; có thể thử giảm lr dần theo epoch (phải ghi vào cột `note`).

## 6. Ảnh minh họa

Mỗi ảnh gồm: ảnh gốc | mask thật | heatmap. Có cho mọi loại và mọi phương pháp tại
`results/figures/<ae|padim|patchcore>/<loại>/<kiểu lỗi>_000.png` (chỉ có trên máy và Drive).

Ví dụ hazelnut, lỗi nứt (`crack`):

| Autoencoder | PaDiM | PatchCore |
|---|---|---|
| ![](../results/figures/ae/hazelnut/crack_000.png) | ![](../results/figures/padim/hazelnut/crack_000.png) | ![](../results/figures/patchcore/hazelnut/crack_000.png) |

Ví dụ tile, lỗi nứt (`crack`):

| Autoencoder | PaDiM | PatchCore |
|---|---|---|
| ![](../results/figures/ae/tile/crack_000.png) | ![](../results/figures/padim/tile/crack_000.png) | ![](../results/figures/patchcore/tile/crack_000.png) |

## 7. Tạo lại bảng

```bash
python -m src.evaluation.summary
```
