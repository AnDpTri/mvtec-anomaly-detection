# Kết quả của các mô hình demo: Autoencoder, PaDiM, PatchCore trên 9 loại

Số liệu của bộ mô hình trong `results/checkpoints_tn/`, tạo bằng `tools/build_tn_models.py` (fit trên Colab T4, 03/11/2026).
Nguồn: `meta.json` của từng loại (trên máy, không đưa lên git) và `results/tn_cost.csv` (thời gian suy luận, dung lượng).
Kết quả của các thí nghiệm TN1–TN5 nằm ở Chương 3 của báo cáo (`bao_cao/`) và trong `notebooks/tn/`.

## 1. Cách đo

- Chỉ fit trên ảnh good. MVTec AD và VisA: fit 85% `train/good`, 15% còn lại làm validation đặt ngưỡng (seed 0).
  MVTec AD 2: fit `train/good`, ngưỡng từ `validation/good` có sẵn.
- Ảnh đưa về cỡ cố định của từng loại (xem bảng 2); anomaly map cùng cỡ với mask. Điểm ảnh = max của map đã làm mượt Gauss.
- Chỉ số tính trên tập test của từng loại, mỗi loại một mô hình. Công thức ở [danh-gia.md](danh-gia.md), cài đặt ở `src/tn/metrics.py`.

## 2. Dữ liệu từng loại

| Loại | Bộ | Cỡ ảnh | Fit | Validation | Test tốt | Test lỗi |
|---|---|---|---|---|---|---|
| grid | MVTec AD | 256×256 | 224 | 40 | 21 | 57 |
| screw | MVTec AD | 256×256 | 272 | 48 | 41 | 119 |
| transistor | MVTec AD | 256×256 | 181 | 32 | 60 | 40 |
| vial | MVTec AD 2 | 256×192 | 291 | 41 | 35 | 105 |
| sheet_metal | MVTec AD 2 | 256×1024 | 137 | 19 | 24 | 90 |
| pcb1 | VisA | 256×320 | 768 | 136 | 100 | 100 |
| pcb2 | VisA | 256×320 | 766 | 135 | 100 | 100 |
| pcb3 | VisA | 256×320 | 769 | 136 | 101 | 100 |
| pcb4 | VisA | 256×320 | 768 | 136 | 101 | 100 |

Coreset PatchCore: 10% cho MVTec AD, 1% cho MVTec AD 2 và VisA.

## 3. Chỉ số trên tập test

| Loại | I-AUROC AE | PaDiM | PatchCore | P-AUROC AE | PaDiM | PatchCore | AUPRO AE | PaDiM | PatchCore |
|---|---|---|---|---|---|---|---|---|---|
| grid | 0.836 | 0.935 | 0.988 | 0.834 | 0.946 | 0.981 | 0.607 | 0.870 | 0.902 |
| screw | 0.766 | 0.708 | 0.943 | 0.978 | 0.970 | 0.991 | 0.908 | 0.889 | 0.949 |
| transistor | 0.786 | 0.967 | 0.999 | 0.899 | 0.970 | 0.979 | 0.718 | 0.929 | 0.952 |
| vial | 0.686 | 0.633 | 0.842 | 0.887 | 0.882 | 0.923 | 0.775 | 0.754 | 0.797 |
| sheet_metal | 0.601 | 0.778 | 0.771 | 0.692 | 0.842 | 0.893 | 0.245 | 0.393 | 0.431 |
| pcb1 | 0.945 | 0.865 | 0.974 | 0.957 | 0.988 | 0.998 | 0.867 | 0.897 | 0.940 |
| pcb2 | 0.928 | 0.836 | 0.955 | 0.985 | 0.981 | 0.987 | 0.883 | 0.840 | 0.901 |
| pcb3 | 0.875 | 0.764 | 0.966 | 0.975 | 0.986 | 0.989 | 0.838 | 0.824 | 0.866 |
| pcb4 | 0.954 | 0.963 | 0.994 | 0.919 | 0.966 | 0.984 | 0.586 | 0.835 | 0.892 |

I-AUROC: phân loại ảnh tốt/lỗi. P-AUROC: pixel. AUPRO: vùng lỗi, tới FPR 0,3.

## 4. Thời gian suy luận và dung lượng

`results/tn_cost.csv` có ms/ảnh (từng ảnh một, trên GPU của máy đo) và MB của file mô hình cho mọi loại × phương pháp.
Tạo lại bằng `python tools/tn_report_figures.py --chi-do`. Thời gian phụ thuộc phần cứng nên chỉ so được giữa các dòng đo cùng lượt.

## 5. Tạo lại

```bash
python tools/build_tn_models.py          # fit và ghi meta.json (loại đã có thì bỏ qua; xóa meta.json để fit lại)
python tools/tn_report_figures.py        # tn_cost.csv và hình cho báo cáo, không fit lại
```
