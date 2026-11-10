# Chạy demo từ mô hình đã lưu

Demo đọc mô hình đã fit, không fit lại khi tải ảnh. Chạy từ thư mục gốc:

```powershell
python -m src.demo.app --host 127.0.0.1 --port 7864
```

Mở `http://127.0.0.1:7864`, chọn loại sản phẩm, phương pháp và ảnh, hoặc bấm một ảnh mẫu. Điểm là độ bất thường, không phải xác suất lỗi.

## Loại sản phẩm

9 loại của thí nghiệm TN1–TN5, mỗi loại đủ Autoencoder, PaDiM, PatchCore:

| Bộ | Loại | Kích thước ảnh | Coreset PatchCore |
|---|---|---|---|
| MVTec AD | grid, screw, transistor | 256×256 | 10% |
| MVTec AD 2 | vial, sheet_metal | 256×192, 256×1024 | 1% |
| VisA | pcb1–pcb4 | 256×320 | 1% |

Ảnh mẫu trong giao diện lấy từ `datasets/`; thiếu dữ liệu thì demo vẫn chạy, chỉ không có ảnh mẫu.

## Tạo mô hình

Trên Colab (T4, khoảng 30 phút cho cả 9 loại): mở `goi_colab/build_tn_models_colab.ipynb`, chọn GPU T4, *Run all*. Kết quả ghi vào `MyDrive/goi_colab/checkpoints_tn/`; tải thư mục đó về và đặt thành `results/checkpoints_tn/`.

Trên máy có GPU, khi đã có dữ liệu trong `datasets/MVTecAD`, `datasets/MVTecAD2`, `datasets/VisA`:

```powershell
python tools/build_tn_models.py              # cả 9 loại
python tools/build_tn_models.py grid pcb1    # một vài loại
```

Loại đã có `meta.json` được bỏ qua, nên chạy lại được sau khi bị ngắt. Notebook Colab được sinh từ code bằng `python tools/make_colab_notebook.py`; sửa `src/tn` hoặc script thì sinh lại notebook.

Mỗi loại nằm trong `results/checkpoints_tn/<loại>/`, tổng khoảng 2,6 GB:

| File | Nội dung |
|---|---|
| `ae.pt` | state_dict của `src.tn.methods.ConvAE` (ghi bằng `save_model`, nạp bằng `load_model`) |
| `padim.pt` | `mu`, `inv` (nghịch đảo hiệp phương sai mỗi vị trí), `H`, `W` |
| `patchcore_bank.pt` | memory bank fp16 sau coreset |
| `meta.json` | kích thước ảnh, ngưỡng, chỉ số trên tập test |
| `test_scores.csv` | điểm của từng ảnh test, cả 3 phương pháp |

Nạp bằng `torch.load(weights_only=True)`. `TN_CHECKPOINT_ROOT` đổi nơi demo đọc mô hình.

## Ngưỡng

Ngưỡng ảnh = trung bình + 3 độ lệch chuẩn của điểm trên ảnh good validation, tính riêng cho từng loại và phương pháp. Validation là 15% train/good tách ra trước khi fit (seed 0); MVTec AD 2 dùng `validation/good` có sẵn. Không dùng ảnh đã fit, không dùng nhãn test. Ngưỡng pixel tính cùng cách và làm đầu thang màu của heatmap.

Ngưỡng chặt: tỷ lệ báo nhầm ảnh tốt trên tập test 0–14%, đổi lại nhiều ảnh lỗi nhẹ không vượt ngưỡng. Chỉ số của từng mô hình nằm trong `meta.json` và hiện ngay trong ô kết luận.

## Kiểm tra ngày 03/11/2026

Fit trên Colab T4. I-AUROC, AUPRO, FPR tại ngưỡng của cả 27 mô hình khớp CSV thí nghiệm trong `Do_An_Deep_Learning/resuilts_6` (lệch I-AUROC tối đa 0,001). Điểm demo tính lại trên máy trùng điểm lúc fit trong `test_scores.csv`.

Kiểm thử: `python -m unittest discover -s tests -v`.
