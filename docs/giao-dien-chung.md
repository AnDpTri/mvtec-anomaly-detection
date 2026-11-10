# Giao diện chung giữa các phần

**Mục tiêu:** ghi lại dữ liệu trao đổi giữa các module để sửa một phần mà không làm hỏng phần khác.
**Quy tắc:** đổi mục nào dưới đây phải báo người kia và sửa file này trong cùng Pull Request.

## 0. Cái gì cố định
Ba phương pháp (Autoencoder, PaDiM, PatchCore) chạy qua cùng một giao diện fit/predict và cùng hàm tính chỉ số
trong `src/tn/`, nên so sánh công bằng. Siêu tham số nằm ở `src/tn/config.py`; mô hình dùng cho demo
còn có cấu hình riêng từng loại trong `tools/build_tn_models.py` (cỡ ảnh, tỷ lệ coreset).
Không chỉnh siêu tham số theo kết quả trên tập test.

## 1. Dữ liệu: `src/tn/data.py`
Mỗi bộ trả về dict:

| Hàm | Trả về |
|---|---|
| `load_mvtec(root, loại)` | `train`: danh sách đường dẫn; `test`: danh sách dict |
| `load_ad2(root, loại)` | thêm `val` (validation/good); mỗi mục test có thêm `cond`, `scene` |
| `load_visa(root, split, loại)` | như MVTec AD; `split` là bảng `split_csv/1cls.csv` |

Mỗi mục `test` là `dict(path, label, dtype, mask, ...)`: `label` 0 = tốt, 1 = lỗi; `dtype` là `"good"` hoặc kiểu lỗi;
`mask` là đường dẫn hoặc `None` (ảnh tốt).
`load_imgs(paths, hw)` đọc ảnh thành `uint8 (N, H, W, 3)` (ảnh xám tự thành 3 kênh, resize `INTER_AREA`);
`load_masks(items, hw, thr)` đọc mask bool `(N, H, W)` (resize nearest; MVTec lưu 0/255 nên `thr=127`, VisA lưu 0/1 nên `thr=0`).

## 2. Phương pháp: `src/tn/methods.py`

| Hàm | Ý nghĩa |
|---|---|
| `fit_method(method, imgs, seed=0, coreset=None)` | `method` là `"ae"`, `"padim"` hoặc `"patchcore"`; trả `(tên, trạng thái)` |
| `predict_method(model, imgs)` | anomaly map `float32 (N, H, W)` cùng cỡ ảnh, càng lớn càng bất thường |
| `save_model(model, thư mục)`, `load_model(thư mục, phương pháp)` | ghi / nạp trạng thái đã fit (mục 4) |

Điểm ảnh = `metrics.image_scores(maps)` (max của map). Map chưa chuẩn hóa về [0, 1] vì AUROC không phụ thuộc thang đo.

## 3. Chỉ số: `src/tn/metrics.py`
`compute_metrics(scores, maps, labels, masks)` trả `i_auroc`, `i_ap`, `i_f1max`, `p_auroc`, `p_ap`, `aupro`.
`thresholds_from_val(val_maps)` trả `thr_img`, `thr_pix` (trung bình + 3 độ lệch chuẩn trên ảnh good validation);
`metrics_full(...)` thêm chỉ số tại ngưỡng đó. Tính riêng từng loại, không gộp ảnh của nhiều loại.

## 4. Mô hình đã lưu: `results/checkpoints_tn/<loại>/`
Ghi bằng `tools/build_tn_models.py`, nạp bằng `src/demo/tn_models.py` (không fit lại). Đổi định dạng thì
mô hình đã fit sẵn không nạp được nữa.

| File | Nội dung |
|---|---|
| `ae.pt` | `state_dict` của `ConvAE` |
| `padim.pt` | `mu`, `inv`, `H`, `W` |
| `patchcore_bank.pt` | memory bank fp16 sau coreset |
| `meta.json` | `category`, `dataset`, `hw`, `coreset`, `n_fit`, `val_paths`, `thr[phương pháp]` (`thr_img`, `thr_pix`, `lo`), `metrics[phương pháp]` |
| `test_scores.csv` | `path`, `dtype`, `label`, `mask` (tương đối so với `datasets/`), `score_<phương pháp>` |

Không đưa file này lên git (`.gitignore` đã chặn `*.pt`).

## 5. Kết quả thí nghiệm: `src/tn/results.py`
`Results(root)` ghi mỗi run một dòng vào `<root>/<thí nghiệm>_<phương pháp>.csv`, cột `key` là khóa của run;
chạy lại thì bỏ qua các `key` đã có. Notebook trong `notebooks/tn/` dùng lớp này.
