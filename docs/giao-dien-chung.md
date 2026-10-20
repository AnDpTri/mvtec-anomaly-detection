# Giao diện chung giữa các phần

**Mục tiêu:** thống nhất dữ liệu trao đổi giữa các phần, để mỗi người viết phần mình mà ghép lại vẫn chạy.
**Lý do:** ba phương pháp phải đánh giá bằng cùng một module `src/evaluation/` thì so sánh mới công bằng.
**Quy tắc:** đổi bất kỳ mục nào dưới đây phải báo người kia và sửa file này trong cùng Pull Request.

## 1. Cấu hình
Mọi phần đọc từ `configs/common.yaml` qua `src.common.config.load_config()`. Không ghi cứng kích thước ảnh,
đường dẫn, seed trong code.

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `data.root` | `datasets/MVTecAD` | Thư mục dữ liệu |
| `data.image_size` | `256` | Ảnh và mask resize về 256×256 |
| `data.normalize_mean/std` | chuẩn ImageNet | Chuẩn hóa ảnh |
| `seed` | `42` | Gọi `src.common.seed.set_seed(seed)` đầu mỗi lần chạy |
| `categories.texture / object` | 5 + 10 loại | Danh sách đủ 15 loại |

## 2. Dữ liệu: `src/data/mvtec.py` → mọi phương pháp
`MVTecDataset(root, category, split, image_size, mean, std)`, mỗi phần tử là một dict:

| Khóa | Kiểu | Hình dạng | Ghi chú |
|---|---|---|---|
| `image` | `torch.FloatTensor` | `3×256×256` | Đã chuẩn hóa; ảnh xám đã nhân thành 3 kênh |
| `mask` | `torch.FloatTensor` | `1×256×256` | Giá trị 0/1; ảnh good là toàn 0; resize bằng nearest |
| `label` | `int` | | 0 = tốt, 1 = lỗi |
| `defect_type` | `str` | | `"good"` hoặc tên kiểu lỗi, ví dụ `"scratch_head"` |
| `path` | `str` | | Đường dẫn ảnh gốc |

`split="train"`: chỉ `train/good`. `split="test"`: toàn bộ `test/`.

## 3. Đầu ra của mỗi phương pháp → `src/evaluation/`
Mỗi phương pháp, với mỗi loại sản phẩm, trả về cho toàn bộ tập test:

| Tên | Kiểu | Hình dạng | Ý nghĩa |
|---|---|---|---|
| `anomaly_maps` | `numpy.float32` | `N×256×256` | Càng lớn càng bất thường, cùng kích thước với mask |
| `image_scores` | `numpy.float32` | `N` | Điểm bất thường của cả ảnh (mặc định: max của anomaly map đã làm mượt) |
| `labels` | `numpy.int` | `N` | Lấy từ dataset |
| `masks` | `numpy.uint8` | `N×256×256` | Lấy từ dataset |

Thứ tự ảnh của 4 mảng phải giống nhau. Anomaly map **chưa cần** chuẩn hóa về [0, 1], vì AUROC
không phụ thuộc thang đo.

## 4. Kết quả: `results/results.csv`
Mỗi dòng là một (loại, phương pháp). Ghi bằng `src.evaluation.results.append_result`.

| Cột | Ví dụ | Ghi chú |
|---|---|---|
| `category` | `screw` | |
| `type` | `object` | `texture` hoặc `object` |
| `method` | `patchcore` | `ae`, `padim`, `patchcore`, `ae_numpy` |
| `image_auroc` | `0.982` | |
| `pixel_auroc` | `0.991` | |
| `train_time_s` | `312.5` | Giây, cho cả loại |
| `infer_time_ms` | `45.1` | Mili giây mỗi ảnh |
| `memory_mb` | `850` | Bộ nhớ mô hình (memory bank, ma trận hiệp phương sai, trọng số) |
| `backbone` | `wide_resnet50_2` | Để trống với AE |
| `note` | `coreset=0.1` | Cấu hình khác mặc định, nếu có |

## 5. Tên file kết quả phụ
- Hình: `results/figures/<method>/<category>/<tên ảnh>.png` (không đưa lên git)
- Checkpoint: `results/checkpoints/<method>/<category>.pt` (không đưa lên git)
