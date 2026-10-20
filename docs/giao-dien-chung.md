# Giao diện chung giữa các phần

**Mục tiêu:** thống nhất dữ liệu trao đổi giữa các phần, để mỗi người viết phần mình mà ghép lại vẫn chạy.
**Lý do:** ba phương pháp phải đánh giá bằng cùng một module `src/evaluation/` thì so sánh mới công bằng.
**Quy tắc:** đổi bất kỳ mục **cố định** nào dưới đây phải báo người kia và sửa file này trong cùng Pull Request.

## 0. Cái gì cố định, cái gì tự do
**Nguyên tắc:** cố định **cách đánh giá**, để so sánh công bằng; **cách triển khai** từng phương pháp thì
tự do, miễn là ghi lại cấu hình đã dùng (cột `implementation`, `backbone`, `note` trong CSV).

| Cố định (giao thức đánh giá) | Tự do (triển khai) |
|---|---|
| Dữ liệu: đủ 15 loại MVTec AD, huấn luyện **chỉ trên `train/good`**, đánh giá trên toàn bộ `test/` | Thư viện: Anomalib, tự cài, hay thư viện khác |
| Đầu ra đưa vào đánh giá: anomaly map **256×256** khớp với mask, và điểm ảnh (mục 3) | Kiến trúc, backbone, tầng đặc trưng, cỡ cổ chai, loss |
| Độ đo: image-AUROC, pixel-AUROC tính qua `src/evaluation/`, riêng từng loại | Siêu tham số: epoch, learning rate, batch, tỷ lệ coreset, số chiều… |
| Kết quả ghi vào `results/results.csv` đúng các cột (mục 4) | Tăng cường dữ liệu (xoay, lật…), độ phân giải đầu vào bên trong mô hình |
| Một mô hình riêng cho mỗi loại; không chỉnh siêu tham số dựa trên tập test | Cách tính điểm ảnh từ anomaly map (mặc định: max của map đã làm mượt) |

Các con số trong `docs/phuong-phap/` và trong code (ResNet18, WideResNet50, tầng, coreset 10%, cỡ mạng…)
chỉ là **điểm xuất phát theo bài gốc**, không bắt buộc. Đổi thì ghi lại; nếu kết quả khác bài báo thì
giải thích trong báo cáo.

**Riêng PatchCore** (hai người cùng chạy, chia loại): hai người phải **dùng chung một cấu hình** với nhau,
cấu hình đó là gì thì tùy nhóm chọn.

## 1. Cấu hình
Đọc từ `configs/common.yaml` qua `src.common.config.load_config()`; không ghi cứng đường dẫn, seed trong code.

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `data.root` | `datasets/MVTecAD` | Thư mục dữ liệu |
| `data.image_size` | `256` | Kích thước anomaly map và mask khi đánh giá (cố định); cũng là kích thước đầu vào mặc định |
| `data.normalize_mean/std` | chuẩn ImageNet | Mặc định cho backbone ImageNet; phương pháp khác có thể chuẩn hóa khác |
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
Phương pháp nào cần đọc ảnh kiểu khác (độ phân giải khác, không chuẩn hóa, dùng loader của Anomalib) thì
được, miễn đầu ra đánh giá vẫn đúng mục 3.

## 3. Đầu ra của mỗi phương pháp → `src/evaluation/`
Mỗi phương pháp, với mỗi loại sản phẩm, trả về cho toàn bộ tập test:

| Tên | Kiểu | Hình dạng | Ý nghĩa |
|---|---|---|---|
| `anomaly_maps` | `numpy.float32` | `N×256×256` | Càng lớn càng bất thường, cùng kích thước với mask |
| `image_scores` | `numpy.float32` | `N` | Điểm bất thường của cả ảnh. Phương pháp tự tính theo cách của mình; nếu không có thì dùng `src.evaluation.metrics.image_score` (max của map đã làm mượt) |
| `labels` | `numpy.int` | `N` | Lấy từ dataset |
| `masks` | `numpy.uint8` | `N×256×256` | Lấy từ dataset |

Thứ tự ảnh của 4 mảng phải giống nhau. Anomaly map **chưa cần** chuẩn hóa về [0, 1], vì AUROC
không phụ thuộc thang đo.

## 4. Kết quả: `results/results.csv`
Mỗi dòng là một lần chạy (loại, phương pháp, cấu hình). Ghi bằng `src.evaluation.results.append_result`.

| Cột | Ví dụ | Ghi chú |
|---|---|---|
| `category` | `screw` | |
| `type` | `object` | `texture` hoặc `object` |
| `method` | `patchcore` | `ae`, `padim`, `patchcore`, `ae_numpy` |
| `implementation` | `anomalib` | `anomalib`, `tu-cai`, `pytorch`, `numpy`… |
| `image_auroc` | `0.982` | |
| `pixel_auroc` | `0.991` | |
| `train_time_s` | `312.5` | Giây, cho cả loại |
| `infer_time_ms` | `45.1` | Mili giây mỗi ảnh |
| `memory_mb` | `850` | Bộ nhớ mô hình (memory bank, ma trận hiệp phương sai, trọng số) |
| `backbone` | `wide_resnet50_2` | Để trống nếu không dùng backbone |
| `note` | `layers=2,3; coreset=0.1; size=256` | **Cấu hình đã dùng**: tầng, loss, siêu tham số, tăng cường dữ liệu… đủ để chạy lại được |

## 5. Tên file kết quả phụ
- Hình: `results/figures/<method>/<category>/<tên ảnh>.png` (không đưa lên git)
- Checkpoint: `results/checkpoints/<method>/<category>.pt` (không đưa lên git)
