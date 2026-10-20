# Gửi Dưỡng: tình hình dự án và việc cần làm

*Cập nhật 20/10/2026. Đọc hết khoảng 10 phút.*

## 1. Những gì đã có (An làm từ 17/10 đến 20/10)
| Đã xong | Ở đâu |
|---|---|
| Repo GitHub (public) | https://github.com/AnDpTri/mvtec-anomaly-detection |
| Khung mã nguồn chia theo module, mỗi module một người | `src/` (xem mục 3) |
| Tài liệu: phân công, giao diện chung, dữ liệu, đánh giá, 3 phương pháp | `docs/` |
| Quy trình nhánh / PR, CODEOWNERS, mẫu PR | `CONTRIBUTING.md`, `.github/` |
| Nhánh làm việc riêng: `duong/dev` (Dưỡng), `an/dev` (An) | GitHub |
| Script tải MVTec AD (bản sao Hugging Face) | `tools/download_mvtec.py` |
| Đã tải đủ **15 loại**, kiểm tra **khớp bảng 1 bài báo** (3.629 train, 467 test tốt, 1.258 test lỗi) | Máy An |
| Đọc bài MVTec AD, tóm tắt thành tài liệu | `docs/du-lieu.md` |
| Trello: sửa nguồn tải, thêm checklist từng thẻ, nhãn màu, đường dẫn file | Board nhóm |

**Code hiện mới là khung:** mỗi file có mô tả, tên hàm, `TODO`; chưa có code chạy thật.

## 2. Bắt đầu (làm một lần)
1. **Chấp nhận lời mời** vào repo (email từ GitHub, hoặc https://github.com/AnDpTri/mvtec-anomaly-detection/invitations).
2. Lấy code và tạo môi trường:
   ```bash
   git clone https://github.com/AnDpTri/mvtec-anomaly-detection.git
   cd mvtec-anomaly-detection
   python -m venv .venv
   .venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt    # PyTorch bản CUDA: cài theo pytorch.org
   ```
3. Tải dữ liệu (~4,9 GB). **Link chính thức của MVTec đã chết (404)**, script dùng bản sao Hugging Face:
   ```bash
   python tools/download_mvtec.py datasets/MVTecAD
   ```
   Chạy lại được nếu bị ngắt (file đủ dung lượng sẽ bỏ qua). Nếu dùng Colab: đưa thư mục này lên Google Drive.
4. Đọc theo thứ tự: `docs/phan-cong.md` → `docs/giao-dien-chung.md` → README trong module của mình.

## 3. Module của Dưỡng
```
src/
├── common/        [Chung]  config.py, seed.py, anomalib_runner.py
├── data/          [Dưỡng]  mvtec.py              ← làm đầu tiên
├── evaluation/    [Dưỡng]  metrics.py, visualize.py, results.py
├── autoencoder/   [An]
├── padim/         [Dưỡng]  model.py (tự cài), run.py
├── patchcore/     [Chung]  run.py                 ← Dưỡng chạy 7 loại
└── demo/          [An]
```
| Việc | File | Hạn |
|---|---|---|
| **DataLoader** | `src/data/mvtec.py` | **22/10** |
| **Đánh giá:** AUROC, vẽ heatmap, ghi CSV | `src/evaluation/*` | **26/10** |
| PaDiM tự cài | `src/padim/model.py` | 29/10 |
| Chạy PaDiM (thử 3 loại → đủ 15), bằng Anomalib hoặc bản tự cài | `src/padim/run.py` | 4/11 → 8/11 |
| Tiện ích chạy Anomalib dùng chung, nếu nhóm chọn Anomalib (làm cùng An) | `src/common/anomalib_runner.py` | trước 1/11 |
| PatchCore 7 loại: pill, screw, tile, toothbrush, transistor, wood, zipper | `src/patchcore/run.py` | 6/11 → 10/11 |
| Thí nghiệm phụ (backbone, coreset), đủ 15 loại | `src/padim`, `src/patchcore` | 9/11 → 11/11 |

## 4. Vì sao `data` và `evaluation` gấp nhất
Cả nhóm phụ thuộc vào hai module này: autoencoder của An đọc dữ liệu qua `src/data` và tính AUROC qua
`src/evaluation`; PaDiM, PatchCore cũng vậy. **Làm xong sớm thì An không phải chờ.**

Phải trả về **đúng định dạng** trong `docs/giao-dien-chung.md`:
- `MVTecDataset` trả dict: `image` 3×256×256 đã chuẩn hóa ImageNet, `mask` 1×256×256 (0/1), `label`, `defect_type`, `path`.
- `evaluation` nhận `anomaly_maps` N×256×256 + `image_scores` N; ghi CSV với các cột: category, type, method,
  image_auroc, pixel_auroc, train_time_s, infer_time_ms, memory_mb, backbone, note.

Lưu ý dữ liệu (đã kiểm tra trên bộ thật):
- grid, screw, zipper là **ảnh xám 1 kênh** → nhân thành 3 kênh.
- Ảnh good **không có mask** → tạo mask toàn 0.
- Mask ở `ground_truth/<kiểu lỗi>/<tên ảnh>_mask.png`; resize bằng **nearest** rồi nhị phân hóa lại.
- Kích thước gốc khác nhau theo loại (700 đến 1024), tất cả resize về 256.
- Kiểm tra `evaluation` trên trường hợp biết trước: điểm = nhãn → AUROC = 1.0; điểm ngẫu nhiên → khoảng 0.5.

## 5. Cách làm việc
- **Nhánh riêng của Dưỡng là `duong/dev`** (của An là `an/dev`). Sau khi clone: `git switch duong/dev`.
- Mỗi ngày trước khi làm: `git pull origin main` để lấy phần An đã gộp.
- Xong một phần dùng được thì mở Pull Request **từ `duong/dev` vào `main`**, điền mẫu (mục tiêu, lý do,
  cách kiểm tra, thẻ Trello); An duyệt. Chi tiết: `CONTRIBUTING.md` mục 3–5.
- Chạy từ thư mục gốc dạng module: `python -m src.padim.run --categories bottle carpet screw`.
- **Triển khai tự do:** thư viện (Anomalib hay tự cài), backbone, tầng, siêu tham số tùy chọn; chỉ giữ cố
  định giao thức đánh giá và ghi cấu hình vào CSV (`docs/giao-dien-chung.md` mục 0).
- Không đưa dữ liệu, checkpoint, memory bank lên git (`.gitignore` đã chặn).
- Sửa `src/common/`, `src/patchcore/`, `configs/` hoặc `docs/giao-dien-chung.md` thì **báo An trước**.

## 6. Việc cần Dưỡng xác nhận
1. **Thẻ Trello "[Dưỡng] Tải MVTec AD, viết DataLoader…" đang bị đánh dấu Hoàn thành** nhưng checklist 0/6
   (có thể bấm nhầm). Phần tải dữ liệu An đã làm; phần DataLoader, thống kê, vẽ ảnh vẫn là của Dưỡng.
2. **Tăng cường dữ liệu:** bài gốc MVTec AD có xoay/lật ảnh khi huấn luyện; Trello của nhóm ghi *không*
   xoay/lật. Cần thống nhất và ghi lý do vào báo cáo.
3. **Khóa phiên bản thư viện** (`torch`, `torchvision`, `anomalib`) trước khi chạy lấy số (khoảng 30/10),
   để kết quả hai người so được với nhau.
4. **Bảo vệ nhánh `main`** đang tắt; khi cả hai đã quen quy trình PR thì bật lại (bắt buộc PR + 1 người duyệt).

## 7. Đọc thêm
| Tài liệu | Để làm gì |
|---|---|
| `docs/du-lieu.md` | Thống kê 15 loại, đặc điểm từng loại, điều bài gốc phát hiện |
| `docs/danh-gia.md` | AUROC hoạt động thế nào, cạm bẫy pixel-AUROC |
| `docs/phuong-phap/padim.md`, `patchcore.md` | Lý thuyết để tự cài PaDiM và viết báo cáo |
| `src/<module>/README.md` | Đầu vào, đầu ra, cách chạy của từng module |
