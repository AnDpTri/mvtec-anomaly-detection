# Phát hiện và định vị bất thường trên sản phẩm công nghiệp bằng học không giám sát

Đề tài 22, môn Học sâu, lớp KHMT K22A.
*Unsupervised Industrial Anomaly Detection and Localization on MVTec AD.*

## Mục tiêu
Dạy máy nhận ra **cái khác thường** khi nó **chỉ được thấy cái bình thường**:
- **Phát hiện:** cho biết ảnh sản phẩm tốt hay lỗi (image-AUROC).
- **Định vị:** chỉ ra vùng lỗi bằng heatmap (pixel-AUROC).
- **Ràng buộc:** chỉ huấn luyện trên ảnh tốt, vì trong thực tế ảnh lỗi hiếm, đa dạng và tốn công gán nhãn.

## Ba phương pháp
| Phương pháp | Vai trò | Cách đo "độ lạ" |
|---|---|---|
| Autoencoder (PyTorch, tự xây) | Baseline | Vẽ lại ảnh tốt; vùng vẽ lại sai là lỗi |
| PaDiM (Anomalib + bản tự cài) | Chính | Đặc trưng lệch khỏi phân phối Gauss của ảnh tốt tại cùng vị trí (Mahalanobis) |
| PatchCore (Anomalib) | Chính | Đặc trưng cách mảnh tốt gần nhất trong memory bank bao xa |

Mỗi phương pháp sau khắc phục điểm yếu của phương pháp trước: so pixel → so đặc trưng → bỏ ràng buộc vị trí.

## Cấu hình chung (để so sánh công bằng)
Xem `configs/common.yaml`: ảnh 256×256, chuẩn hóa ImageNet, seed 42, cùng module `src/evaluation/`,
AUROC tính riêng từng loại rồi mới lấy trung bình. Chạy thử trên bottle, carpet, screw trước khi chạy đủ 15 loại.

## Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt  # PyTorch bản CUDA: cài theo hướng dẫn tại pytorch.org
```

## Dữ liệu
Không có trong repo. Xem [datasets/README.md](datasets/README.md):
```bash
python tools/download_mvtec.py datasets/MVTecAD
```

## Cấu trúc: mỗi module một người phụ trách
```
src/
├── common/        [Chung]  cấu hình, seed, quy trình chạy Anomalib dùng chung
├── data/          [Dưỡng]  đọc MVTec AD (DataLoader)
├── evaluation/    [Dưỡng]  image/pixel-AUROC, vẽ heatmap, ghi CSV
├── autoencoder/   [An]     conv autoencoder (PyTorch) + numpy_ae/ (bản thuần NumPy)
├── padim/         [Dưỡng]  PaDiM tự cài + chạy bằng Anomalib
├── patchcore/     [Chung]  PatchCore bằng Anomalib (An 8 loại, Dưỡng 7 loại)
└── demo/          [An]     demo Gradio
configs/common.yaml         cấu hình chung
tools/                      tải dữ liệu, ghép ảnh mẫu
results/                    kết quả (chỉ CSV được đưa lên git)
docs/                       tài liệu: phân công, giao diện chung, dữ liệu, đánh giá, phương pháp
.github/                    CODEOWNERS (ai duyệt module nào), mẫu Pull Request
```
Mỗi module có `README.md` riêng: người phụ trách, mục tiêu, đầu vào/đầu ra, cách chạy.

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.autoencoder.train            # Autoencoder, đủ 15 loại
python -m src.padim.run_anomalib           # PaDiM
python -m src.patchcore.run_anomalib --categories bottle cable ...   # PatchCore theo phần chia
python -m src.demo.app                     # Demo
```

## Phân công
- **An (Đặng Hoàng An, AnDpTri):** `src/autoencoder/`, `src/demo/`, README.
- **Dưỡng (Trần Văn Dưỡng, NeoFortran):** `src/data/`, `src/evaluation/`, `src/padim/`, thí nghiệm phụ.
- **Chung:** `src/common/`, `src/patchcore/` (An 8 loại, Dưỡng 7 loại).

Mọi phương pháp chạy trên **đủ 15 loại**. Chi tiết: [docs/phan-cong.md](docs/phan-cong.md).
Kế hoạch theo ngày và checklist: board Trello của nhóm. Hạn nộp: 27/11/2026.

## Tài liệu
| | |
|---|---|
| [docs/giao-dien-chung.md](docs/giao-dien-chung.md) | Định dạng dữ liệu trao đổi giữa các phần (đọc trước khi viết code) |
| [docs/du-lieu.md](docs/du-lieu.md) | Thống kê và lưu ý xử lý 15 loại MVTec AD |
| [docs/danh-gia.md](docs/danh-gia.md) | Image-AUROC, pixel-AUROC |
| [docs/phuong-phap/](docs/phuong-phap/) | Autoencoder, PaDiM, PatchCore: mục tiêu, cách hoạt động, điểm yếu |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Quy trình nhánh, commit, Pull Request |

## Quy ước
- Mỗi người một nhánh làm việc riêng: `an/dev` (An), `duong/dev` (Dưỡng); gộp vào `main` qua Pull Request, người kia duyệt.
- Không đưa dữ liệu, checkpoint, memory bank lên git.

## Trích dẫn
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Anomalib: <https://github.com/open-edge-platform/anomalib>

Dữ liệu MVTec AD theo giấy phép CC BY-NC-SA 4.0 (chỉ dùng phi thương mại).
