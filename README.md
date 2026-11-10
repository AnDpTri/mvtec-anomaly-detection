# Phát hiện và định vị bất thường trên sản phẩm công nghiệp bằng học không giám sát

Đề tài 22, môn Học sâu, lớp KHMT K22A.
*Unsupervised Industrial Anomaly Detection and Localization on MVTec AD, MVTec AD 2 and VisA.*

## Mục tiêu
Dạy máy nhận ra **cái khác thường** khi nó **chỉ được thấy cái bình thường**:
- **Phát hiện:** cho biết ảnh sản phẩm tốt hay lỗi (image-AUROC).
- **Định vị:** chỉ ra vùng lỗi bằng heatmap (pixel-AUROC).
- **Ràng buộc:** chỉ huấn luyện trên ảnh tốt, vì trong thực tế ảnh lỗi hiếm, đa dạng và tốn công gán nhãn.

## Ba phương pháp
| Phương pháp | Vai trò | Cách đo "độ lạ" |
|---|---|---|
| Autoencoder (tự xây) | Baseline | Vẽ lại ảnh tốt; vùng vẽ lại sai là lỗi |
| PaDiM | Chính | Đặc trưng lệch khỏi phân phối Gauss của ảnh tốt tại cùng vị trí (Mahalanobis) |
| PatchCore | Chính | Đặc trưng cách mảnh tốt gần nhất trong memory bank bao xa |

Mỗi phương pháp sau khắc phục điểm yếu của phương pháp trước: so pixel → so đặc trưng → bỏ ràng buộc vị trí.

Cả ba phương pháp tự cài bằng PyTorch (`src/tn/methods/`, mỗi phương pháp một file), cùng giao diện fit/predict, cùng hàm tính chỉ số
(`src/tn/metrics.py`) và cùng siêu tham số chung (`src/tn/config.py`); backbone của PaDiM và PatchCore là
WideResNet-50-2 học trước trên ImageNet. Đề tài dùng 9 loại: MVTec AD (grid, screw, transistor),
MVTec AD 2 (vial, sheet_metal), VisA (pcb1–pcb4). Mỗi loại một mô hình, chỉ fit trên ảnh good,
kết quả tính riêng từng loại. Giao diện giữa các phần: [docs/giao-dien-chung.md](docs/giao-dien-chung.md).

## Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt  # PyTorch bản CUDA: cài theo hướng dẫn tại pytorch.org
```

## Dữ liệu
Không có trong repo. Xem [datasets/README.md](datasets/README.md):
```bash
python tools/download_mvtec.py datasets/MVTecAD grid screw transistor
```
MVTec AD 2 và VisA tải bằng notebook Colab (xem [datasets/README.md](datasets/README.md)).

## Cấu trúc
```
src/
├── tn/            [Dưỡng]  config, data, download, splits, metrics; gói methods/ (ae, padim, patchcore),
│                           experiments/ (tn1_tn2, tn2b, tn3, tn4), analysis/
└── demo/          [An]     demo Gradio: app.py, inference.py (nạp mô hình đã lưu, suy luận)
tools/                      fit mô hình demo, sinh notebook Colab, số liệu và hình cho báo cáo, tải MVTec AD
notebooks/tn/               notebook Colab chạy TN1–TN5
goi_colab/                  notebook Colab fit mô hình cho demo (sinh bằng tools/make_colab_notebook.py)
tests/                      kiểm thử CPU: chia tập, chỉ số, fit/predict, vòng chạy thí nghiệm, demo
results/                    kết quả (chỉ CSV được đưa lên git; mô hình trong results/checkpoints_tn/ không)
docs/                       tài liệu: phân công, giao diện chung, dữ liệu, đánh giá, phương pháp, demo
bao_cao/                    báo cáo LaTeX và bản Word
.github/                    CODEOWNERS (ai duyệt module nào), mẫu Pull Request, kiểm thử tự động
```

## Cách chạy (từ thư mục gốc dự án)
```bash
python tools/build_tn_models.py            # Fit mô hình cho demo (9 loại), ghi vào results/checkpoints_tn/
python -m src.demo.app                     # Demo, cổng 7864
python -m unittest discover -s tests -v    # Kiểm thử
python tools/make_colab_notebook.py        # Sinh lại goi_colab/build_tn_models_colab.ipynb sau khi sửa src/tn
```

Demo kết luận lỗi/không lỗi theo ngưỡng đặt trên ảnh good validation. Cách tạo mô hình (trên máy hoặc Colab) và cách
chọn ngưỡng: [hướng dẫn demo](docs/demo.md). Chỉ số của các mô hình: [docs/ket-qua.md](docs/ket-qua.md).
Chương 3 của [báo cáo LaTeX](bao_cao/main.tex) lấy bảng trực tiếp từ CSV.

## Kết quả và báo cáo

Chỉ số của 27 mô hình (9 loại × 3 phương pháp): [docs/ket-qua.md](docs/ket-qua.md). Thí nghiệm TN1–TN5: Chương 3 của báo cáo và `notebooks/tn/`.

- Báo cáo chỉnh sửa bằng Word, có trường Zotero: [bao_cao/bao_cao.docx](bao_cao/bao_cao.docx).
- Bản PDF biên dịch từ LaTeX: [bao_cao/main.pdf](bao_cao/main.pdf).
- Nguồn LaTeX: [bao_cao/main.tex](bao_cao/main.tex), dùng XeLaTeX + Biber.
- Script xuất Word: `bao_cao/xuat_word.py`, cần Pandoc, python-docx, Microsoft Word trên Windows;
  Zotero cùng Word add-in dùng để làm mới trích dẫn. Cài `python-docx` riêng nếu cần xuất Word.
  Trước khi xuất lại phải giữ các chỉnh sửa trực tiếp của người dùng trong bản Word.

Demo đọc mô hình đã lưu trong `results/checkpoints_tn/`; dataset và mô hình không nằm trên GitHub.

## Phân công
- **An (Đặng Hoàng An, AnDpTri):** `src/demo/`, README.
- **Dưỡng (Trần Văn Dưỡng, NeoFortran):** `src/tn/`, `notebooks/tn/`.
- **Chung:** `tools/`, `tests/`, `docs/`, báo cáo.

Chi tiết: [docs/phan-cong.md](docs/phan-cong.md). Kế hoạch theo ngày và checklist: board Trello của nhóm. Hạn nộp: 27/11/2026.

## Tài liệu
| | |
|---|---|
| [docs/giao-dien-chung.md](docs/giao-dien-chung.md) | Hàm và file trao đổi giữa các phần (đọc trước khi viết code) |
| [docs/du-lieu.md](docs/du-lieu.md) | Ba bộ dữ liệu, 9 loại đang dùng, lưu ý xử lý |
| [docs/danh-gia.md](docs/danh-gia.md) | AUROC, AUPRO, ngưỡng |
| [docs/phuong-phap/](docs/phuong-phap/) | Autoencoder, PaDiM, PatchCore: mục tiêu, cách hoạt động, điểm yếu |
| [docs/demo.md](docs/demo.md), [docs/ket-qua.md](docs/ket-qua.md) | Chạy demo, chỉ số các mô hình |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Quy trình nhánh, commit, Pull Request |

## Quy ước
- Mỗi người một nhánh làm việc riêng: `an/dev` (An), `duong/dev` (Dưỡng); gộp vào `main` qua Pull Request, người kia duyệt.
- Không đưa dữ liệu, checkpoint, memory bank lên git.

## Trích dẫn
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Heckler-Kram et al., *The MVTec AD 2 Dataset*; Zou et al., *SPot-the-Difference ... (VisA)*, ECCV 2022.

Dữ liệu MVTec AD theo giấy phép CC BY-NC-SA 4.0 (chỉ dùng phi thương mại); xem giấy phép của MVTec AD 2 và VisA trong thư mục dữ liệu.
