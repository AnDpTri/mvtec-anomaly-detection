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
Xem `configs/common.yaml`: ảnh 256×256, chuẩn hóa ImageNet, seed 42, cùng `src/evaluate.py`,
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

## Cấu trúc
```
configs/common.yaml        cấu hình chung
src/data/mvtec.py          DataLoader MVTec AD                     [Dưỡng]
src/models/autoencoder.py  conv autoencoder                        [An]
src/models/padim.py        PaDiM tự cài                            [Dưỡng]
src/evaluate.py            image/pixel-AUROC, vẽ heatmap, ghi CSV   [Dưỡng]
scripts/train_ae.py        huấn luyện + đánh giá AE                [An]
scripts/run_anomalib.py    PaDiM, PatchCore bằng Anomalib          [Dưỡng / Chung]
numpy_ae/                  autoencoder thuần NumPy (bài học)       [An]
demo/app.py                demo Gradio                             [An]
tools/                     tải dữ liệu, ghép ảnh mẫu
results/                   kết quả (chỉ CSV được đưa lên git)
```

## Phân công
- **An (Đặng Hoàng An):** Autoencoder, demo Gradio, README.
- **Dưỡng (Trần Văn Dưỡng):** PaDiM, data pipeline, `evaluate.py`, thí nghiệm phụ.
- **PatchCore:** làm chung (An 8 loại, Dưỡng 7 loại).

Kế hoạch chi tiết và checklist: board Trello của nhóm. Hạn nộp: 27/11/2026.

## Quy ước
- Mọi thay đổi qua Pull Request, người kia duyệt trước khi merge vào `main`.
- Không đưa dữ liệu, checkpoint, memory bank lên git.

## Trích dẫn
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Anomalib: <https://github.com/open-edge-platform/anomalib>

Dữ liệu MVTec AD theo giấy phép CC BY-NC-SA 4.0 (chỉ dùng phi thương mại).
