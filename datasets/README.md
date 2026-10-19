# Dữ liệu MVTec AD

Thư mục này không được đưa lên git. Mỗi người tự tải về máy.

## Cách tải
Link tải chính thức của MVTec (mydrive.ch) đã chết (lỗi 404, kiểm tra ngày 17/10/2026).
Dùng bản sao trên Hugging Face: <https://huggingface.co/datasets/foersben/mvtec-ad>
(giữ nguyên cấu trúc gốc, số ảnh khớp bộ gốc).

```bash
python tools/download_mvtec.py datasets/MVTecAD              # cả 15 loại (~4,9 GB)
python tools/download_mvtec.py datasets/MVTecAD bottle screw # chỉ một vài loại
```

## Cấu trúc sau khi tải
```
datasets/MVTecAD/
├── license.txt, readme.txt
└── <loại>/
    ├── train/good/             ảnh tốt để huấn luyện
    ├── test/good/              ảnh tốt để kiểm tra
    ├── test/<kiểu lỗi>/        ảnh lỗi để kiểm tra
    └── ground_truth/<kiểu lỗi>/  mask lỗi (<tên ảnh>_mask.png)
```

## Giấy phép
MVTec AD theo CC BY-NC-SA 4.0: chỉ dùng phi thương mại, phải trích dẫn:
> Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
