"""[An] Huấn luyện và đánh giá conv autoencoder trên các loại MVTec AD.

Cách dùng (dự kiến):
    python scripts/train_ae.py --categories bottle carpet screw   # chạy thử
    python scripts/train_ae.py                                   # đủ 15 loại

Yêu cầu (thẻ Trello "Cố định siêu tham số AE, viết script chạy 15 loại"):
- Cùng siêu tham số (epoch, lr, bottleneck) cho mọi loại; không chỉnh theo tập test.
- Chỉ huấn luyện trên ảnh good, loss MSE; không lật/xoay ảnh khi augment.
- Ghi kết quả vào CSV sau mỗi loại; chạy lại thì bỏ qua loại đã xong.
- Lưu checkpoint (trên Colab: lưu vào Google Drive).
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại trong configs/common.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    # TODO: load_config + set_seed → với mỗi loại: MVTecDataset train/test → ConvAutoencoder
    #       → huấn luyện → anomaly_map → evaluate.image_auroc / pixel_auroc → append_result
    raise NotImplementedError


if __name__ == "__main__":
    main()
