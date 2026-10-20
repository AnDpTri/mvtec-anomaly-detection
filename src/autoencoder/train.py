"""[An] Huấn luyện và đánh giá autoencoder trên các loại MVTec AD.

Cách dùng (dự kiến, chạy từ thư mục gốc dự án):
    python -m src.autoencoder.train --categories bottle carpet screw   # chạy thử
    python -m src.autoencoder.train                                   # đủ 15 loại

Yêu cầu (thẻ Trello "Cố định siêu tham số AE, viết script chạy 15 loại"):
- Chỉ huấn luyện trên ảnh good; không chỉnh siêu tham số dựa trên tập test.
- Loss, tăng cường dữ liệu (có xoay/lật hay không), siêu tham số: tự chọn, ghi vào cột note.
  Giá trị mặc định của các tham số dưới đây chỉ là điểm xuất phát.
- Ghi kết quả vào CSV sau mỗi loại; chạy lại thì bỏ qua loại đã xong.
- Lưu checkpoint (trên Colab: lưu vào Google Drive).
- Kết quả ghi vào results/results.csv với method = ae.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại trong configs/common.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    # TODO: src.common.config.load_config + src.common.seed.set_seed
    #       → với mỗi loại: src.data.mvtec.MVTecDataset train/test → src.autoencoder.model.ConvAutoencoder
    #       → huấn luyện → anomaly_map → src.evaluation.metrics → src.evaluation.results.append_result
    raise NotImplementedError


if __name__ == "__main__":
    main()
