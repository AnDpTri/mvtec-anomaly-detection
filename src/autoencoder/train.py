"""[An] Huấn luyện và đánh giá autoencoder trên các loại MVTec AD.

Cách dùng (chạy từ thư mục gốc dự án):
    python -m src.autoencoder.train --categories bottle carpet screw   # chạy thử
    python -m src.autoencoder.train                                   # đủ 15 loại
    python -m src.autoencoder.train --data-root /content/drive/MyDrive/MVTecAD   # Colab

Quy ước:
- Chỉ huấn luyện trên ảnh train/good; không chỉnh siêu tham số dựa trên tập test.
- Cùng siêu tham số cho mọi loại; giá trị mặc định chỉ là điểm xuất phát, ghi vào cột note.
- Không tăng cường dữ liệu bằng xoay/lật (tránh biến lỗi thành "bình thường"); đổi được nếu cần.
- Ghi kết quả vào CSV sau mỗi loại; chạy lại thì bỏ qua loại đã xong (Colab hay bị ngắt).
- Kết quả ghi vào results/results.csv với method = ae.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.autoencoder.model import ConvAutoencoder, anomaly_map, reconstruction_loss
from src.common.config import ROOT, all_categories, category_type, load_config
from src.common.seed import set_seed
from src.data.mvtec import MVTecDataset
from src.evaluation.metrics import evaluate, gaussian_smooth
from src.evaluation.results import append_result, done_categories
from src.evaluation.visualize import save_examples


def train_one(category, args, cfg, device):
    data_root = args.data_root or cfg["data"]["root"]
    size = cfg["data"]["image_size"]
    train_ds = MVTecDataset(data_root, category, "train", size)  # ảnh [0, 1], không chuẩn hóa ImageNet
    test_ds = MVTecDataset(data_root, category, "test", size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ConvAutoencoder(latent_dim=args.latent_dim, image_size=size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            x = batch["image"].to(device)
            loss = reconstruction_loss(x, model(x), args.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [{category}] epoch {epoch}/{args.epochs}  loss {total / len(train_ds):.5f}", flush=True)
    train_time = time.time() - t0

    model.eval()
    maps, labels, masks = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    for batch in test_dl:
        maps.append(anomaly_map(model, batch["image"].to(device), args.loss).squeeze(1).cpu().numpy())
        labels.append(batch["label"].numpy())
        masks.append(batch["mask"].squeeze(1).numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.time() - t1) / len(test_ds) * 1000

    maps = gaussian_smooth(np.concatenate(maps), sigma=args.sigma)
    scores = maps.reshape(len(maps), -1).max(axis=1)
    result = evaluate(np.concatenate(labels), scores, np.concatenate(masks), maps)

    ckpt = ROOT / "results" / "checkpoints" / "ae" / f"{category}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    if not args.no_figures:
        save_examples(test_ds, maps, ROOT / "results" / "figures" / "ae" / category)

    n_params = sum(p.numel() for p in model.parameters())
    return result, train_time, infer_ms, n_params * 4 / 2 ** 20


def main():
    parser = argparse.ArgumentParser(description="Autoencoder cho MVTec AD")
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại trong configs/common.yaml")
    parser.add_argument("--data-root", help="thư mục MVTecAD (mặc định theo configs/common.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--loss", choices=["mse", "ssim"], default="mse")
    parser.add_argument("--sigma", type=float, default=4, help="làm mượt anomaly map trước khi lấy max")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--csv", default=None, help="mặc định: results_csv trong cấu hình")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="chạy lại cả loại đã có kết quả")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.csv or cfg["results_csv"]
    note = (f"loss={args.loss}; latent={args.latent_dim}; epochs={args.epochs}; lr={args.lr}; "
            f"batch={args.batch_size}; sigma={args.sigma}; size={cfg['data']['image_size']}; aug=none")
    done = set() if args.rerun else done_categories(csv_path, "ae", note)
    print(f"Thiết bị: {device}. Cấu hình: {note}")

    for category in args.categories or all_categories(cfg):
        if category in done:
            print(f"Bỏ qua {category} (đã có kết quả)")
            continue
        print(f"== {category}")
        set_seed(cfg["seed"])
        result, train_time, infer_ms, memory_mb = train_one(category, args, cfg, device)
        print(f"  image-AUROC {result['image_auroc']:.4f}  pixel-AUROC {result['pixel_auroc']:.4f}")
        append_result(csv_path, {
            "category": category, "type": category_type(cfg, category), "method": "ae",
            "implementation": "pytorch", **{k: f"{v:.4f}" for k, v in result.items()},
            "train_time_s": f"{train_time:.1f}", "infer_time_ms": f"{infer_ms:.2f}",
            "memory_mb": f"{memory_mb:.1f}", "backbone": "", "note": note,
        })


if __name__ == "__main__":
    main()
