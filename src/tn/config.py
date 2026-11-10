"""Siêu tham số chung cho mọi thí nghiệm TN (giữ đúng giá trị đã dùng để ra kết quả trong resuilts_6)."""
import gc
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]      # thư mục gốc dự án
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256
SQUARE = (IMG_SIZE, IMG_SIZE)      # (cao, rộng) mặc định của ảnh
METHODS = ("ae", "padim", "patchcore")
BLUR_SIGMA = 4                  # làm mượt anomaly map trước khi lấy max
PADIM_DIM = 100                 # số chiều PaDiM chọn ngẫu nhiên từ 256 + 512 + 1024
PADIM_IDX_SEED = 1234
PC_DIM = 1024                   # số chiều patch của PatchCore
PC_CORESET = 0.10               # tỷ lệ coreset mặc định
PC_CORESET_MIN_BANK = 20_000    # bank nhỏ hơn mức này thì giữ nguyên (few-shot)
AE_STEPS = 1500                 # số bước train AE, cố định cho mọi thiết lập
AE_BATCH = 16
AE_LR = 5e-4
VAL_FRAC = 0.15                 # tách 15% ảnh train good để đặt ngưỡng (TN3, TN4)


def check_gpu() -> None:
    """In tên GPU và thử một phép nhân; sau lỗi CUDA trên Colab phải xóa runtime rồi chạy lại."""
    print("torch", torch.__version__, "| GPU:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU (đổi runtime sang T4!)")
    if torch.cuda.is_available():
        x = torch.randn(64, 100, 100, device="cuda")
        torch.cuda.synchronize()
        print("GPU OK:", float((x @ x.transpose(1, 2)).mean()))


def require_gpu() -> None:
    """Dừng nếu không có GPU (dùng trước các vòng chạy nặng)."""
    assert DEVICE == "cuda", "Không có GPU (hết hạn mức Colab?) — dừng lại, chạy lại khi có GPU T4."


def free_gpu() -> None:
    """Giải phóng bộ nhớ Python và cache CUDA giữa các lần fit."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
