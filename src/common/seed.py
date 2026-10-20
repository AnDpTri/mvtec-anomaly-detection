"""[Chung] Cố định seed để kết quả lặp lại được. Gọi set_seed(cfg["seed"]) ở đầu mỗi lần chạy."""
import random

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
