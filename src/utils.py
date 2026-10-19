"""Tiện ích dùng chung: đọc cấu hình, cố định seed.

Mục tiêu: mọi phương pháp đọc cùng một file cấu hình và cùng seed, để kết quả lặp lại được.
"""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs" / "common.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_categories(cfg):
    return cfg["categories"]["texture"] + cfg["categories"]["object"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
