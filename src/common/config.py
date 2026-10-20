"""[Chung] Đọc cấu hình chung.

Mục tiêu: mọi module đọc cùng một file configs/common.yaml, để 3 phương pháp chạy cùng điều kiện.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "configs" / "common.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_categories(cfg):
    """Đủ 15 loại: 5 texture trước, 10 object sau."""
    return cfg["categories"]["texture"] + cfg["categories"]["object"]


def category_type(cfg, category):
    """'texture' hoặc 'object', dùng cho cột `type` trong CSV kết quả."""
    return "texture" if category in cfg["categories"]["texture"] else "object"
