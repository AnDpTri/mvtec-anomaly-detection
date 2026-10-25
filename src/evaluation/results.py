"""[Dưỡng] Ghi kết quả vào file CSV chung results/results.csv.

Mỗi dòng là một lần chạy (loại, phương pháp, cấu hình). Cột theo docs/giao-dien-chung.md (mục 4).
Cách triển khai là tự do, nên cột `implementation`, `backbone`, `note` phải ghi đủ cấu hình đã dùng.
"""
import csv
from pathlib import Path

from src.common.config import ROOT

COLUMNS = [
    "category", "type", "method", "implementation", "image_auroc", "pixel_auroc",
    "train_time_s", "infer_time_ms", "memory_mb", "backbone", "note",
]


def _path(csv_path):
    p = Path(csv_path)
    return p if p.is_absolute() else ROOT / p


def append_result(csv_path, row):
    """Ghi thêm một dòng (dict có các khóa trong COLUMNS); tạo file kèm dòng tiêu đề nếu chưa có."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"Cột không có trong giao diện chung: {sorted(unknown)}")
    p = _path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


def done_categories(csv_path, method, note=None):
    """Các loại đã có kết quả của `method` (và cùng `note` nếu đưa vào), để chạy lại thì bỏ qua."""
    p = _path(csv_path)
    if not p.exists():
        return set()
    with open(p, newline="", encoding="utf-8") as f:
        return {r["category"] for r in csv.DictReader(f)
                if r["method"] == method and (note is None or r["note"] == note)}
