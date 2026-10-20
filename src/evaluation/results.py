"""[Dưỡng] Ghi kết quả vào file CSV chung results/results.csv.

Mỗi dòng là một lần chạy (loại, phương pháp, cấu hình). Cột theo docs/giao-dien-chung.md (mục 4).
Cách triển khai là tự do, nên cột `implementation`, `backbone`, `note` phải ghi đủ cấu hình đã dùng.
"""

COLUMNS = [
    "category", "type", "method", "implementation", "image_auroc", "pixel_auroc",
    "train_time_s", "infer_time_ms", "memory_mb", "backbone", "note",
]


def append_result(csv_path, row):
    """Ghi thêm một dòng (dict có các khóa trong COLUMNS); tạo file kèm dòng tiêu đề nếu chưa có."""
    raise NotImplementedError


def done_categories(csv_path, method):
    """Các loại đã có kết quả của `method`, để chạy lại thì bỏ qua (Colab hay bị ngắt)."""
    raise NotImplementedError
