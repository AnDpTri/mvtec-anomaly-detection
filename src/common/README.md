# common · Phụ trách: Chung (An + Dưỡng)

**Mục tiêu:** những thứ mọi module đều dùng: đọc cấu hình, cố định seed, quy trình chạy Anomalib.
**Lý do:** viết một lần, dùng chung, để 3 phương pháp chạy cùng điều kiện. Sửa ở đây ảnh hưởng mọi người,
nên **phải báo người kia** trước khi đổi.

| File | Nội dung |
|---|---|
| `config.py` | `load_config()`, `all_categories(cfg)` (đủ 15 loại), `category_type(cfg, category)` |
| `seed.py` | `set_seed(seed)` cho random, NumPy, PyTorch |
| `anomalib_runner.py` | Tùy chọn: `run(model, method_name, categories)` cho phương pháp nào chọn triển khai bằng Anomalib |

Cấu hình nằm ở `configs/common.yaml`. Ý nghĩa từng khóa: [docs/giao-dien-chung.md](../../docs/giao-dien-chung.md) mục 1.
