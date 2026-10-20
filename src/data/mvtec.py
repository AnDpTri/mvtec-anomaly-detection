"""[Dưỡng] Đọc dữ liệu MVTec AD cho PyTorch.

Mục tiêu: một DataLoader chung cho AE và PaDiM tự cài, trả về (ảnh, mask, nhãn) đúng định dạng.

Yêu cầu (xem thẻ Trello "Tải MVTec AD, viết DataLoader"):
- Ảnh xám (grid, screw, zipper) → chuyển sang 3 kênh.
- Ảnh good không có mask → tạo mask toàn 0.
- Resize mask bằng nearest rồi nhị phân hóa lại; crop ảnh thì crop mask y hệt.
- Kích thước và chuẩn hóa lấy từ configs/common.yaml.

Định dạng trả về là giao diện chung cho cả nhóm: docs/giao-dien-chung.md (mục 2).
Đặc điểm 15 loại: docs/du-lieu.md.
"""
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """Một loại sản phẩm, một split.

    split="train": chỉ ảnh train/good, nhãn 0.
    split="test":  ảnh test/<mọi kiểu>, nhãn 0 (good) hoặc 1 (lỗi), kèm mask.

    __getitem__ trả về dict:
        image: tensor 3×H×W đã chuẩn hóa
        mask:  tensor 1×H×W, giá trị 0/1
        label: 0 hoặc 1
        path:  đường dẫn ảnh gốc
        defect_type: tên kiểu lỗi ("good" nếu tốt)
    """

    def __init__(self, root, category, split, image_size, mean, std):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
