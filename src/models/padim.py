"""[Dưỡng] PaDiM tự cài (để hiểu và đối chiếu với bản Anomalib).

Mục tiêu: lập "bảng tiêu chuẩn" cho từng vị trí trên ảnh từ ảnh tốt, rồi đo mỗi mảnh
của ảnh mới lệch khỏi tiêu chuẩn của đúng vị trí đó bao xa.

Các bước (Defard et al., 2020; thẻ Trello "Đọc bài PaDiM, tự cài bản PaDiM đơn giản"):
1. Trích đặc trưng layer1–3 của ResNet18 pretrained, phóng về cùng kích thước, ghép lại.
2. Chọn ngẫu nhiên d chiều (ví dụ 100) để giảm chi phí.
3. Tại mỗi vị trí (i, j): tính trung bình μ và hiệp phương sai Σ + εI từ mọi ảnh tốt.
4. Ảnh test: khoảng cách Mahalanobis tại từng vị trí → upsample + Gaussian blur → anomaly map.

Tài liệu: docs/phuong-phap/padim.md. Đầu ra đưa vào evaluate.py: docs/giao-dien-chung.md (mục 3).
"""


class PaDiM:
    def __init__(self, backbone="resnet18", layers=("layer1", "layer2", "layer3"), d=100, eps=0.01, seed=42):
        raise NotImplementedError

    def fit(self, train_loader):
        """Tính μ và Σ⁻¹ cho từng vị trí từ ảnh tốt. Không huấn luyện mạng."""
        raise NotImplementedError

    def predict(self, images):
        """Trả về (anomaly_map B×1×H×W, image_score B)."""
        raise NotImplementedError
