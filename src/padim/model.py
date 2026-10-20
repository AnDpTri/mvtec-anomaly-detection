"""[Dưỡng] PaDiM tự cài (tùy chọn: để hiểu, hoặc dùng làm bản chính thay cho Anomalib).

Mục tiêu: lập "bảng tiêu chuẩn" cho từng vị trí trên ảnh từ ảnh tốt, rồi đo mỗi mảnh
của ảnh mới lệch khỏi tiêu chuẩn của đúng vị trí đó bao xa.

Các bước theo bài gốc (Defard et al., 2020); backbone, tầng, số chiều là điểm xuất phát, đổi tùy ý:
1. Trích đặc trưng nhiều tầng của CNN pretrained (bài gốc: layer1–3 ResNet18), phóng về cùng kích thước, ghép lại.
2. Giảm chiều (bài gốc: chọn ngẫu nhiên d = 100 chiều).
3. Tại mỗi vị trí (i, j): tính trung bình μ và hiệp phương sai Σ + εI từ mọi ảnh tốt.
4. Ảnh test: khoảng cách Mahalanobis tại từng vị trí → upsample + Gaussian blur → anomaly map.

Tài liệu: docs/phuong-phap/padim.md. Đầu ra đưa vào src/evaluation: docs/giao-dien-chung.md (mục 3).
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
