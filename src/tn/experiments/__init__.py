"""Vòng chạy các thí nghiệm TN1–TN4. Mỗi run ghi một dòng CSV qua `Results`; run đã có key thì bỏ qua.

    tn1_tn2   TN1 train lẫn ảnh lỗi, TN2 few-shot (MVTec AD)
    tn2b      TN2b góc xoay của screw (screw.py đo và căn góc)
    tn3       TN3 đổi loại sản phẩm, thích nghi sang bo mới
    tn4       TN4 domain shift: ánh sáng (MVTec AD 2), shift nhân tạo (shifts.py)
    common    cache ảnh/đặc trưng một category, fit + đặt ngưỡng
    results   ghi CSV theo từng run

Key, tên file và cột CSV giữ đúng như bản notebook cũ để đọc tiếp được kết quả đã chạy.
TN5 (phân tích lỗi) không có vòng chạy riêng: dùng anomaly map TN2 mức 'all' (xem analysis).
"""
from src.tn.experiments.common import CategoryCache
from src.tn.experiments.results import Results
from src.tn.experiments.tn1_tn2 import run_tn1, run_tn2
from src.tn.experiments.tn2b import prepare_rotation, run_tn2b
from src.tn.experiments.tn3 import run_adapt, run_matrix
from src.tn.experiments.tn4 import run_tn4a, run_tn4b, run_tn4b_geo

__all__ = ["CategoryCache", "Results", "run_tn1", "run_tn2", "prepare_rotation", "run_tn2b", "run_matrix",
           "run_adapt", "run_tn4a", "run_tn4b", "run_tn4b_geo"]
