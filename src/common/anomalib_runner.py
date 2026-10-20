"""[Chung] Tiện ích TÙY CHỌN: chạy một mô hình Anomalib trên nhiều loại MVTec AD.

Mục tiêu: nếu PaDiM (src/padim) hoặc PatchCore (src/patchcore) chọn triển khai bằng Anomalib thì dùng
chung quy trình này, khỏi viết lặp code. Không bắt buộc: phương pháp tự cài thì không cần file này.

Quy trình cho mỗi loại:
    datamodule MVTecAD(root, category) → model → Engine().fit → Engine().test
    → lấy anomaly map + điểm ảnh → src.evaluation (AUROC) → ghi results/results.csv

Ghi chú API: Anomalib 2.x dùng `anomalib.data.MVTecAD`, `anomalib.engine.Engine`;
bản 1.x dùng `MVTec`. Kiểm tra lại theo phiên bản đã chốt trong requirements.txt.
"""


def run(model, method_name, categories, note=""):
    """Chạy `model` (đối tượng mô hình Anomalib đã khởi tạo) trên từng loại trong `categories`.

    method_name: "padim" hoặc "patchcore" (cột `method` trong CSV).
    note: cấu hình khác mặc định, ví dụ "coreset=0.01" (cột `note`).
    """
    # TODO: load_config + set_seed → với mỗi loại: datamodule → Engine().fit/test
    #       → anomaly maps, điểm ảnh → src.evaluation.metrics → src.evaluation.results.append_result
    raise NotImplementedError
