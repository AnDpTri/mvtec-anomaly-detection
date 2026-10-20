"""[Dưỡng / Chung] Chạy PaDiM và PatchCore bằng Anomalib.

Cách dùng (dự kiến):
    python scripts/run_anomalib.py --model padim --categories bottle carpet screw
    python scripts/run_anomalib.py --model patchcore --categories bottle cable capsule ...

Yêu cầu:
- Cùng cấu hình (backbone, tầng đặc trưng, tỷ lệ coreset, seed) và cùng phiên bản Anomalib
  cho cả hai người (PatchCore chia 8 loại / 7 loại).
- Lấy anomaly map Anomalib xuất ra, đưa qua src/evaluate.py để tính lại AUROC,
  đối chiếu với số Anomalib tự báo.
- Ghi thời gian suy luận mỗi ảnh và bộ nhớ.

Tài liệu: docs/phuong-phap/padim.md, docs/phuong-phap/patchcore.md. Chia loại PatchCore: docs/phan-cong.md.

Ghi chú API: Anomalib 2.x dùng `anomalib.data.MVTecAD`, `anomalib.models.Padim / Patchcore`,
`anomalib.engine.Engine`. Bản 1.x dùng `MVTec` thay cho `MVTecAD`. Kiểm tra lại theo phiên bản
đã chốt trong requirements.txt.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["padim", "patchcore"], required=True)
    parser.add_argument("--categories", nargs="*", help="mặc định: cả 15 loại")
    parser.add_argument("--backbone", default=None, help="mặc định theo Anomalib")
    parser.add_argument("--coreset-ratio", type=float, default=0.1, help="chỉ dùng cho PatchCore")
    args = parser.parse_args()
    # TODO: với mỗi loại: datamodule MVTecAD(root=..., category=...) → model → Engine().fit → Engine().test
    #       → lấy anomaly map + điểm ảnh → evaluate → append_result
    raise NotImplementedError


if __name__ == "__main__":
    main()
