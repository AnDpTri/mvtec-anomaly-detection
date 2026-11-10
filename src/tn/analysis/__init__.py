"""Bảng tổng hợp và biểu đồ cho phần phân tích TN1–TN5; mọi hình lưu vào đường dẫn được truyền vào.

    common    màu, thứ tự nhãn, vẽ chồng ảnh
    tn1_tn2   đường cong train lẫn ảnh lỗi / few-shot
    tn2b      góc xoay của screw
    tn3       ma trận nguồn -> đích, thích nghi
    tn4       ánh sáng và shift nhân tạo
    tn5       ca thất bại, ví dụ theo loại lỗi

Notebook gọi qua `from src.tn import analysis as A`, nên các hàm công khai được nhập lại ở đây.
"""
from src.tn.analysis.common import COLORS, COND_ORDER, SHOT_ORDER, mask_overlay, overlay
from src.tn.analysis.tn1_tn2 import plot_curves, shots_to_x
from src.tn.analysis.tn2b import plot_alignment_check, plot_rotation, plot_score_vs_angle, rotation_correlation
from src.tn.analysis.tn3 import diagonal_summary, plot_adapt, plot_matrices
from src.tn.analysis.tn4 import merge_geo, plot_conditions, plot_same_scene, plot_score_shift, plot_shifts, worst_case
from src.tn.analysis.tn5 import draw_grid, failure_cases, load_maps

__all__ = ["COLORS", "COND_ORDER", "SHOT_ORDER", "mask_overlay", "overlay", "plot_curves", "shots_to_x",
           "plot_alignment_check", "plot_rotation", "plot_score_vs_angle", "rotation_correlation",
           "diagonal_summary", "plot_adapt", "plot_matrices", "merge_geo", "plot_conditions", "plot_same_scene",
           "plot_score_shift", "plot_shifts", "worst_case", "draw_grid", "failure_cases", "load_maps"]
