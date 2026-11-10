# Phân công theo module

**Mục tiêu:** mỗi module có một người chịu trách nhiệm chính, để hai người làm song song
trong phần của mình mà không sửa đè lên nhau. Kế hoạch theo ngày và checklist: board Trello của nhóm.
Bảng dưới ghi theo kế hoạch phân công gốc của nhóm; ai làm gì trong báo cáo thì theo báo cáo.

## Thành viên
| Người | GitHub | Nhánh làm việc | Phụ trách |
|---|---|---|---|
| Đặng Hoàng An | AnDpTri | `an/dev` | `src/demo/`, README |
| Trần Văn Dưỡng | NeoFortran | `duong/dev` | `src/tn/`, `notebooks/tn/` |
| Cả hai | | qua PR vào `main` | `tools/`, `tests/`, `docs/`, báo cáo |

## Module
```
src/
├── tn/            [Dưỡng]  config, data, splits, methods, metrics, shifts, screw, results, experiments, analysis
└── demo/          [An]     app.py (Gradio), inference.py (nạp mô hình đã lưu, suy luận)
tools/             [Chung]  build_tn_models.py, make_colab_notebook.py, tn_report_figures.py, download_mvtec.py
notebooks/tn/               notebook Colab chạy TN1–TN5
goi_colab/                  notebook Colab fit mô hình cho demo (sinh bằng tools/make_colab_notebook.py)
tests/                      kiểm thử trên CPU
```

## Phụ thuộc giữa các module
```
src/tn/config ──> mọi module trong src/tn, src/demo
src/tn/data, splits, metrics, methods ──> experiments, analysis (notebooks/tn)
                                      └─> tools/build_tn_models.py ──> results/checkpoints_tn/
src/tn/methods ──> src/demo/inference.py ──> src/demo/app.py
```
`src/demo` chỉ đọc mô hình trong `results/checkpoints_tn/` và gọi lại hàm của `src/tn`, nên sửa
`src/tn/methods/` (tiền xử lý, định dạng lưu) phải báo người phụ trách demo; xem [giao-dien-chung.md](giao-dien-chung.md).
