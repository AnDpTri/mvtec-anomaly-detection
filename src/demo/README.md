# demo · Phụ trách: An

**Mục tiêu:** giao diện Gradio: chọn loại sản phẩm và phương pháp, tải ảnh lên → điểm bất thường,
heatmap chồng lên ảnh, kết luận lỗi/không lỗi. Dùng 9 loại của thí nghiệm TN1–TN5: MVTec AD grid, screw,
transistor; MVTec AD 2 vial, sheet_metal; VisA pcb1–pcb4.
**Lý do:** cho hội đồng thấy hệ thống chạy thật trên ảnh bất kỳ, không chỉ bảng số.
**Ý nghĩa:** buộc phải chọn **ngưỡng cụ thể** cho từng loại (AUROC thì không cần); cách chọn ngưỡng
phải ghi rõ ở đây.

| File | Nội dung |
|---|---|
| `app.py` | `predict(image, category, method)` và giao diện Gradio |
| `tn_models.py` | Nạp `results/checkpoints_tn/<loại>/` (ae.pt, padim.pt, patchcore_bank.pt, meta.json) bằng `src.tn.methods.load_model` và chấm một ảnh bằng `predict_method` |

## Cách chạy (từ thư mục gốc dự án)
```bash
python -m src.demo.app
```

## Cách chọn ngưỡng
Ngưỡng ảnh = trung bình + 3 độ lệch chuẩn của điểm trên ảnh good validation, chọn riêng cho từng loại và phương pháp. Validation là 15% train/good tách ra trước khi fit (seed 0); MVTec AD 2 dùng thư mục validation/good có sẵn. Không dùng ảnh train đã fit, không dùng nhãn test. Ngưỡng pixel tính cùng cách trên từng điểm ảnh, dùng làm đầu thang màu của heatmap.

Ngưỡng này chặt: tỷ lệ báo nhầm ảnh tốt trên tập test chỉ 0–14%, đổi lại nhiều ảnh lỗi nhẹ không vượt ngưỡng (xem `img_tpr_thr` trong meta.json).

Tạo mô hình: `python tools/build_tn_models.py` (máy có GPU) hoặc notebook `goi_colab/build_tn_models_colab.ipynb` trên Colab, rồi chép thư mục `checkpoints_tn/` vào `results/`.

Thẻ Trello: "[An] Làm demo Gradio".
