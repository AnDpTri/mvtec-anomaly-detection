# demo · Phụ trách: An

**Mục tiêu:** giao diện Gradio: chọn loại sản phẩm và phương pháp, tải ảnh lên → điểm bất thường,
heatmap chồng lên ảnh, kết luận lỗi / không lỗi.
**Lý do:** cho hội đồng thấy hệ thống chạy thật trên ảnh bất kỳ, không chỉ bảng số.
**Ý nghĩa:** buộc phải chọn **ngưỡng cụ thể** cho từng loại (AUROC thì không cần); cách chọn ngưỡng
phải ghi rõ ở đây.

| File | Nội dung |
|---|---|
| `app.py` | `predict(image, category, method)` và giao diện Gradio |

## Cách chạy (dự kiến, từ thư mục gốc dự án)
```bash
python -m src.demo.app
```

## Cách chọn ngưỡng
TODO: ghi khi làm (ví dụ: ngưỡng tối đa F1 trên tập validation tách từ ảnh tốt + một ít ảnh lỗi).

Thẻ Trello: "[An] Làm demo Gradio".
