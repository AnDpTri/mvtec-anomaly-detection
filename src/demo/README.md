# demo · Phụ trách: An

**Mục tiêu:** giao diện Gradio: chọn loại sản phẩm và phương pháp, tải ảnh lên → điểm bất thường,
heatmap chồng lên ảnh, kết luận lỗi / không lỗi.
**Lý do:** cho hội đồng thấy hệ thống chạy thật trên ảnh bất kỳ, không chỉ bảng số.
**Ý nghĩa:** buộc phải chọn **ngưỡng cụ thể** cho từng loại (AUROC thì không cần); cách chọn ngưỡng
phải ghi rõ ở đây.

| File | Nội dung |
|---|---|
| `app.py` | `predict(image, category, method)` và giao diện Gradio |

## Cách chạy (từ thư mục gốc dự án)
```bash
python -m src.demo.app
```

## Cách chọn ngưỡng
Phân vị 99,5% trên tối đa 64 ảnh train/good đã tham gia fit, riêng cho mỗi loại/phương pháp. Đây là ngưỡng tham khảo trong mẫu, không bảo đảm FPR 0,5% trên ảnh mới. Không dùng nhãn test để chọn ngưỡng. Nếu checkpoint chưa có ngưỡng thì chỉ hiển thị điểm và heatmap.

Hướng dẫn tạo/lưu trọng số, chạy offline và phân biệt kết quả Colab với lần bổ sung local: [docs/demo.md](../../docs/demo.md).

Thẻ Trello: "[An] Làm demo Gradio".
