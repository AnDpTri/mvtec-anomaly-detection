# demo · Phụ trách: An

**Mục tiêu:** giao diện Gradio: chọn loại sản phẩm và phương pháp, tải ảnh lên → điểm bất thường,
heatmap chồng lên ảnh. Kết luận lỗi/không lỗi chỉ có khi checkpoint được hiệu chuẩn trên ảnh tốt giữ riêng.
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
Checkpoint hiện có dùng phân vị 99,5% trên tối đa 64 ảnh train/good đã tham gia fit. Ngưỡng trong mẫu này không phù hợp để phân loại ảnh mới: giao diện chỉ hiển thị điểm và heatmap. Muốn kết luận tốt/lỗi cần hiệu chuẩn trên ảnh tốt không tham gia fit và đánh giá báo động giả độc lập; công cụ tạo checkpoint hiện chưa thực hiện bước đó. Không dùng nhãn test để chọn ngưỡng.

Hướng dẫn tạo/lưu trọng số, chạy offline và phân biệt kết quả Colab với lần bổ sung local: [docs/demo.md](../../docs/demo.md).

Thẻ Trello: "[An] Làm demo Gradio".
