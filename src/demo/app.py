"""[An] Demo Gradio: tải ảnh lên → điểm bất thường + heatmap.

Yêu cầu (thẻ Trello "Làm demo Gradio"):
- Chọn loại sản phẩm và phương pháp (AE / PaDiM / PatchCore), tải ảnh lên.
- Hiển thị điểm bất thường, heatmap chồng lên ảnh, kết luận lỗi / không lỗi.
- Ngưỡng chọn riêng cho từng loại; ghi rõ cách chọn trong README.

Chạy (dự kiến, từ thư mục gốc dự án): python -m src.demo.app
"""


def predict(image, category, method):
    """Trả về (heatmap chồng ảnh, điểm bất thường, kết luận)."""
    raise NotImplementedError


def main():
    # TODO: gr.Interface(fn=predict, inputs=[ảnh, chọn loại, chọn phương pháp], outputs=[ảnh, số, chữ])
    raise NotImplementedError


if __name__ == "__main__":
    main()
