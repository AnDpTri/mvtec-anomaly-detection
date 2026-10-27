"""[An] Demo Gradio: tải ảnh lên → điểm bất thường + heatmap.

Yêu cầu (thẻ Trello "Làm demo Gradio"):
- Chọn loại sản phẩm và phương pháp (AE / PaDiM / PatchCore), tải ảnh lên.
- Hiển thị điểm bất thường, heatmap chồng lên ảnh, kết luận lỗi / không lỗi.
- Ngưỡng chọn riêng cho từng loại; ghi rõ cách chọn trong README.

Chạy từ thư mục gốc dự án: python -m src.demo.app
"""

import argparse
import numpy as np
from PIL import Image
from src.common.config import all_categories, load_config
from src.demo.artifacts import predict_file


def predict(image, category, method):
    """Trả về (heatmap chồng ảnh, điểm bất thường, kết luận)."""
    if image is None:
        raise ValueError("Chọn ảnh sản phẩm trước khi chạy.")
    anomaly, score, config = predict_file(image, category, method)
    from matplotlib import colormaps

    image = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )
    image = image.resize(
        (anomaly.shape[1], anomaly.shape[0]), Image.Resampling.BILINEAR
    )
    scale = max(float(config.get("map_scale", 0.05)), 1e-8)
    heat = (colormaps["inferno"](np.clip(anomaly / scale, 0, 1))[..., :3] * 255).astype(
        np.uint8
    )
    overlay = Image.fromarray((0.55 * np.asarray(image) + 0.45 * heat).astype(np.uint8))
    threshold = config.get("threshold")
    if threshold is None or config.get("calibration") != "held_out_normal_quantile":
        conclusion = (
            "Chưa có ngưỡng được hiệu chuẩn trên tập ảnh tốt độc lập; "
            "chỉ hiển thị điểm và heatmap, chưa kết luận tốt/lỗi."
        )
    else:
        label = (
            "Có dấu hiệu bất thường"
            if score > threshold
            else "Chưa vượt ngưỡng bất thường"
        )
        conclusion = f"{label}. Ngưỡng hiệu chuẩn trên ảnh tốt giữ riêng: {threshold:.6g}."
    return overlay, score, conclusion


def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description="Demo MVTec AD từ checkpoint đã lưu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    def run(image, category, method):
        try:
            return predict(image, category, method)
        except (ValueError, FileNotFoundError) as e:
            raise gr.Error(str(e)) from e

    with gr.Blocks(title="MVTec AD · Phát hiện bất thường") as demo:
        gr.Markdown(
            "# Phát hiện và định vị bất thường\nChọn đúng loại sản phẩm và mô hình đã lưu. Demo không huấn luyện lại và không gửi ảnh tới dịch vụ ngoài."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode="RGB", label="Ảnh sản phẩm")
                category = gr.Dropdown(
                    all_categories(load_config()), value="bottle", label="Loại sản phẩm"
                )
                method = gr.Dropdown(
                    [
                        ("Autoencoder", "ae"),
                        ("PaDiM", "padim"),
                        ("PatchCore", "patchcore"),
                    ],
                    value="patchcore",
                    label="Phương pháp",
                )
                button = gr.Button("Phân tích ảnh", variant="primary")
            with gr.Column():
                overlay = gr.Image(
                    type="pil",
                    label="Vùng nghi ngờ · thang màu cố định theo mẫu train tốt",
                )
                score = gr.Number(
                    label="Điểm bất thường · không phải xác suất", precision=6
                )
                conclusion = gr.Textbox(label="Nhận xét", interactive=False, lines=4)
        gr.Markdown(
            "Các checkpoint hiện có chưa có ngưỡng được hiệu chuẩn trên tập độc lập, nên demo chỉ hiển thị điểm và heatmap. Điểm cao hơn nghĩa là khác thường hơn; kết quả AUROC không phụ thuộc việc chọn ngưỡng."
        )
        button.click(run, [image, category, method], [overlay, score, conclusion])
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
