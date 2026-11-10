"""[An] Demo Gradio: tải ảnh lên → điểm bất thường + heatmap.

Yêu cầu (thẻ Trello "Làm demo Gradio"):
- Chọn loại sản phẩm và phương pháp (AE / PaDiM / PatchCore), tải ảnh lên.
- Hiển thị điểm bất thường, heatmap chồng lên ảnh, kết luận lỗi / không lỗi.
- Ngưỡng chọn riêng cho từng loại; ghi rõ cách chọn trong README.

Mô hình: 9 loại của thí nghiệm TN1–TN5 (MVTec AD, MVTec AD 2, VisA), tạo bằng tools/build_inference.py.
Ngưỡng = trung bình + 3 độ lệch chuẩn của điểm trên ảnh good validation (không dùng ảnh train, không dùng ảnh test).

Chạy từ thư mục gốc dự án: python -m src.demo.app
"""

import argparse
import numpy as np
from PIL import Image
from src.demo import inference

METHOD_NAME = {"ae": "Autoencoder", "padim": "PaDiM", "patchcore": "PatchCore"}


def predict(image, category, method):
    """Trả về (heatmap chồng ảnh, điểm bất thường, kết luận)."""
    if image is None:
        raise ValueError("Chọn ảnh sản phẩm trước khi chạy.")
    img, anomaly, score, meta = inference.predict(image, category, method)
    from matplotlib import colormaps

    thr = meta["thr"][method]
    # thang màu: trung bình pixel của ảnh good validation -> 0, ngưỡng pixel -> 1 (đỏ/vàng = vượt ngưỡng pixel)
    norm = np.clip((anomaly - thr["lo"]) / max(thr["thr_pix"] - thr["lo"], 1e-8), 0, 1)
    heat = (colormaps["inferno"](norm)[..., :3] * 255).astype(np.uint8)
    overlay = Image.fromarray((0.55 * img + 0.45 * heat).astype(np.uint8))

    met = meta["metrics"][method]
    label = "CÓ LỖI" if score >= thr["thr_img"] else "KHÔNG LỖI"
    conclusion = (
        f"{label}: điểm {score:.4g} / ngưỡng {thr['thr_img']:.4g} (gấp {score / thr['thr_img']:.2f} lần).\n"
        f"Mô hình {METHOD_NAME[method]} cho {category}, trên tập test: I-AUROC {met['i_auroc']:.3f}, "
        f"báo nhầm ảnh tốt {met['img_fpr_thr']:.0%}, bắt được ảnh lỗi {met['img_tpr_thr']:.0%}."
    )
    return overlay, score, conclusion


def main():
    import gradio as gr

    parser = argparse.ArgumentParser(description="Demo phát hiện bất thường từ mô hình đã lưu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    cats = inference.categories()
    if not cats:
        raise SystemExit(f"Chưa có mô hình trong {inference.MODEL_ROOT}. Chạy tools/build_inference.py trước.")

    def run(image, category, method):
        try:
            return predict(image, category, method)
        except (ValueError, FileNotFoundError) as e:
            raise gr.Error(str(e)) from e

    with gr.Blocks(title="Phát hiện bất thường") as demo:
        gr.Markdown(
            "# Phát hiện và định vị bất thường\nChọn đúng loại sản phẩm và phương pháp. "
            "Demo chỉ suy luận từ mô hình đã lưu, không huấn luyện lại và không gửi ảnh ra ngoài."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode="RGB", label="Ảnh sản phẩm")
                category = gr.Dropdown(cats, value=cats[0][1], label="Loại sản phẩm")
                method = gr.Dropdown(
                    [(v, k) for k, v in METHOD_NAME.items()], value="patchcore", label="Phương pháp"
                )
                button = gr.Button("Phân tích ảnh", variant="primary")
            with gr.Column():
                overlay = gr.Image(type="pil", label="Vùng nghi ngờ · sáng = vượt ngưỡng pixel")
                score = gr.Number(label="Điểm bất thường · không phải xác suất", precision=6)
                conclusion = gr.Textbox(label="Kết luận", interactive=False, lines=4)
        ex = inference.examples()
        if ex:
            gr.Examples(ex, [image, category], label="Ảnh mẫu từ tập test (mỗi loại 1 tốt, 1 lỗi)")
        gr.Markdown(
            "Ngưỡng của từng loại và phương pháp = trung bình + 3 độ lệch chuẩn của điểm trên ảnh good "
            "validation (15% train/good tách riêng; MVTec AD 2 dùng validation/good có sẵn). "
            "Điểm cao hơn nghĩa là khác ảnh tốt hơn."
        )
        button.click(run, [image, category, method], [overlay, score, conclusion])
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
