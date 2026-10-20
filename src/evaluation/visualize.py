"""[Dưỡng] Vẽ hình kết quả: ảnh gốc | mask | heatmap chồng ảnh.

Dùng cùng một thang màu cho mọi phương pháp để hình trong báo cáo so sánh được.
Hình lưu ở results/figures/<method>/<category>/ (không đưa lên git).
"""


def save_figure(image, mask, anomaly_map, out_path, vmin=None, vmax=None):
    raise NotImplementedError
