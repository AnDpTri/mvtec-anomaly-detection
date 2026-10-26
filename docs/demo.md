# Chạy demo từ mô hình đã lưu

Demo đọc checkpoint, không fit lại khi tải ảnh. Chạy từ thư mục gốc:

```powershell
python -m src.demo.app --host 127.0.0.1 --port 7864
```

Mở `http://127.0.0.1:7864`, chọn loại sản phẩm, phương pháp và ảnh RGB. Điểm là độ bất thường, không phải xác suất lỗi. Heatmap dùng thang cố định từ mẫu train tốt; màu giữa các phương pháp không biểu thị cùng một đơn vị đo.

## Tạo hoặc bổ sung mô hình

```powershell
python -m tools.build_demo_models --data-root datasets/MVTecAD --ae-source PATH_TO_TRAINED_AE
```

`--ae-source` chứa 15 trọng số AE đã train (`bottle.pt`, ...). Lệnh sao chép vào thư mục của dự án trước khi bổ sung metadata, không ghi lên nguồn. PaDiM và PatchCore fit trên `train/good`, tải backbone ImageNet lần đầu. Có thể giới hạn `--methods padim patchcore` hoặc `--categories bottle`. Batch mặc định 4 phù hợp GPU 4 GB; không đồng nhất với batch 16 của lần chạy Colab gốc.

Checkpoint nằm ở `results/checkpoints/{ae,padim,patchcore}/CATEGORY.pt`, kèm JSON cấu hình. Chỉ bỏ qua một loại khi có cả hai tệp; có thể tiếp tục sau gián đoạn. `MVTEC_CHECKPOINT_ROOT` chỉ đổi nơi demo đọc checkpoint. Sau khi có đủ trọng số, demo suy luận offline và mặc định chạy CPU để tránh thiếu VRAM.

Định dạng lưu gồm trọng số, method, category và thông số chuẩn hóa; dùng `torch.load(weights_only=True)`. Với PaDiM, trạng thái Gauss nằm trong state dict; với PatchCore, memory bank cũng nằm trong state dict. Nạp backbone với `pre_trained=False` khi reload để không tải lại dữ liệu qua mạng. Chỉ mở checkpoint từ nguồn tin cậy.

## Ngưỡng

Mỗi loại/method có ngưỡng riêng: phân vị 99,5% điểm trên tối đa 64 ảnh `train/good`, chọn đều theo danh sách. Các ảnh đã tham gia fit nên đây là ngưỡng tham khảo trong mẫu, **không bảo đảm FPR 0,5% trên ảnh mới**. Không dùng nhãn test để đặt ngưỡng. AUROC từ bộ đánh giá không phụ thuộc ngưỡng demo. Khi chưa có calibration, demo chỉ hiển thị điểm, không kết luận tốt/lỗi.

## Kết quả và lưu trữ

- `results/results.csv`: 45 dòng Colab ngày 25/10/2026, giữ nguyên.
- `results/demo_models/results.csv`: kết quả chạy bổ sung tại máy local; không gộp thời gian với Colab.
- `docs/results_provenance.json`: hash và trung bình tính trực tiếp từ CSV gốc.
- Checkpoint và dataset không đẩy GitHub theo quy định repo. Cần sao lưu thư mục checkpoint ra ổ lưu trữ riêng nếu chuyển máy.

Kiểm thử: `python -m unittest discover -s tests -v`. Sinh lại bảng báo cáo: `python -m tools.report_results`. Báo cáo chính biên dịch XeLaTeX + Biber từ `bao_cao/main.tex`.

## Kiểm tra hoàn tất ngày 26/10/2026

Đã lưu đủ **45 checkpoint**, và nạp lại/suy luận thật trên một ảnh tốt cho từng mô hình với socket mạng bị chặn. Cả 45 trả bản đồ 256×256 và điểm hữu hạn. [Manifest](demo_validation.json) ghi SHA-256, kích thước và phiên bản môi trường; phép smoke test này không thay thế đánh giá AUROC toàn bộ test. CSV bổ sung đủ 30 dòng riêng cho PaDiM/PatchCore, CSV Colab 45 dòng không đổi byte. API Gradio đã nhận ảnh và trả ảnh heatmap/điểm/nhận xét cho cả ba phương pháp trên bottle.

Chạy lại kiểm tra bằng `python -m tools.verify_demo_models --data-root PATH_TO_MVTEC`. Checkpoint hiện lưu tại worktree riêng ở `D:\Codex-Branches\2026-10-26\deep-learning\results\checkpoints`. **Giữ thư mục này khi dọn dự án**: GitHub chứa mã và manifest, không chứa trọng số. File `bao_cao/portable.tex` dùng backend BibTeX cho Tectonic; `main.tex` vẫn dùng Biber. Bản PDF đầy đủ đã được kiểm tra bố cục từng trang.
