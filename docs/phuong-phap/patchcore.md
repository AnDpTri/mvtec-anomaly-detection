# PatchCore · Phụ trách: cả hai (An 8 loại, Dưỡng 7 loại)

Module: `src/patchcore/` · `run.py` (dùng Anomalib hoặc tự cài)
Bài gốc: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.

## Mục tiêu
So mỗi mảnh của ảnh mới với **mảnh tốt giống nó nhất** trong một kho chứa đặc trưng của mọi mảnh tốt.

## Lý do
Khắc phục điểm yếu của PaDiM: **bỏ ràng buộc vị trí**. Mảnh tốt ở bất kỳ đâu trong ảnh train đều làm
"chuẩn" được, nên vật xoay thế nào cũng có mảnh tương ứng để so. Và giữ lại **mẫu thật** thay vì ép vào
phân phối Gauss.

## Cách hoạt động
```
Huấn luyện (chỉ ảnh tốt, không cập nhật trọng số mạng):
  ảnh → CNN pretrained (bài gốc: WideResNet50) → đặc trưng tầng giữa (bài gốc: layer2, layer3)
      → gộp đặc trưng các mảnh lân cận → bỏ hết vào memory bank (không quan tâm vị trí)
  coreset subsampling (bài gốc: giữ 1–10%): chọn sao cho vẫn phủ đều không gian đặc trưng (greedy)
Kiểm tra:
  mỗi mảnh x: tìm mảnh gần nhất m* trong memory bank → điểm = ‖x − m*‖
  → phóng về 256×256 → làm mượt → anomaly map; điểm ảnh = điểm của mảnh bất thường nhất
```
- **Tầng giữa:** đủ chi tiết để thấy vết xước nhỏ, nhưng chưa quá thiên về các lớp ImageNet.
- **Coreset:** memory bank đầy đủ có thể lên hàng trăm nghìn vector; coreset giảm bộ nhớ và thời gian tìm
  kiếm mà vẫn giữ độ chính xác.

## Ý nghĩa
Đạt độ chính xác cao nhất trong ba phương pháp (image-AUROC khoảng 99% trên MVTec AD theo bài gốc), và cho
thấy **"ghi nhớ mẫu thật" đôi khi tốt hơn "tóm tắt bằng công thức"**.

## Điểm yếu và đánh đổi (đưa vào phân tích)
- Tốn RAM và chậm hơn PaDiM khi memory bank lớn → thí nghiệm phụ: coreset 1% vs 10%.
- Lỗi logic (thiếu, lệch linh kiện ở transistor; đổi chỗ dây ở cable): từng mảnh nhìn vẫn bình thường.

## Phối hợp hai người
Hai người chia loại nên phải **dùng chung một cấu hình với nhau** (cùng cách triển khai, backbone, tầng,
tỷ lệ coreset, seed); cấu hình đó là gì thì nhóm tự chọn và ghi vào cột `note`. Ghi chung một `results/results.csv`.

## Triển khai: tự do
Thư viện, kiến trúc, backbone, tầng, loss, siêu tham số, tăng cường dữ liệu đều **tự chọn**; chỉ cần giữ
giao thức đánh giá chung và ghi cấu hình đã dùng vào CSV (xem [giao-dien-chung.md](../giao-dien-chung.md) mục 0).
Các con số trong phần "Cách hoạt động" là **điểm xuất phát theo bài gốc**, không bắt buộc.

