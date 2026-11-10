# Quy trình làm việc

**Mục tiêu:** hai người viết code song song mà không đè lên nhau, và mỗi người hiểu phần của người kia.

## 1. Ai làm phần nào
Xem [docs/phan-cong.md](docs/phan-cong.md). Mỗi module trong `src/` có một người chịu trách nhiệm chính
(ghi trong README của module và trong `.github/CODEOWNERS`). **Làm trong module của mình.** Muốn sửa module
của người kia thì báo trước hoặc mở PR để người đó duyệt. Phần dùng chung (`tools/`, `tests/`, `docs/`) sửa
xong phải báo người kia.

## 2. Ranh giới giữa hai người: giao diện chung
Hai người chỉ "gặp nhau" ở một số điểm cố định, ghi trong [docs/giao-dien-chung.md](docs/giao-dien-chung.md):
hàm đọc dữ liệu, giao diện fit/predict, định dạng anomaly map, file mô hình đã lưu, các cột CSV kết quả.
Giữ đúng giao diện này thì mỗi người làm phần mình mà không cần chờ người kia.
**Đổi giao diện chung phải báo người kia trước.**

## 3. Nhánh
Không làm trực tiếp trên `main`. **Mỗi người một nhánh làm việc riêng:**

| Nhánh | Của ai |
|---|---|
| `an/dev` | An |
| `duong/dev` | Dưỡng |
| `main` | Bản chung đã duyệt; chỉ nhận code qua Pull Request |

Lần đầu:
```bash
git fetch origin
git switch an/dev            # Dưỡng: git switch duong/dev
```
Mỗi ngày trước khi làm, lấy phần đã gộp của người kia về nhánh mình:
```bash
git switch an/dev
git pull origin main
```
Việc lớn hoặc muốn thử nghiệm riêng thì có thể tách thêm nhánh con từ nhánh của mình, ví dụ
`an/numpy-ae`, rồi gộp lại vào `an/dev`.

## 4. Commit
Viết tiếng Việt, ngắn, nói rõ làm gì: `Thêm encoder cho conv AE`, `Sửa resize mask bằng nearest`.
Commit và đẩy nhánh của mình thường xuyên (`git push`), để người kia thấy tiến độ và không mất code.

## 5. Pull Request
Xong một phần dùng được (ví dụ một thí nghiệm chạy được, demo nạp được mô hình mới):
1. `git push` nhánh của mình.
2. Mở PR **từ `an/dev` (hoặc `duong/dev`) vào `main`**, điền mẫu (mục tiêu, lý do, cách kiểm tra, thẻ Trello).
3. Người kia đọc và duyệt. Đọc PR cũng là cách hiểu phần của nhau để trả lời hội đồng.
4. Gộp xong: người kia chạy `git pull origin main` trên nhánh của mình để lấy phần mới; tích checklist trên Trello.

PR nhỏ, gộp thường xuyên (vài ngày một lần) thì ít xung đột hơn để dồn cả tuần.

## 6. Những thứ không đưa lên git
Dữ liệu (`datasets/`), checkpoint (`*.pt`, `*.ckpt`), memory bank, ảnh kết quả. Chỉ đưa file CSV số liệu
trong `results/`. `.gitignore` đã chặn sẵn.

## 7. Trước khi đẩy code và trước khi chạy lấy số cho báo cáo
- Chạy kiểm thử: `python -m unittest discover -s tests -v` (CPU, không cần dữ liệu; phần demo tự bỏ qua nếu chưa có mô hình).
- Siêu tham số chung chỉ đổi ở `src/tn/config.py`; đổi thì phải chạy lại mọi thí nghiệm bị ảnh hưởng và sinh lại notebook Colab (`python tools/make_colab_notebook.py`).
- Cùng phiên bản thư viện (xem `requirements.txt`). Không đổi định dạng mô hình lưu trong `results/checkpoints_tn/` khi chưa fit lại cả 9 loại.
