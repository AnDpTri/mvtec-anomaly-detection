# Quy trình làm việc

**Mục tiêu:** hai người viết code song song mà không đè lên nhau, và mỗi người hiểu phần của người kia.

## 1. Ai làm phần nào
Xem [docs/phan-cong.md](docs/phan-cong.md). Mỗi module trong `src/` có một người chịu trách nhiệm chính
(ghi trong README của module và trong `.github/CODEOWNERS`). **Làm trong module của mình.** Muốn sửa module
của người kia thì báo trước hoặc mở PR để người đó duyệt. Module `[Chung]` (`common/`, `patchcore/`) sửa
xong phải báo người kia.

## 2. Ranh giới giữa hai người: giao diện chung
Hai người chỉ "gặp nhau" ở một số điểm cố định, ghi trong [docs/giao-dien-chung.md](docs/giao-dien-chung.md):
dữ liệu trả về từ DataLoader, định dạng anomaly map, điểm ảnh, các cột CSV kết quả.
Giữ đúng giao diện này thì mỗi người làm phần mình mà không cần chờ người kia.
**Đổi giao diện chung phải báo người kia trước.**

## 3. Nhánh
Không làm trực tiếp trên `main`. Mỗi việc một nhánh, đặt tên theo người và việc:
```
an/numpy-ae          an/conv-ae          an/demo-gradio
duong/dataloader     duong/evaluate      duong/padim
chung/patchcore      docs/...
```
```bash
git switch main && git pull
git switch -c an/conv-ae
```

## 4. Commit
Viết tiếng Việt, ngắn, nói rõ làm gì: `Thêm encoder cho conv AE`, `Sửa resize mask bằng nearest`.

## 5. Pull Request
1. Đẩy nhánh lên: `git push -u origin an/conv-ae`
2. Mở PR vào `main`, điền mẫu (mục tiêu, lý do, cách kiểm tra, thẻ Trello).
3. Người kia đọc và duyệt. Đọc PR cũng là cách hiểu phần của nhau để trả lời hội đồng.
4. Merge xong thì tích checklist trên Trello.

## 6. Những thứ không đưa lên git
Dữ liệu (`datasets/`), checkpoint (`*.pt`, `*.ckpt`), memory bank, ảnh kết quả. Chỉ đưa file CSV số liệu
trong `results/`. `.gitignore` đã chặn sẵn.

## 7. Trước khi chạy lấy số cho báo cáo
- Chạy thử 3 loại (bottle, carpet, screw) để kiểm tra code, rồi mới chạy **đủ 15 loại**.
- Dùng đúng `configs/common.yaml`, cùng phiên bản thư viện (xem `requirements.txt`).
- Ghi kết quả vào `results/results.csv` theo đúng các cột trong giao diện chung.
