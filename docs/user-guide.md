# Hướng dẫn sử dụng — ImageGuard Forensics

## Chạy app

```powershell
.\.venv\Scripts\python.exe -m imageguard        # GUI mặc định
.\.venv\Scripts\python.exe -m imageguard gui    # tường minh
```

Hoặc bản đóng gói: `dist/ImageGuard/ImageGuard.exe` (xem `docs/packaging.md`).

## Phân tích qua CLI (báo cáo JSON)

```bash
python -m imageguard analyze <file> [--methods keypoint,block,ela] [--no-persist]
```

Lệnh in ra stdout cùng cấu trúc JSON mà ứng dụng lưu khi phân tích (báo cáo dấu hiệu).
Nếu thoát mã khác 0, đó là `ErrorResult` (ví dụ file không đọc được), không phải một verdict
forensics trên ảnh.

## Quy trình phân tích

1. **Mở ảnh** — kéo-thả vào canvas hoặc nút "Mở ảnh…". Định dạng: JPEG,
   PNG, BMP, TIFF, WebP (≤100MB, ≤80MP, cạnh ≥64px — ảnh quá lớn/corrupt
   báo lỗi thân thiện, không crash).
2. **Xem preview + metadata** — kích thước, định dạng, dung lượng, EXIF.
3. **Chọn phương pháp** —
   - *Copy-Move keypoint (SIFT/ORB)*: tìm vùng nhân đôi trong cùng ảnh.
   - *Copy-Move block (DCT)*: baseline kinh điển, chậm hơn nhưng độc lập.
   - *ELA*: chỉ có ý nghĩa với JPEG; PNG/BMP sẽ hiện "Không áp dụng".
   - *Noise inconsistency / Frequency / Splicing cues* (L2): bằng chứng
     khám phá — KHÔNG BAO GIỜ tự kết luận.
4. **Phân tích** — thanh progress theo stage; nút "Hủy" dừng mềm giữa
   stages (cooperative cancel).
5. **Đọc kết quả** —
   - Banner verdict 3 trạng thái chính: *Có dấu hiệu nghi vấn* (cam),
     *Không thấy dấu hiệu* (xám), *Không đủ bằng chứng* (vàng trung tính).
   - Cards từng method: score + định nghĩa score + limitations.
   - Dropdown overlay: mask đỏ / heatmap trên ảnh gốc (zoom: lăn chuột,
     pan: kéo).
   - Thanh disclaimer cố định: kết quả là *dấu hiệu*, không phải kết luận.
6. **Lịch sử** — tab "Lịch sử" liệt kê phân tích đã lưu (SQLite); xóa sẽ
   xóa cả artifacts. Tab "Thống kê" tổng hợp phiên làm việc.

## Tab Tăng cường (xem, không phân tích)

Tab "Tăng cường" chạy ba operator cổ điển — *Unsharp mask*, *CLAHE*,
*Non-local means* — để xem thử hiệu ứng trên ảnh. Tab này **không phân
tích**, không gọi `AnalysisService`, không ghi lịch sử hay database, và
không kết luận điều gì về ảnh.

- **Banner cố định**: "Ảnh đã tăng cường bằng thuật toán hoặc mô hình AI.
  Đây là bản xem, không phải bằng chứng. Phân tích dấu hiệu chỉ gắn với
  file gốc." Nghĩa là ảnh tăng cường chỉ để xem; dấu hiệu forensics chỉ
  được gắn với file gốc, không phải với bản tăng cường.
- **Thanh trộn**: 0 = ảnh gốc, 100 = ảnh tăng cường. Thanh chỉ đổi hiển thị
  (pixmap), không ghi pixel và không chạy lại operator.
- **Xuất**: chọn đích PNG, hệ thống ghi thêm một file `.json` kế bên
  (cùng tên) ghi `parent_sha256`, `derived_sha256`, `method`, `params`,
  `model_sha256` (null) và `created_utc`. File gốc không bao giờ bị ghi đè;
  đích trùng file gốc bị từ chối, đích đã tồn tại phải xác nhận ghi đè.
- Tab này không khôi phục dấu vết camera và không phục hồi ảnh về trạng
  thái gốc.

## Đọc verdict đúng cách

- *Có dấu hiệu nghi vấn* ≠ "ảnh giả" — nghĩa là ≥1 detector tìm thấy
  bằng chứng copy-move có cấu trúc; nên kiểm tra mask/overlay thủ công.
- *Không thấy dấu hiệu* ≠ "ảnh thật" — method có thể miss (resize mạnh,
  vùng nhỏ, texture kém).
- *Không đủ bằng chứng* = method không kết luận được (ít texture, lỗi
  một phần, tín hiệu mâu thuẫn) — đây là câu trả lời hợp lệ, không phải lỗi.

## Evaluation qua CLI

```bash
python -m imageguard eval --dataset synth --root data/synth_eval \
    --split dev --methods keypoint,block
```

Report JSON/CSV vào `reports/` kèm experiment_id, seed, config hash.

## Ingest ảnh từ thư mục

Quét một lần hoặc theo dõi liên tục một thư mục ảnh tĩnh:

```bash
python -m imageguard ingest once <dir>
python -m imageguard ingest watch <dir>
```

Thêm `--config <file>` sau `<dir>` để dùng cấu hình riêng. Cấu hình và log
mặc định nằm trong `data/`, thư mục đã được gitignore. `enhance_sibling` mặc
định tắt. Khi bật, ingest ghi thêm `<tên>.view.png` để xem, không phân tích
file đó và không đổi file gốc.
