# Hướng dẫn sử dụng — ImageGuard Forensics

## Chạy app

```powershell
.\.venv\Scripts\python.exe -m imageguard        # GUI mặc định
.\.venv\Scripts\python.exe -m imageguard gui    # tường minh
```

Hoặc bản đóng gói: `dist/ImageGuard/ImageGuard.exe` (xem `docs/packaging.md`).

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
