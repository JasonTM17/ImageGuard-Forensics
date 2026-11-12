# Hướng dẫn sử dụng — ImageGuard Forensics

## Chạy app

**Khuyến nghị (source):**

```powershell
.\run_app.bat
# hoặc
.\.venv\Scripts\python.exe -m imageguard        # GUI mặc định
.\.venv\Scripts\python.exe -m imageguard gui    # tường minh
```

**Bản đóng gói (`dist/ImageGuard/ImageGuard.exe`):** chỉ thử sau khi đọc
[`docs/packaging.md`](packaging.md). Tài liệu đó ghi một lần QA: PyInstaller
build thành công nhưng executable lỗi import `QtWidgets` (DLL); **chưa có**
xác nhận bản `.exe` chạy trọn luồng phân tích. Đừng coi file exe là đường
chạy chính cho demo đồ án trừ khi bạn tự verify lại trên máy mình.

## Phân tích qua CLI (báo cáo JSON)

```powershell
.\.venv\Scripts\python.exe -m imageguard analyze <file> `
  [--methods keypoint,block,ela] [--no-persist]
```

Lệnh in ra stdout cùng cấu trúc JSON mà ứng dụng lưu khi phân tích (báo cáo dấu hiệu).
Nếu thoát mã khác 0, đó là `ErrorResult` (ví dụ file không đọc được), không phải một verdict
forensics trên ảnh. Giữa hai lần chạy trên cùng file, chỉ `report_id`, `created_at` và
`methods[].duration_ms` có thể thay đổi; các trường forensic còn lại giữ nguyên khi cùng
byte file, cùng `--methods`, cùng bản build và cùng ngưỡng.

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
   - **Xuất Markdown:** sau khi có báo cáo, nút *Xuất báo cáo (Markdown)*
     lưu file `.md` (qua `ReportService.export_markdown`): disclaimer, metadata
     ảnh, hash ngưỡng, bảng từng method — không xác suất “ảnh giả”.
6. **Lịch sử** — tab "Lịch sử" liệt kê phân tích đã lưu (SQLite); xóa sẽ
   xóa cả artifacts. Chọn một dòng → *Xuất báo cáo* (cùng định dạng Markdown).
   Tab "Thống kê" tổng hợp phiên làm việc.

## Tab Tăng cường (xem, không phân tích)

Tab "Tăng cường" có ba operator cổ điển — *Unsharp mask*, *CLAHE*,
*Non-local means* — và lựa chọn tùy chọn *NAFNet (AI)* để xem thử hiệu ứng
trên ảnh. Tab này **không phân
tích**, không gọi `AnalysisService`, không ghi lịch sử hay database, và
không kết luận điều gì về ảnh.

- **Banner cố định**: "Ảnh đã tăng cường bằng thuật toán hoặc mô hình AI.
  Đây là bản xem, không phải bằng chứng. Phân tích dấu hiệu chỉ gắn với
  file gốc." Nghĩa là ảnh tăng cường chỉ để xem; dấu hiệu forensics chỉ
  được gắn với file gốc, không phải với bản tăng cường.
- **NAFNet (AI)**: không có tham số chỉnh trong tab; chọn nó sẽ hiện banner
  cảnh báo AI riêng. Extra tùy chọn trong `pyproject.toml` tên **`lama`**
  (dùng chung cho mọi suy luận ONNX model-forge, gồm NAFNet) — cài bằng
  `pip install -e ".[lama]"` để kéo **onnxruntime**; không bắt buộc cho
  copy-move/ELA. Cung cấp riêng file
  `models/deblurring_nafnet_2025may.onnx`. Ứng dụng kiểm tra kích thước
  91.736.251 byte và SHA-256
  `07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587`
  trước khi nạp; thiếu/sai weight sẽ báo lỗi. Chạy CPU, xử lý trên canvas
  có cạnh tối thiểu 512px và cạnh dài tối đa 1024px, rồi trả bản xem về
  kích thước ban đầu. Đây không phải model phát hiện giả mạo.
- **Thanh trộn**: 0 = ảnh gốc, 100 = ảnh tăng cường. Thanh chỉ đổi hiển thị
  (pixmap), không ghi pixel và không chạy lại operator.
- **Xuất**: chọn đích PNG, hệ thống ghi thêm một file `.json` kế bên
  (cùng tên) ghi `parent_sha256`, `derived_sha256`, `method`, `params`,
  `model_sha256` (null với operator cổ điển, hash weight với NAFNet) và
  `created_utc`. Hộp thoại thành công hiển thị cả PNG và JSON. File gốc không bao giờ bị ghi đè;
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

Tên method không có trong registry (ví dụ gõ nhầm) in một dòng lỗi ra stderr và thoát mã `2` — không in traceback.

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

Mỗi lần quét in một dòng tổng kết đúng dạng:

```text
ingest once: analyzed=1 errors=0 skipped_duplicate=0 deferred=2
```

`max_pending` (khóa cấp cao nhất trong file YAML của ingest, số nguyên dương,
mặc định `1`) giới hạn số file **mỗi lần quét** gửi vào phân tích. File vượt
hạn mức bị **deferred**: vẫn nằm trên đĩa và có dòng custody `deferred`, nhưng
**chưa được phân tích, chưa được decode hay kiểm tra hợp lệ** — mới chỉ qua
kiểm tra magic bytes và độ ổn định của file. Vì vậy `errors=0` cùng
`deferred>0` **không** có nghĩa mọi file trong thư mục đều hợp lệ: một file
hỏng nằm trong nhóm deferred chỉ làm tăng `errors` khi lần quét sau thực sự
gửi nó đi phân tích. `deferred` chỉ đếm số file **mới** bị đánh dấu trong lần
quét đó; file đã bị đánh dấu từ lần trước không được đếm lại dù vẫn đang chờ.

Để xử lý phần còn lại, chạy lại `ingest once <dir>` (file từng bị deferred
được xếp trước ở lần quét kế tiếp) hoặc tăng `max_pending` trong file cấu hình
ingest; `ingest watch` lặp lại hạn mức đó ở mỗi lần quét.
