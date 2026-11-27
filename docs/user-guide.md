# Hướng dẫn sử dụng — ImageGuard Forensics

## Chạy app

**Khuyến nghị (source):**

```powershell
.\run_app.bat
# hoặc
.\.venv\Scripts\python.exe -m imageguard        # GUI mặc định
.\.venv\Scripts\python.exe -m imageguard gui    # tường minh
```

**Bản đóng gói (`dist/ImageGuard/ImageGuard.exe`):** bản build
PyInstaller 6.22.3 + Python 3.13.12 đã được verify 2026-11-16 — mở GUI,
nạp ảnh và chạy trọn luồng phân tích với verdict đúng. Chi tiết QA trong
[`docs/packaging.md`](packaging.md). Nếu gặp lỗi import `QtWidgets` trên
máy khác, quay lại cách chạy từ source ở trên.

Mốc QA của bản đóng gói là bằng chứng cho bản được kiểm tra tại thời điểm đó.
Các sửa lỗi mới cần bản build và QA tương ứng; khi chưa có receipt cho bản
mới, dùng source hiện tại thay vì mặc định executable cũ đã chứa các sửa lỗi này.

## Phân tích qua CLI (báo cáo JSON)

```powershell
.\.venv\Scripts\python.exe -m imageguard analyze <file> `
  [--methods keypoint,block,ela] [--no-persist]
```

Lệnh in ra stdout cùng cấu trúc JSON mà ứng dụng lưu khi phân tích (báo cáo dấu hiệu).
Exit 1 là `ErrorResult` đầu vào (ví dụ file không đọc được); exit 2 là lỗi
tham số/cấu hình/model. Các mã này không phải verdict forensics trên ảnh.
Giữa hai lần chạy trên cùng file, chỉ `report_id`, `created_at` và
`methods[].duration_ms` có thể thay đổi; các trường forensic còn lại giữ nguyên khi cùng
byte file, cùng `--methods`, cùng bản build và cùng ngưỡng.

Nếu đã huấn luyện mô hình DIP theo [giao thức Deepfake](deepfake-evaluation.md),
thêm `--methods deepfake --deepfake-model <model.json> --config <thresholds.yaml>`.
Model phải khớp bộ trích đặc trưng và cấu hình đã dùng khi fit. Không có model
đi kèm repo; tùy chọn này không tải model từ Internet.

## Quy trình phân tích

1. **Mở ảnh hoặc Chụp từ Camera…** — kéo-thả vào canvas, dùng nút "Mở ảnh…", hoặc nút **"Chụp từ Camera…"** (`CameraCaptureDialog`) để thu khung hình thô từ Webcam máy tính (`index 0`), Camera USB ngoài (`index 1..5`), hoặc Camera IP/IoT (`HTTP Snapshot URL`). Định dạng ảnh mở: JPEG,
   PNG, BMP, TIFF, WebP (≤100MB, ≤80MP, cạnh ≥64px — ảnh quá lớn/corrupt
   báo lỗi thân thiện, không crash).
2. **Xem preview + metadata** — kích thước, định dạng, dung lượng, EXIF.
   Nếu file bị thay đổi sau khi mở, kết quả có hash khác preview sẽ bị từ
   chối hiển thị; mở lại ảnh rồi phân tích để bảo đảm đúng ảnh/kết quả.
3. **Chọn phương pháp** —
   - *Copy-Move keypoint (SIFT/ORB)*: tìm vùng nhân đôi trong cùng ảnh.
   - *Copy-Move block (DCT)*: baseline kinh điển, chậm hơn nhưng độc lập.
   - *ELA*: chỉ có ý nghĩa với JPEG; PNG/BMP sẽ hiện "Không áp dụng".
   - *Noise inconsistency / Frequency / Splicing cues / Dấu hiệu Deepfake / AI (biên khuôn mặt & phổ)* (L2 — `deepfake`): bằng chứng
     khám phá (`AUX_METHODS`) — KHÔNG BAO GIỜ tự kết luận. Khi chụp khung hình mới từ Camera trong tab Phân tích, hệ thống tự bật ô kiểm *Dấu hiệu Deepfake / AI* để kiểm tra trực tiếp trên ảnh cảm biến gốc.
4. **Phân tích** — thanh progress theo stage; nút "Hủy" dừng mềm giữa
   stages (cooperative cancel).
5. **Đọc kết quả** —
   - Banner verdict 3 trạng thái chính: *Có dấu hiệu nghi vấn* (cam),
     *Không thấy dấu hiệu* (xám), *Không đủ bằng chứng* (vàng trung tính).
   - Cards từng method: verdict + thời gian chạy (ms) + score + định nghĩa score + chỉ số evidence + limitations.
   - **Công cụ hỗ trợ thuyết trình & kiểm thử ngay trên GUI:**
     - *Tạo kịch bản ảnh mẫu…*: sinh nhanh 4 kịch bản tổng hợp (`Copy-Move`, `Splicing`, `Deepfake / AI`, `Ảnh đối chứng sạch`) và tự động chọn nhóm method phù hợp.
     - Các nút chọn nhanh (*Mặc định (3)*, *Chế độ Camera / AI*, *Tất cả (8)*) và nút *Đưa sang Tăng cường / Siêu phân giải →* chuyển ảnh gốc sang tab Tăng cường.
   - Dropdown overlay: mask đỏ / heatmap trên ảnh gốc (zoom: lăn chuột,
     pan: kéo).
   - Kết quả là *dấu hiệu*, không phải kết luận; câu lưu ý nằm trong báo cáo xuất ra.
   - **Xuất Markdown:** sau khi có báo cáo, nút *Xuất báo cáo (Markdown)*
     lưu file `.md` (qua `ReportService.export_markdown`): disclaimer, metadata
     ảnh, hash ngưỡng, bảng từng method — không xác suất “ảnh giả”.
6. **Lịch sử** — tab "Lịch sử" liệt kê phân tích đã lưu (SQLite); xóa sẽ
   xóa cả artifacts. Chọn một dòng → *Xuất báo cáo* (cùng định dạng Markdown).
   Tab "Thống kê" tổng hợp phiên làm việc.

Nếu lưu lịch sử/artifact thất bại, kết quả tính toán vẫn được trả về kèm
cảnh báo trong limitations, nhưng lượt lưu chưa hoàn tất được thu hồi.
Nếu cảnh báo ghi cleanup không hoàn tất hoặc cần `recovery`, giữ mã báo cáo
được nêu để kiểm tra và phục hồi trước khi tin vào lịch sử hay bản xuất còn sót.

Nhánh Deepfake/AI chỉ bật chế độ hai camera khi ảnh PNG có khai báo bố cục
thu nhận hợp lệ; ảnh panorama hoặc webcam góc rộng vẫn là ảnh đơn. Ranh giới
camera được lưu đúng kể cả khi hai ảnh có chiều rộng khác nhau. ROI đủ kích
thước, tương phản và nền đối chiếu được ưu tiên. Khi thiếu mặt hoặc thiếu
điều kiện đo, phương pháp hiện *Không đủ bằng chứng*, lý do và không cấp
điểm. Các số đo vùng trung tâm không có mặt chỉ phục vụ minh họa.

Với ảnh tải lên, bật *Dấu hiệu Deepfake / AI* hoặc chọn *Chế độ Camera / AI*.
Điểm chỉ số trên thang 0–1 không phải xác suất ảnh giả. `NO_INDICATION` cũng
không chứng minh ảnh thật. Muốn đo chất lượng trên dữ liệu có nhãn, dùng
[hướng dẫn đánh giá Deepfake](deepfake-evaluation.md).

**Dùng mô hình đã học:** bấm *Nạp mô hình DIP…* trong nhóm Phương pháp và
chọn JSON đã huấn luyện/đánh giá. Chọn hợp lệ sẽ bật phương pháp Deepfake và
xóa kết quả cũ để tránh lẫn mô hình. File không hợp lệ giữ lựa chọn trước và
báo lý do; *Dùng trọng số mặc định* quay lại heuristic. Khi chạy, GUI khóa
việc đổi model; nếu file bị sửa sau khi chọn, cần chọn lại. Card kết quả ghi
rõ nhánh learned, chỉ số chưa hiệu chuẩn và nhắc nếu ảnh thuộc dữ liệu dev.
Ảnh không đủ điều kiện vẫn không có score; heatmap không phải mask của
classifier. Mô hình chỉ áp dụng ở tab Phân tích, chưa thay nhánh preview
camera hoặc hỗ trợ ảnh ghép hai camera.

Trong hộp thoại camera, webcam mặc định chụp 5 khung. Ảnh xem trước và khung
vừa chụp hiện các trụ cột khám phá trên khung gốc. Điểm đứng yên hiện khi bước
phân tích burst định vị được vùng mặt. Bấm *Dùng khung này để phân tích* để đưa
PNG sang tab Phân tích. Ô *Chụp 5 khung trong vài giây* chỉ dùng cho một webcam
hoặc một camera USB. Trên tab Phân tích, *Lưu khung này làm mốc cùng phiên* chuẩn hoá khung
hiện tại thành PNG lossless dưới `data/session_ref/`, không copy EXIF/GPS. Mốc
có hiệu lực 8 giờ và chỉ được dùng khi sha256 còn khớp sidecar; sidecar chỉ
giữ hash nguồn/hash bản chuẩn hoá và lifetime, không lưu basename hay đường dẫn
nguồn. Mốc hết hạn hoặc bị sửa sẽ không tham gia phân tích. Mốc và burst không cộng vào điểm không gian. Burst chỉ mô tả mức chuyển động;
mặt đứng yên không làm tăng nghi vấn Deepfake. Mốc có thể tạo cờ chênh lệch
riêng, chỉ dùng ảnh đơn; ảnh ghép hai camera không được lưu làm mốc mặt.
Công cụ không nhận diện danh tính và không kết luận ảnh giả.

## Tab Tăng cường (xem, không phân tích)

Tab "Tăng cường" có sáu operator cổ điển — *Unsharp mask*, *CLAHE*,
*Non-local means*, *Siêu phân giải cổ điển (Bicubic + IBP)* (`super_res`),
*Hiệu chuẩn Cảm biến & Khớp màu Camera* (`cam_calibrate`: cân bằng ánh sáng cục bộ /
khớp histogram song song 2 camera) và *Khử nhiễu & Làm nét Miền tần số (FFT)* (`freq_sharpen`:
lọc trung vị kết hợp khuếch đại cao tần Butterworth trên kênh độ chói) — và lựa chọn tùy chọn
*NAFNet (AI)* để xem thử hiệu ứng
trên ảnh (hỗ trợ cả mở file và nút *"Chụp từ Camera…"*). Tab này **không phân
tích**, không gọi `AnalysisService`, không ghi lịch sử hay database, và
không kết luận điều gì về ảnh. Khi dùng ảnh từ camera, luôn chạy phân tích
Deepfake/Forensics trên khung hình gốc từ cảm biến trước khi dùng *Siêu phân giải*
cho mục đích quan sát.

- **Banner cố định**: "Ảnh đã tăng cường bằng thuật toán hoặc mô hình AI.
  Đây là bản xem, không phải bằng chứng. Phân tích dấu hiệu chỉ gắn với
  file gốc." Nghĩa là ảnh tăng cường chỉ để xem; dấu hiệu forensics chỉ
  được gắn với file gốc, không phải với bản tăng cường.
- **NAFNet (AI)**: không có tham số chỉnh trong tab; chọn nó sẽ hiện banner
  cảnh báo AI riêng. Extra tùy chọn trong `pyproject.toml` tên **`lama`**
  (dùng chung cho mọi suy luận ONNX model-forge, gồm NAFNet) — cài bằng
  `pip install -e ".[lama]"` để kéo **onnxruntime**; không bắt buộc cho
  copy-move/ELA. Cung cấp riêng file
  `models/deblurring_nafnet_2025may.onnx` — với bản exe đóng gói thì đặt
  thư mục `models/` cạnh `ImageGuard.exe` (và build bằng `.[dev,lama]`,
  xem [`docs/packaging.md`](packaging.md)). Ứng dụng kiểm tra kích thước
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
- Nếu ghi JSON thất bại sau khi ghi PNG, ứng dụng phục hồi PNG trước đó
  hoặc xóa PNG vừa tạo nếu trước đó chưa có. Nếu phục hồi thất bại hoặc
  phát hiện file đích đã bị thay đổi, lỗi `recovery required` chỉ rõ bản
  sao `.enhance-*.bak` và SHA-256 cũ (nếu có). Giữ bản sao này, ngừng ghi
  vào cùng đích và đối chiếu hash trước khi phục hồi thủ công; cặp xuất
  đang lỗi chưa đủ điều kiện sử dụng. Đừng phân tích bản xem như ảnh gốc.
  Các lượt xuất trong cùng tiến trình được tuần tự hóa. Các tiến trình
  khác phải chọn đích riêng. Cơ chế này xử lý lỗi ghi thông thường;
  không bảo đảm phục hồi khi tiến trình bị dừng hoặc máy mất điện giữa hai lần ghi.
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

## Kiểm chứng camera thật

`PYTHONPATH=src python scripts/validate_camera_burst.py --device 0` chạy một live negative control và một frozen positive control trong thư mục tạm tự xoá. Output chỉ là JSON số/digest, không lưu ảnh hoặc database; kết quả dùng để kiểm tra pipeline, không phải accuracy cohort.

