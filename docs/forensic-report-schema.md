# Hợp đồng JSON máy đọc — `python -m imageguard analyze`

Lệnh `analyze` ghi **một dòng JSON** ra stdout (UTF-8, `indent=1`). Caller phân biệt hai dạng:

| Dạng | Khi nào | Mã thoát |
|------|---------|----------|
| **ForensicReport** | Mở ảnh thành công và pipeline phân tích chạy xong (kể cả verdict `error` ở từng method) | `0` |
| **ErrorResult** | Không tạo được báo cáo (thường lỗi `safe_open` trước pipeline) | `1` |

Lỗi **cú pháp CLI / tham số** trên lệnh `eval` hoặc `ingest` dùng mã thoát `2` (không in ForensicReport/ErrorResult JSON kiểu trên). Lệnh `analyze` không dùng mã `2`.

Nguồn sự thật trong mã: `report_to_dict()` (`imageguard.services.analysis_service`), payload ErrorResult trong `imageguard.cli._run_analyze_command`, enum `Verdict` / `ErrorCode`, model `ForensicReport` / `AlgorithmResult`.

## Khóa cố định (drift test)

Khối sau liệt kê **đúng tập khóa** mà CLI xuất; test `tests/test_report_schema_doc.py` so khớp với `report_to_dict` và ErrorResult thực tế.

```json forensic-report-contract
{
  "report_id": "00000000-0000-4000-8000-000000000000",
  "image_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "image_name": "example.jpg",
  "image_width": 128,
  "image_height": 128,
  "created_at": "1970-01-01T00:00:00+00:00",
  "tool_version": "0.1.0",
  "env": {
    "python": "3.14.0",
    "opencv": "5.0.0",
    "numpy": "2.5.0"
  },
  "thresholds_hash": "0000000000000000",
  "overall_verdict": "inconclusive",
  "limitations": [],
  "disclaimer": "…",
  "methods": [
    {
      "method": "ela",
      "verdict": "no_indication",
      "score": 0.0,
      "score_definition": "…",
      "evidence": {},
      "limitations": [],
      "params_hash": "0000000000000000",
      "duration_ms": 0,
      "has_mask": false
    }
  ]
}
```

```json error-result-contract
{
  "code": "DECODE_FAILED",
  "message": "…",
  "detail": {}
}
```

## ForensicReport — cấp gốc

| Trường | Kiểu JSON | Khi null / rỗng | Ý nghĩa |
|--------|-----------|-----------------|--------|
| `report_id` | string | không | UUID4 cho phiên phân tích; **thay đổi mỗi lần chạy**. |
| `image_sha256` | string (64 hex) | không | SHA-256 của **byte file gốc**; xác định ảnh trong history. |
| `image_name` | string | không | Basename file (không có đường dẫn tuyệt đối). |
| `image_width` | number (int) | không | Chiều rộng pixel sau `exif_transpose`. |
| `image_height` | number (int) | không | Chiều cao pixel sau `exif_transpose`. |
| `created_at` | string | không | ISO-8601 UTC; **thay đổi mỗi lần chạy**. |
| `tool_version` | string | không | Phiên bản gói (`imageguard.__version__`). |
| `env` | object | không (có thể rỗng trong model nhưng CLI luôn điền) | Môi trường chạy — xem bảng con bên dưới. |
| `thresholds_hash` | string | có thể `""` | 16 ký tự đầu hash snapshot ngưỡng đã dùng. |
| `overall_verdict` | string | không | Một giá trị enum `Verdict` (bảng enum). |
| `limitations` | array of string | có thể `[]` | Cảnh báo gộp từ mọi method + fusion + (nếu có) lỗi persist. |
| `disclaimer` | string | không | Văn bản pháp lý mặc định trên model `ForensicReport`. |
| `methods` | array of object | có thể `[]` | Kết quả từng phương pháp — xem bảng method. |

### `env` (lồng trong ForensicReport)

| Trường | Kiểu | Ý nghĩa |
|--------|------|--------|
| `python` | string | `platform.python_version()` |
| `opencv` | string | `cv2.__version__` |
| `numpy` | string | `numpy.__version__` |

Không có khóa `env` nào được phép null trong JSON CLI hiện tại.

## ForensicReport — phần tử `methods[]`

Mỗi phần tử tương ứng một `AlgorithmResult` sau khi **loại bỏ** mảng NumPy (`mask`, `heatmap`, và mọi giá trị ndarray trong `evidence`).

| Trường | Kiểu JSON | Khi null / rỗng | Ý nghĩa |
|--------|-----------|-----------------|--------|
| `method` | string | không | Giá trị enum `MethodName` (ví dụ `copy_move_keypoint`, `ela`, `exif_audit`). |
| `verdict` | string | không | Enum `Verdict` cho method đó. |
| `score` | number hoặc `null` | `null` khi method không gán số (ví dụ EXIF) | Chỉ số định lượng **có định nghĩa**; không phải xác suất “ảnh giả”. |
| `score_definition` | string | `""` khi `score` là `null` | Bắt buộc mô tả đơn vị/ý nghĩa khi `score` ≠ null (validation trong model). |
| `evidence` | object | có thể `{}` | Số liệu JSON-serializable; khóa phụ thuộc method (bảng gợi ý bên dưới). |
| `limitations` | array of string | có thể `[]` | Giới hạn áp dụng riêng cho kết quả method này. |
| `params_hash` | string | có thể `""` | 16 ký tự đầu hash cấu hình/tham số method. |
| `duration_ms` | number (int) | không | Thời gian chạy method (ms); **thay đổi theo máy/tải**. |
| `has_mask` | boolean | không | `true` nếu method có mask trong bộ nhớ; mask **không** nhúng trong JSON (chỉ artifact khi persist). |

**Cặp `score` + `score_definition`:** Nếu `score` là số thì `score_definition` phải khác rỗng và giải thích đúng đại lượng (ví dụ inlier RANSAC, phần diện tích mask, p99 heatmap). Caller không được diễn giải `score` mà không đọc `score_definition`.

### `evidence` — khóa thường gặp (không đóng; có thể thêm khóa mới khi đổi detector)

| Method (giá trị `method`) | Khóa evidence điển hình | Ghi chú |
|---------------------------|-------------------------|---------|
| `copy_move_keypoint` | `n_keypoints`, `n_pairs_raw`, `n_pairs_filtered`, `n_verified`, `n_clusters`, `backend`, `pairs_image` | `pairs_image` luôn `null` trên stdout (ảnh chỉ khi persist). |
| `copy_move_block` | `n_blocks`, `n_candidate_pairs`, `n_shifts_kept`, `top_shift_votes`, `work_size` | `work_size` là `[height, width]`. |
| `ela` | `qualities`, `p99`, `per_quality_p99` | Heatmap không có trong JSON. |
| `exif_audit` | `findings`, `exif_present`, `exif_missing` | Không có `score`. |
| Các method khác (`noise_inconsistency`, `noise_local`, `frequency_analysis`, `splicing_cues`) | khóa riêng từng pipeline | Cùng quy tắc: không ndarray trên JSON. |

Giá trị `null` trong evidence có nghĩa field không áp dụng hoặc không xuất được trên kênh JSON.

## Enum `Verdict` (đóng)

Giá trị string trên JSON **không** dùng `forged` / `fake`. Ý nghĩa vận hành:

| Giá trị | Ý nghĩa |
|---------|--------|
| `suspicious_indication` | Có dấu hiệu nghi vấn theo luật method (detector conclusive như keypoint/block); fusion có thể đẩy `overall_verdict` lên giá trị này. |
| `no_indication` | Method không thấy dấu hiệu đủ mạnh theo ngưỡng. |
| `inconclusive` | Tín hiệu yếu, mâu thuẫn, hoặc chỉ mang tính khám phá (nhiều auxiliary method). |
| `not_applicable` | Method không chạy được cho ảnh/định dạng này (loại khỏi fusion). |
| `error` | Method lỗi hoặc bị hủy; fusion ghi warning, **không** làm fail CLI (`exit 0` vẫn có thể). |

`overall_verdict` do `combine()` (`imageguard.forensics.fusion`) tính từ danh sách method; không có trường `signals`/`warnings` riêng trên JSON CLI (warnings có thể lẫn vào `limitations`).

## Enum `MethodName` (đóng, giá trị `method`)

`copy_move_keypoint`, `copy_move_block`, `ela`, `exif_audit`, `noise_inconsistency`, `noise_local`, `frequency_analysis`, `splicing_cues`.

CLI `--methods` nhận alias registry (`keypoint`, `block`, `ela`, …); JSON luôn dùng **giá trị đầy đủ** ở trên.

## Ổn định giữa các lần chạy

Chỉ ba trường sau **thay đổi** giữa hai lần chạy trên cùng file byte, cùng
`--methods`, cùng bản build và cùng ngưỡng:

| Trường | Nguồn trong mã |
|--------|-----------------|
| `report_id` | `str(uuid.uuid4())` cho mỗi phiên phân tích |
| `created_at` | `datetime.now(UTC).isoformat()` khi dựng report |
| `methods[].duration_ms` | `time.perf_counter()` của từng pipeline, làm tròn xuống mili-giây |

Mọi trường forensic còn lại giữ nguyên: `image_sha256`, `image_name`,
`image_width`, `image_height`, `thresholds_hash`, `overall_verdict`,
`limitations`, `disclaimer`, và trong từng phần tử `methods[]`: `method`,
`verdict`, `score`, `score_definition`, `evidence`, `limitations`,
`params_hash`, `has_mask`. Ngoại lệ duy nhất là khi lần chạy gặp lỗi lưu
lịch sử/artifact: report thêm một cảnh báo vào `limitations` (xem
`AnalysisService.analyze`). `tool_version` và `env.*` chỉ đổi khi đổi bản cài
hoặc môi trường Python/OpenCV/NumPy, không đổi giữa hai lần chạy trên cùng
máy.

Tổng thời gian pipeline (wall-clock) **không** có trường ở JSON stdout; chỉ tổng `methods[].duration_ms` có thể dùng để ước lượng (không bằng wall-clock).

## ErrorResult (stdout, `exit 1`)

| Trường | Kiểu | Khi rỗng | Ý nghĩa |
|--------|------|----------|--------|
| `code` | string | không | Một giá trị `ErrorCode` (bảng dưới). |
| `message` | string | không | Thông báo ngắn cho người vận hành. |
| `detail` | object | có thể `{}` | Dữ kiện máy đọc tùy mã lỗi (`size`, `width`, `height`, `head`, …). |

### Enum `ErrorCode` (đóng)

| Mã | Ý nghĩa ngắn |
|----|----------------|
| `PATH_INVALID` | Không phải file tồn tại. |
| `FILE_UNREADABLE` | Không đọc được byte (quyền, khóa, I/O). |
| `UNSUPPORTED_FORMAT` | Magic bytes không khớp định dạng hỗ trợ. |
| `FILE_TOO_LARGE` | Vượt `max_file_bytes`. |
| `TOO_MANY_PIXELS` | Header/decode vượt giới hạn pixel (bomb / quá lớn). |
| `IMAGE_TOO_SMALL` | Cạnh nhỏ hơn `min_side`. |
| `DECODE_FAILED` | Header/decode hỏng hoặc không nhận dạng được. |
| `INTERNAL` | Lỗi không phân loại (CLI fallback nếu kiểu kết quả lạ). |

## Ví dụ ForensicReport (ảnh JPEG 128×128, `--no-persist`, mặc định `keypoint,block,ela` + EXIF)

Sinh bằng lệnh (đã chạy trên worktree `docs/report-schema`, seed RNG 42 khi tạo ảnh):

`python -m imageguard analyze <ảnh> --no-persist`

Đã thay `report_id`, `created_at`, `image_sha256` bằng placeholder; giữ số liệu evidence từ một lần chạy thật.

```json forensic-report-example
{
 "report_id": "<uuid-v4>",
 "image_sha256": "<sha256-64-hex>",
 "image_name": "tiny.jpg",
 "image_width": 128,
 "image_height": 128,
 "created_at": "<iso-8601-utc>",
 "tool_version": "0.1.0",
 "env": {
  "python": "3.14.2",
  "opencv": "5.0.0",
  "numpy": "2.5.4"
 },
 "thresholds_hash": "df80dcb9a6d32070",
 "overall_verdict": "inconclusive",
 "limitations": [
  "block-DCT chỉ phát hiện translate copy-move — rotate/scale/flip làm hệ số DCT đổi → không match (giới hạn trường phái)",
  "ELA chỉ là bằng chứng khám phá — heatmap cao có thể do cạnh sắc/texture tự nhiên, không đồng nghĩa vùng bị chỉnh sửa"
 ],
 "disclaimer": "Công cụ chỉ phát hiện dấu hiệu nghi vấn từ phân tích tự động — không khẳng định ảnh bị can thiệp, cũng không xác thực nguyên bản.",
 "methods": [
  {
   "method": "copy_move_keypoint",
   "verdict": "no_indication",
   "score": 0.0,
   "score_definition": "số cặp keypoint match qua geometric verify (RANSAC affine inliers)",
   "evidence": {
    "n_keypoints": 63,
    "n_pairs_raw": 63,
    "n_pairs_filtered": 0,
    "n_verified": 0,
    "n_clusters": 0,
    "backend": "sift",
    "pairs_image": null
   },
   "limitations": [],
   "params_hash": "df80dcb9a6d32070",
   "duration_ms": 31,
   "has_mask": false
  },
  {
   "method": "copy_move_block",
   "verdict": "no_indication",
   "score": 0.0,
   "score_definition": "matched_area_fraction = diện tích mask / diện tích ảnh (shift-voted)",
   "evidence": {
    "n_blocks": 12769,
    "n_candidate_pairs": 0,
    "n_shifts_kept": 0,
    "top_shift_votes": 0,
    "work_size": [128, 128]
   },
   "limitations": [
    "block-DCT chỉ phát hiện translate copy-move — rotate/scale/flip làm hệ số DCT đổi → không match (giới hạn trường phái)"
   ],
   "params_hash": "df80dcb9a6d32070",
   "duration_ms": 96,
   "has_mask": true
  },
  {
   "method": "ela",
   "verdict": "inconclusive",
   "score": 1.0,
   "score_definition": "p99 của heatmap ELA (max qua sweep quality), scale [0,1] đã amplify",
   "evidence": {
    "qualities": [75, 85, 90, 95],
    "p99": 1.0,
    "per_quality_p99": {"75": 1.0, "85": 1.0, "90": 1.0, "95": 0.902}
   },
   "limitations": [],
   "params_hash": "df80dcb9a6d32070",
   "duration_ms": 18,
   "has_mask": true
  },
  {
   "method": "exif_audit",
   "verdict": "inconclusive",
   "score": null,
   "score_definition": "",
   "evidence": {
    "findings": ["ảnh không có EXIF — phổ biến sau re-save/qua mạng xã hội; không phải bằng chứng giả mạo"],
    "exif_present": {},
    "exif_missing": ["Make", "Model", "Software", "DateTime", "DateTimeOriginal", "Orientation"]
   },
   "limitations": [],
   "params_hash": "",
   "duration_ms": 0,
   "has_mask": false
  }
 ]
}
```

## Ví dụ ErrorResult (file JPEG hỏng, `exit 1`)

```json error-result-example
{
 "code": "DECODE_FAILED",
 "message": "Header không đọc được: cannot identify image file <_io.BytesIO object at 0x…>",
 "detail": {}
}
```

Thông báo có thể chứa địa chỉ đối tượng Python trong traceback Pillow; chỉ dùng cho hiển thị, không parse ổn định.

## Mã thoát tiến trình

| Mã | `analyze` | Ghi chú |
|----|-----------|--------|
| `0` | Có | JSON ForensicReport, kể cả method `verdict: "error"`. |
| `1` | Có | JSON ErrorResult hoặc fallback INTERNAL. |
| `2` | Không | Dành cho `eval` / `ingest` khi tham số không hợp lệ. |
