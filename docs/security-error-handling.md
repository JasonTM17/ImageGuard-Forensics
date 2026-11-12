# Security & Error Handling

## Mặt đe dọa ảnh đầu vào

| Threat | Mitigation |
|---|---|
| Decompression bomb (ảnh vài KB → hàng gigapixel) | `MAX_PIXELS=80MP` check qua header probe TRƯỚC decode; `MAX_DECODE_SIDE=16384` chặn cạnh đơn; `DecompressionBombError` của Pillow map về `TOO_MANY_PIXELS` |
| File khổng lồ | `MAX_FILE_BYTES=100MB` reject sớm bằng stat |
| Magic bytes giả (đuôi .png nhưng nội dung khác) | Sniff magic + Pillow header probe — format thật quyết định, không tin extension |
| Truncated/corrupt | Decode trong try → `ErrorResult` có mã, không crash |
| Path traversal artifact | ArtifactStore: `analysis_id`/filename sanitize `[a-z0-9_-]` (filename thêm `.ext`), `abs_path` resolve rồi kiểm tra còn trong root |
| OOM trên ảnh lớn (aux methods) | `AUX_MAX_SIDE=4096` downscale working copy cho noise/freq (kernel đổi tỉ lệ thuận, mask/heat upscale về kích thước gốc, ghi `scale` trong evidence + limitation) — integral float64/FFT complex128 ~10-25 byte/px |
| SQL injection | SQLAlchemy ORM parameterized — không raw SQL string concat |
| Unicode path Windows | imencode+write_bytes (không cv2.imwrite trực tiếp) |
| Qt worker crash | Worker không chạm QWidget; exception → `failed` signal, không chết thread pool |

## Error taxonomy

`ErrorResult.code`: `PATH_INVALID` · `FILE_UNREADABLE` · `UNSUPPORTED_FORMAT` ·
`FILE_TOO_LARGE` · `TOO_MANY_PIXELS` · `IMAGE_TOO_SMALL` · `DECODE_FAILED` ·
`INTERNAL`. Failure mở/đọc ảnh trả structured result — caller (CLI/UI) quyết
định hiển thị, không raise qua tầng domain.

## Không làm

- Không tải ảnh lên dịch vụ khác; chỉ tạo yêu cầu kéo HTTP snapshot khi được cấu hình như mô tả bên dưới, không telemetry.
- Không ghi path tuyệt đối vào DB — chỉ relative artifact paths.
- Không log secrets (project không có secret nào theo thiết kế).

## Ingest HTTP snapshots

- Chỉ kéo ảnh từ host có trong allowlist; mọi redirect được kiểm tra lại. Loopback cần bật `allow_loopback`; link-local và metadata cần bật `allow_link_local`. RFC1918 vẫn được phép khi host hoặc IP nằm trong allowlist.
- Mật khẩu chỉ đọc từ tên biến môi trường đã cấu hình hoặc từ tệp secrets tuyệt đối nằm ngoài repository. Không đặt userinfo trong URL. Log và kết quả không chứa URL hay thông tin xác thực. Cấu hình thông tin đăng nhập không xác minh danh tính camera.
- `urllib` chuẩn phân giải DNS trước khi kết nối, nhưng DNS rebinding giữa lần kiểm tra và TCP connect vẫn là rủi ro còn lại.
- Snapshot tải về được giữ lại cho đến khi operator xóa; ứng dụng không tải ảnh lên dịch vụ khác.
- Khi được bật, tính năng này tạo yêu cầu HTTP(S) tới các host trong allowlist để kéo snapshot; ảnh không được gửi tới dịch vụ bên ngoài.
- Thư mục ingest có thể chứa ảnh người. `consent_note` trong log thư mục là ghi chú do operator nhập, không phải bằng chứng về sự đồng ý.

## Ingest viewing sibling

`enhance_sibling` defaults off. When an operator turns it on, ingest writes
`<stem>.view.png` and `<stem>.view.json` beside the source after a successful
analysis. The source bytes stay unchanged. Later scans ignore those `.view`
names, and `AnalysisService` is called only with the source path. The sibling
is a viewing file with a new digest. It is not the parent exhibit, and its
pixels are not passed to fusion. A custody failure tag is the fixed string
`enhance_sibling_failed` with no exception text and no path. The operator
still has to treat the watched folder as sensitive: it holds the original
images and, when this option is on, a derived copy.
