# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` (branch `main`) | Yes |

Không có bản phát hành có tag semver riêng tại thời điểm hiện tại; bản vá bảo mật áp dụng cho commit mới nhất trên `main`.

## Báo cáo lỗ hổng (private)

**Không** mở issue công khai cho lỗ hổng bảo mật.

Vui lòng dùng **[GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)** cho repository này (tab **Security** → **Report a vulnerability**). Nếu tính năng chưa bật trên fork, liên hệ maintainer qua kênh lớp học/đồ án đã được công bố trên repo — **không** gửi báo cáo qua issue tracker công khai.

Cho phép maintainer hỏi lại, đề xuất bản vá, và phối hợp thời gian công bố.

## Phạm vi

ImageGuard Forensics là **ứng dụng desktop cục bộ**: ảnh được mở và phân tích trên máy người dùng. Không có dịch vụ đám mây bắt buộc và không thiết kế telemetry.

### Đầu vào ảnh (`safe_open`)

Mô tả đầy đủ: [`docs/security-error-handling.md`](docs/security-error-handling.md).

- Từ chối file **> 100 MB** (`MAX_FILE_BYTES`).
- Từ chối ảnh **> 80 megapixel** sau header probe (`MAX_PIXELS`).
- Kiểm tra magic bytes, decode an toàn → `ErrorResult` thay vì crash.
- Artifact store: sanitize tên file, chặn path traversal.

### Ingest thư mục và HTTP snapshot

Mô tả vận hành: [`docs/ingest.md`](docs/ingest.md). Chính sách HTTP: [`docs/security-error-handling.md`](docs/security-error-handling.md) (mục *Ingest HTTP snapshots*).

- HTTP(S) chỉ tới host trong **allowlist**; redirect được kiểm tra lại.
- Loopback, link-local, metadata IP cần cờ bật tường minh; RFC1918 chỉ khi nằm trong allowlist.
- Không đặt mật khẩu trong URL; credentials qua biến môi trường hoặc file secrets **ngoài** repo.
- Log/kết quả không chứa URL đầy đủ hay thông tin xác thực.

### Ngoài phạm vi (thông thường)

- Hiệu năng thuần túy hoặc false positive forensics không phải CVE (dùng issue thường + [`docs/limitations.md`](docs/limitations.md)).
- Model ONNX tùy chọn do người dùng tự tải (`docs/modelforge.md`).
- PyInstaller bundle — xem [`docs/packaging.md`](docs/packaging.md).

## Rủi ro còn lại đã biết

- **DNS rebinding** giữa bước phân giải DNS và kết nối TCP khi dùng ingest HTTP: được ghi nhận trong [`docs/security-error-handling.md`](docs/security-error-handling.md); không coi là đã triệt tiêu hoàn toàn.
- Thư mục ingest có thể chứa ảnh nhạy cảm; `consent_note` là ghi chú operator, không phải bằng chứng pháp lý về đồng ý.

## Ranh giới sản phẩm

Báo cáo bảo mật không thay đổi ranh giới khoa học: công cụ **không** khẳng định forgery; chỉ xử lý dữ liệu đầu vào và artifact cục bộ.
