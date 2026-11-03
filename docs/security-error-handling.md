# Security & Error Handling

## Mặt đe dọa ảnh đầu vào

| Threat | Mitigation |
|---|---|
| Decompression bomb (ảnh vài KB → hàng GP pixel) | `MAX_PIXELS=80MP` check qua header probe TRƯỚC decode; `Image.MAX_IMAGE_PIXELS` Pillow |
| File khổng lồ | `MAX_FILE_BYTES=100MB` reject sớm bằng stat |
| Magic bytes giả (đuổi .png nhưng nội dung khác) | Sniff magic + Pillow header probe — format thật quyết định, không tin extension |
| Truncated/corrupt | Decode trong try → `ErrorResult` có mã, không crash |
| Path traversal artifact | ArtifactStore: filename sanitize `[a-z0-9_-]`, `abs_path` resolve rồi kiểm tra còn trong root |
| SQL injection | SQLAlchemy ORM parameterized — không raw SQL string concat |
| Unicode path Windows | imencode+write_bytes (không cv2.imwrite trực tiếp) |
| Qt worker crash | Worker không chạm QWidget; exception → `failed` signal, không chết thread pool |

## Error taxonomy

`ErrorResult.code`: `PATH_INVALID` · `TOO_LARGE` · `TOO_MANY_PIXELS` ·
`TOO_SMALL` · `FORMAT_UNSUPPORTED` · `DECODE_FAILED`. Mọi failure trả
structured result — caller (CLI/UI) quyết định hiển thị, không raise qua
tầng domain.

## Không làm

- Không gửi ảnh ra ngoài (offline hoàn toàn; không telemetry).
- Không ghi path tuyệt đối vào DB — chỉ relative artifact paths.
- Không log secrets (project không có secret nào theo thiết kế).
