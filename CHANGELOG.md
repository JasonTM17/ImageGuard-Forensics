# Changelog

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dự án tuân [Semantic Versioning](https://semver.org/spec/v2.0.0.html) khi có tag; hiện **chưa** có tag phát hành — mọi thay đổi nằm dưới **Unreleased**.

## [Unreleased]

### Added

- Ứng dụng desktop PySide6: phân tích copy-move (keypoint SIFT/ORB, block DCT), ELA JPEG, tín hiệu phụ (noise/tần số/metadata/EXIF), fusion verdict thận trọng.
- CLI `python -m imageguard analyze` — in `ForensicReport` JSON; lỗi I/O → `ErrorResult`, exit khác 0.
- GUI với overlay, lịch sử artifact, tab Enhance **chỉ xem** (không đưa vào fusion verdict).
- Ingest thư mục: `ingest once` / `ingest watch`, log custody JSONL; tùy chọn HTTP snapshot allowlist và `enhance_sibling` (file `.view` bên cạnh nguồn).
- NAFNet ONNX tùy chọn cho xem thử (model-forge); operator cổ điển unsharp/CLAHE/…
- Khung đánh giá: metrics R1–R5, ModelBench, runner tổng hợp; protocol chất lượng detector đóng băng; study enhancement cổ điển có seed/bootstrap.
- SynthForge / split dataset; SQLite artifact store.
- Bộ operator xử lý ảnh cổ điển NumPy-only (`ipkit`) — demo/education, không dùng cho verdict.
- CI GitHub Actions (`ci`): Windows, Python 3.14, `ruff check src tests scripts`, `pytest -q` với `QT_QPA_PLATFORM=offscreen`; job `test-py313` tương thích 3.13.
- Tài liệu `docs/` (kiến trúc, thuật toán, ingest, bảo mật, đánh giá, user guide).

### Changed

- Đồng bộ README và hướng dẫn runtime với layout code và giới hạn đo CASIA đã sàng lọc.
- CI: job Python 3.13 không chặn merge (`continue-on-error`).
- Vector hóa một số toán tử `ipkit` (threshold thích ứng, morphology).

### Fixed

- Forensics: abstain block khi texture không đủ; cancel cooperative; splicing cue thiếu → inconclusive.
- Evaluation: loại trừ ground truth không dùng được; tách study theo scene gốc; enhancement không ghi đè tập nguồn.
- UI: đồng bộ canvas/overlay; xóa overlay stale; export enhance đặt tên đủ file; argv/drag-drop không giữ progress cũ.
- Runtime/PyInstaller: đường dẫn resource standalone; báo lỗi build rõ hơn.

### Docs

- Ghi nhận study enhancement N=24 (cổ điển); NAFNet arm NOT_RUN trên cohort dễ (`docs/enhancement-study.md`).
- Tài liệu ingest HTTP, frozen comparison studies, giới hạn đo trên README.
- Thêm `CONTRIBUTING.md`, `SECURITY.md`, mẫu GitHub issue/PR, `docs/limitations.md`.
