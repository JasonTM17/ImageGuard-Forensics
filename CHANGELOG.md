# Changelog

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dự án tuân [Semantic Versioning](https://semver.org/spec/v2.0.0.html) khi có tag; hiện **chưa** có tag phát hành — mọi thay đổi nằm dưới **Unreleased**.

## [Unreleased]

### Added

- Ứng dụng desktop PySide6: phân tích copy-move (keypoint SIFT/ORB, block DCT), ELA JPEG, tín hiệu phụ (noise/tần số/metadata/EXIF), fusion verdict thận trọng.
- CLI `python -m imageguard analyze` — in `ForensicReport` JSON; lỗi I/O → `ErrorResult`, exit khác 0.
- GUI với overlay, lịch sử artifact, tab Enhance **chỉ xem** (không đưa vào fusion verdict); tên trợ năng cho control, chuỗi giao diện tập trung và phản hồi trạng thái.
- Ingest thư mục: `ingest once` / `ingest watch`, log custody JSONL; tùy chọn HTTP snapshot allowlist và `enhance_sibling` (file `.view` bên cạnh nguồn).
- NAFNet ONNX tùy chọn cho xem thử (model-forge); operator cổ điển unsharp/CLAHE/…
- Khung đánh giá: metrics R1–R5, ModelBench, runner tổng hợp; protocol chất lượng detector đóng băng; study enhancement cổ điển có seed/bootstrap.
- SynthForge / split dataset; SQLite lưu metadata phân tích, artifact store ghi file trên đĩa.
- Bộ operator xử lý ảnh cổ điển NumPy-only (`ipkit`) — demo/education, không dùng cho verdict.
- Script tổng hợp verdict fusion offline từ các hàng detector đã lưu (`scripts/fusion_offline_summary.py`), kèm khoảng tin cậy Wilson.
- Script benchmark phương pháp (`scripts/bench_methods.py`) và script sinh hình báo cáo (`scripts/make_report_figures.py`).
- CI GitHub Actions (`ci`): Windows, Python 3.14, `ruff check src tests scripts`, `pytest -q` với `QT_QPA_PLATFORM=offscreen`; job `test-py313` tương thích 3.13.
- Giấy phép MIT (`LICENSE`).

### Changed

- Đồng bộ README và hướng dẫn runtime với layout code và giới hạn đo CASIA đã sàng lọc; README có mục lục trỏ tới các tài liệu chuyên đề.
- CI: job Python 3.13 không chặn merge (`continue-on-error`).
- Vector hóa một số toán tử `ipkit` (threshold thích ứng, morphology).
- Thay tham chiếu kế hoạch nội bộ trong comment/docstring bằng mô tả hành vi.
- Test `ipkit`: bỏ assert thời gian wall-clock để kết quả ổn định.

### Fixed

- Forensics: abstain block khi texture không đủ; cancel cooperative; splicing cue thiếu → inconclusive.
- Evaluation: loại trừ ground truth không dùng được; tách study theo scene gốc; enhancement không ghi đè tập nguồn.
- CLI eval: bắt buộc ít nhất một phương thức; từ chối phương thức lạ hoặc trùng, không in traceback.
- Enhancement: xác minh digest ảnh cha trước khi ghi file `.view` dẫn xuất.
- UI: đồng bộ canvas/overlay; xóa overlay stale; export enhance đặt tên đủ file; argv/drag-drop không giữ progress cũ; nhãn overlay được bản địa hóa và báo enhancement bị hủy.
- UI: cột lịch sử co theo nội dung, tiêu đề được bản địa hóa; tên trợ năng của checkbox phương pháp không trùng nhau và đường dẫn có khoảng trắng được che.
- `ipkit`: yêu cầu phần tử cấu trúc nhị phân; script hình báo cáo không ghi PNG vượt giới hạn kích thước.
- Runtime/PyInstaller: đường dẫn resource standalone; báo lỗi build rõ hơn.

### Documentation

- Ghi nhận study enhancement N=24 (cổ điển); NAFNet arm NOT_RUN trên cohort dễ (`docs/enhancement-study.md`).
- Tài liệu ingest HTTP, frozen comparison studies, giới hạn đo trên README; làm rõ file ingest bị hoãn và các trường báo cáo dễ biến động.
- Thêm `CONTRIBUTING.md`, `SECURITY.md`, mẫu GitHub issue/PR, `docs/limitations.md`.
- Thêm kịch bản demo trực tiếp kèm Q&A cho giảng viên, dàn ý slide trình bày và bảng thuật ngữ ánh xạ tới code.
- Thêm `docs/troubleshooting.md` cho Windows và bản tóm tắt tiếng Anh (`docs/README.en.md`).
- Thêm bản đồ chủ đề môn học (`docs/course-coverage.md`), hợp đồng JSON `ForensicReport` (`docs/forensic-report-schema.md`) và tài liệu hiệu năng (`docs/performance.md`) kèm runtime đo được.
- Thêm phân tích fusion offline trên artifact đã lưu (`docs/fusion-offline-analysis.md`).
- Thêm bản nháp báo cáo capstone chương 1–7 và abstract, cùng hình sinh tái lập được (`docs/report/`).
- Tách bằng chứng CASIA chính thức khỏi hình khám phá trong so sánh; nêu số block chính xác trên ảnh đã chỉnh.
- Hiệu đính tài liệu thuật toán/bảo mật theo code; sửa lệnh không khả chuyển và đường dẫn máy host trong tài liệu và test.
- Căn số liệu ModelForge/CoMoFoD theo artifact tái lập; cập nhật README và hướng dẫn theo code và giới hạn đã đo.

### Tests

- End-to-end CLI: analyze, ingest và eval.
- End-to-end UI: analyze, export và history.
- Hợp đồng help/exit code của CLI; eval thiếu phương thức, phương thức lạ và trùng.
- Biên threshold loader và metric đánh giá; drift test cho tài liệu schema `ForensicReport`.
- Phủ thêm service/core chưa test: ingest, custody log, artifact store, eval runner, ModelBench, runtime paths, stats/settings screens.
- GUI: robustness với input rỗng/dài/thù địch, tên trợ năng/che đường dẫn, bảng lịch sử, polish vòng 2.
- `ipkit`: validation và biên morphology/noise/threshold.
- Script: smoke test benchmark, validation sinh hình báo cáo, tổng hợp fusion offline, build script.
