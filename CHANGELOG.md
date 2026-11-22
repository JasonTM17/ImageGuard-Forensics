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

- 2026-11-21: cập nhật ma trận đã chạy thử trong `docs/demo-script.md` (smoke GUI 4 s offscreen đã có receipt cục bộ, lưu ngày 2026-11-21) và thêm mục "Trạng thái bằng chứng 2026-11-21" trong `docs/packaging.md` (journey offscreen PASS trên `main @ 5ec332e`, UI smoke pytest-qt offscreen PASS/exit 0; build exe mới BLOCKED do RAM < 4 GiB — chưa claim packaged-ready). Không đổi mã nguồn, ngưỡng hay schema.
- Đồng bộ README và hướng dẫn runtime với layout code và giới hạn đo CASIA đã sàng lọc; README có mục lục trỏ tới các tài liệu chuyên đề.
- CI: job Python 3.13 không chặn merge (`continue-on-error`).
- Vector hóa một số toán tử `ipkit` (threshold thích ứng, morphology).
- Thay tham chiếu kế hoạch nội bộ trong comment/docstring bằng mô tả hành vi.
- Test `ipkit`: bỏ assert thời gian wall-clock để kết quả ổn định.

### Fixed

- 2026-11-22 (vòng 2):
  - `storage/db.py`: dựng URL SQLite bằng `sqlalchemy.engine.URL.create(...)` (tránh lỗi path chứa `#`/`%`), giải quyết động `data_root()` khi `db_path=None`, giải phóng `Engine` trong nhánh lỗi WAL/schema (`eng.dispose()`) và gắn `weakref.finalize` tránh khóa file Windows; `cli.py` & `ingest/commands.py` đóng `Engine` trong khối `finally`.
  - Runtime đóng gói (`FROZEN=True`): giải quyết động `config_path()` / `data_root()` trong `core/config.py`, `storage/artifacts.py`, `ingest/camera.py`, `ui/widgets/camera_dialog.py`, `ui/widgets/sample_dialog.py`, và `cli.py`.
  - `forensics/localize/heatmap.py` & `services/analysis_service.py`: xử lý an toàn mảng rỗng, không phải `uint8`, `bool`, `NaN`/`Inf`, `(H, W, 1)`, BGRA và kênh `0` trong `render_heatmap`/`overlay_heatmap`/`overlay_mask` và khi xuất artifact `*_heat.png`.
  - `forensics/auxiliary/splicing.py`: trả `mask=None` thay vì mask toàn 0 khi `ela_hot_px == 0` hoặc `overlap_px == 0`.
  - `ui/widgets/image_canvas.py` & `ui/screens/` (`analyze.py`, `enhance.py`, `history.py`): bỏ qua cuộn chuột ngang (`angleDelta().y() == 0`) thay vì zoom nhầm; hỗ trợ BGRA, `(H, W, 1)`, `bool`, `float` có `NaN`/`Inf`, ảnh rỗng và từ chối `complex`/`object` trong `ndarray_to_qpixmap`/`heat_to_qpixmap`; chuẩn hóa kiểm tra kích thước mask/ảnh chi tiết lịch sử.
  - `ipkit/_util.py` & `ipkit/edges.py`: `apply_per_channel` bảo toàn dtype đầu ra (như `float32`) và xử lý trục kênh rỗng; sửa annotation kiểu trả về `tuple[np.ndarray, np.ndarray]` cho `sobel`/`prewitt`.
  - `evaluation/blind_kernel.py`: chỉ nhân 255 cho ảnh float trong `[0, 1]`, tránh làm sai ảnh `uint8` tối (`max == 1`).
  - `ingest/http_pull.py`: đọc phản hồi qua `response.read1` thay vì `response.fp.read1` để giải mã đúng `Transfer-Encoding: chunked` và kết thúc đúng `Content-Length` trên kết nối `Connection: keep-alive`.
  - Đường dẫn Unicode / `cp1252` trên Windows: thay `cv2.imread`/`cv2.imwrite` bằng `Path` bytes + `cv2.imdecode`/`cv2.imencode` trong `evaluation/robustness.py`, `evaluation/synthforge.py`, `scripts/ipkit_demo.py`, và `scripts/make_report_figures.py`; thêm kiểm tra tên file rỗng/`"."` và in UTF-8 an toàn trong `scripts/ipkit_demo.py`.
- 2026-11-22: `heatmap.py`: lọc `NaN`/`Inf` và giá trị âm trong `normalize_map` trước khi chia `max` để tránh tạo bản đồ nhiệt lỗi.
- 2026-11-22: `enhance_ops.py`: bảo toàn trục kênh cuối cho ảnh `H×W×1` trong `_super_res_ibp` (tránh lỗi broadcast khi `GaussianBlur`/`resize` ép 3D xuống 2D), kiểm tra `dtype == uint8` nhất quán ở nhánh xám 2D của `cam_calibrate`/`freq_sharpen` và kẹp `tile` theo kích thước ảnh nhỏ.
- Forensics: abstain block khi texture không đủ; cancel cooperative; splicing cue thiếu → inconclusive.
- Evaluation: loại trừ ground truth không dùng được; tách study theo scene gốc; enhancement không ghi đè tập nguồn.
- CLI eval: bắt buộc ít nhất một phương thức; từ chối phương thức lạ hoặc trùng, không in traceback.
- Enhancement: xác minh digest ảnh cha trước khi ghi file `.view` dẫn xuất.
- UI: đồng bộ canvas/overlay; xóa overlay stale; export enhance đặt tên đủ file; argv/drag-drop không giữ progress cũ; nhãn overlay được bản địa hóa và báo enhancement bị hủy.
- UI: cột lịch sử co theo nội dung, tiêu đề được bản địa hóa; tên trợ năng của checkbox phương pháp không trùng nhau và đường dẫn có khoảng trắng được che.
- `ipkit`: yêu cầu phần tử cấu trúc nhị phân; script hình báo cáo không ghi PNG vượt giới hạn kích thước.
- Runtime/PyInstaller: đường dẫn resource standalone; báo lỗi build rõ hơn.

### Documentation

- 2026-11-22 (vòng 2): cập nhật số liệu kiểm thử mới nhất (**1623 passed, 5 skipped**, 0 warnings, độ phủ **94%** trên `9944` statements) trong `docs/testing-strategy.md`.
- 2026-11-22: bổ sung `deepfake_cues` vào danh sách enum `MethodName` và bảng evidence trong `docs/forensic-report-schema.md`, thêm tóm tắt `deepfake_cues` vào `docs/algorithms.md`, và ghi nhận số liệu độ phủ kiểm thử (92%, 1598 passed) trong `docs/testing-strategy.md`.
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

- 2026-11-22 (vòng 2): bổ sung `tests/test_pixel_and_ui_canvas_fixes.py`, `tests/test_adversarial_fixes_and_coverage.py`, và `tests/test_low_coverage_modules.py` (bao phủ hợp đồng điểm ảnh/kênh, `ImageCanvas` wheel ngang & drag-drop `QPointF`, `splicing` zero-ELA, dynamic frozen paths, dọn `Engine` SQLite, HTTP chunked/keep-alive/SSRF, đường dẫn Unicode/`cp1252`, và các module ít phủ như `nafnet_arm`, `blind_kernel`, `modelforge`, `enhance_study`, `derive`, `folder`, `http_pull`, `cli`); đóng kết nối `sqlite3` trong `test_cli_end_to_end_flow.py` và `test_ingest_folder.py` để đạt `0` warnings.
- 2026-11-22: bổ sung kiểm thử mức điểm ảnh (`tests/test_pixel_overlays.py`: overlay/heatmap math, copy-move downscale/coverage safeguard, ELA tampered patch), custody & determinism khôi phục (`tests/test_restoration_custody.py`), hợp đồng kênh/độ sâu bit của `safe_open` (`tests/test_safe_open_channel_contract.py`), hợp đồng JSON subprocess CLI `analyze --no-persist` (`tests/test_cli_analyze_subprocess.py`), drift lock `MethodName` trong `tests/test_report_schema_doc.py`, edge cases camera/EXIF/ImageCanvas/EnhanceScreen (`tests/test_camera_and_canvas_edges.py`), và phủ `__main__`, `ingest watch`, `nafnet`, `haar_feature`, sanitization `heatmap` (`tests/test_remaining_coverage_edges.py`).
- End-to-end CLI: analyze, ingest và eval.
- End-to-end UI: analyze, export và history.
- Hợp đồng help/exit code của CLI; eval thiếu phương thức, phương thức lạ và trùng.
- Biên threshold loader và metric đánh giá; drift test cho tài liệu schema `ForensicReport`.
- Phủ thêm service/core chưa test: ingest, custody log, artifact store, eval runner, ModelBench, runtime paths, stats/settings screens.
- GUI: robustness với input rỗng/dài/thù địch, tên trợ năng/che đường dẫn, bảng lịch sử, polish vòng 2.
- `ipkit`: validation và biên morphology/noise/threshold.
- Script: smoke test benchmark, validation sinh hình báo cáo, tổng hợp fusion offline, build script.
