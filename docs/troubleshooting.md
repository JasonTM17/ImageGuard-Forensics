# Khắc phục sự cố (Windows) — ImageGuard Forensics

Hướng dẫn dành sinh viên chạy đồ án trên **Windows**. Các lệnh ví dụ dùng placeholder `<repo>` thay cho thư mục gốc clone (không ghi đường dẫn máy host vào tài liệu).

## 1. `pip` / virtualenv (`.venv`)

### Triệu chứng thường gặp

- `python` không tìm thấy hoặc trỏ sai bản Python (cần **≥ 3.13**, phát triển trên 3.14 — xem `README.md`).
- `pip install` báo lỗi quyền, SSL, hoặc không cài được `opencv-python` / `PySide6`.
- Chạy `python -m imageguard` nhưng import `imageguard` thất bại (chưa cài editable package).

### Cách làm đúng (theo `README.md`)

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\verify_env.py
```

- **Không bắt buộc** `activate` venv: luôn gọi trực tiếp `.\.venv\Scripts\python.exe` (hoặc `run_app.bat`).
- Khi chạy test/script ngoài thư mục gốc, đặt `PYTHONPATH` trỏ `src`:

```powershell
$env:PYTHONPATH = '<repo>\src'
& '<repo>\.venv\Scripts\python.exe' -m pytest ...
```

### Gợi ý xử lý nhanh

| Vấn đề | Hướng xử lý |
| --- | --- |
| Sai interpreter trong IDE | Chọn Python tại `<repo>\.venv\Scripts\python.exe`. |
| `verify_env.py` báo thiếu gói | Chạy lại `pip install -e ".[dev]"` **trong** venv đó; không trộn pip global. |
| Đổi nhánh / pull mới | Cài lại editable: `pip install -e ".[dev]"`. |
| Nhiều bản OpenCV cài chồng | Chỉ dùng một venv; `pip uninstall opencv-python opencv-contrib-python` rồi cài lại qua `pyproject.toml` của dự án. |

**Lệnh kiểm tra môi trường trong phiên làm tài liệu:** `scripts\verify_env.py` — **NOT_RUN** (sinh viên tự chạy trên máy mình; số phiên bản tham chiếu `docs/environment.md`).

---

## 2. Lỗi Qt platform plugin (`windows` vs `offscreen`)

### Khi nào dùng gì

| Mục đích | `QT_QPA_PLATFORM` | Ghi chú |
| --- | --- | --- |
| GUI desktop (dùng app thật) | **Không đặt** hoặc mặc định **windows** | Cần màn hình / session Windows bình thường. |
| `pytest` + UI (`pytest-qt`) | **offscreen** | Theo `docs/testing-strategy.md`. |
| Script chụp figure GUI có chữ Việt | **Không** dùng offscreen | `docs/report/figures.md` — offscreen thiếu font hệ thống. |

### Triệu chứng

- GUI: *"This application failed to start because no Qt platform plugin could be initialized"* — thường do thiếu DLL Qt, PySide6 lỗi cài, hoặc biến môi trường ép sai platform.
- Test pass nhưng GUI fail (hoặc ngược lại): bạn đang để `QT_QPA_PLATFORM=offscreen` trong profile PowerShell rồi mở GUI.

### Khắc phục

```powershell
# Mở GUI: xóa override offscreen
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m imageguard
```

```powershell
# Chạy test UI
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = '<repo>\src'
& '<repo>\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

Smoke GUI tối thiểu (e2e trong repo): subprocess với `QT_QPA_PLATFORM=offscreen` — process GUI **không** thoát ngay trong vài giây đầu (`tests/e2e/test_cli_process.py`). **Không chạy lại smoke đó trong tài liệu này** — **NOT_RUN**.

Nếu import GUI thất bại (thiếu PySide6), CLI in:

`GUI khong kha dung (...); dung: python -m imageguard eval --help`

và **thoát mã 0** (không `sys.exit(1)`) — vẫn dùng được `eval` / `analyze` / `ingest`.

---

## 3. SIFT (`cv2.SIFT_create`)

Copy-move keypoint dùng SIFT khi runtime có `cv2.SIFT_create` (patent OpenCV đã hết hạn; gói `opencv-python` chính).

**Đã chạy khi soạn tài liệu:**

```powershell
& '<python>' -c "import cv2; print('opencv', cv2.__version__); print('SIFT_create', hasattr(cv2, 'SIFT_create'))"
```

Kết quả quan sát: `opencv 5.0.0`, `SIFT_create True` (khớp `docs/environment.md`: SIFT: yes).

Nếu `SIFT_create False`:

- Kiểm tra không dùng bản OpenCV cắt tính năng; cài lại qua `pip install -e ".[dev]"` trong venv dự án.
- Chạy `pytest tests/test_smoke.py::test_opencv_sift_available` trên máy bạn — **NOT_RUN** trong phiên soạn doc.

ORB vẫn là fallback trong backend; thiếu SIFT không làm crash import nhưng ảnh hưởng hành vi detector (xem `docs/algorithms/02-keypoint.md`).

---

## 4. Chữ Việt hiện ô vuông (tofu) / thiếu glyph

- **GUI:** Qt lấy font hệ thống. Trên Windows nên có **Segoe UI** hoặc **Tahoma**. Script báo cáo thử lần lượt `Segoe UI`, `Tahoma`, `Arial` (`scripts/make_report_figures.py`).
- **offscreen + pytest:** không đại diện cho font GUI thật — đừng kết luận lỗi font chỉ từ test headless.
- **Console PowerShell:** CLI cấu hình stdout/stderr UTF-8 (`cli.py`); nếu terminal cũ (cp1252) vẫn có thể hỏng ký tự khi copy log — ưu tiên Windows Terminal hoặc `chcp 65001`.
- **matplotlib/OpenCV figure:** dùng font có glyph Latin + Việt; nếu grab GUI cho báo cáo, làm theo `docs/report/figures.md` (platform **windows**, không offscreen).

---

## 5. Phân tích rất chậm trên ảnh lớn (keypoint)

Số liệu **chỉ** lấy từ `docs/performance.md` (một máy Windows, Python 3.14.2, OpenCV 5.0.0, `cv2.getNumThreads()` = 16).

**Trung vị thời gian `median_s` (giây) — method `keypoint`:**

| Cạnh ảnh (px) | `median_s` |
| --- | --- |
| 256 | 0.028280600003199652 |
| 512 | 0.14604559999861522 |
| 1024 | 1.6158136000012746 |
| 2048 | 47.603493499998876 |

**Pipeline mặc định `analysis_service` (keypoint + block + ela + EXIF, không persist):**

| Cạnh (px) | `median_s` |
| --- | --- |
| 2048 | 51.22097900000517 |

Nhận xét (từ cùng tài liệu): tại **2048 px**, `keypoint` `median_s` **47.603493499998876 s** lớn hơn mọi method khác trong bảng; `block` đứng thứ hai với **6.8929092999969725 s**. Tỷ lệ tăng `median_s` keypoint: 256→512 ×5.16; 512→1024 ×11.06; 1024→2048 ×29.46.

**Việc có thể làm (không đổi code detector):**

- Thu nhỏ ảnh bản sao trước khi demo, hoặc bật `--no-persist` khi thử CLI để tránh I/O DB.
- Giảm method: `analyze ... --methods ela` hoặc `block` khi không cần keypoint.
- Biết giới hạn an toàn decode: `MAX_PIXELS`, `MAX_DECODE_SIDE`, `AUX_MAX_SIDE` — `docs/security-error-handling.md` (ảnh quá lớn có thể bị từ chối với mã `TOO_MANY_PIXELS` / `FILE_TOO_LARGE`, không phải “chậm”).

Tái lập benchmark: `scripts/bench_methods.py` như trong `docs/performance.md` — **NOT_RUN** trong phiên soạn doc.

---

## 6. Mã thoát CLI và `ErrorResult.code`

### Mã thoát tiến trình (quan sát / khóa trong `tests/test_cli_help_contract.py`)

| Tình huống | Mã thoát |
| --- | --- |
| `analyze` thành công (`ForensicReport` JSON stdout) | **0** |
| `analyze` lỗi mở/đọc ảnh (`ErrorResult` JSON stdout) | **1** |
| `analyze` kiểu kết quả bất ngờ | **1** |
| argparse: thiếu đối số, lựa chọn sai (`eval` không `--dataset`) | **2** |
| `eval`: method không hợp lệ / trùng tên (`cli.py`) | **2** |
| `ingest`: config hoặc thư mục watch không hợp lệ | **2** |
| `ingest`: lỗi setup / scan / watch | **1** |
| `ingest once` thành công; `watch` dừng bằng Ctrl+C | **0** |

**Đã chạy:** `python -m imageguard eval` không đủ đối số → thoát **2**, stderr có `the following arguments are required: --dataset, --root`.

`analyze` in JSON lỗi ra **stdout** (không traceback nếu pipeline đúng thiết kế) — contract test `test_analyze_bad_input_json_error_exit_1_no_traceback`.

### Bảng `ErrorCode` (`src/imageguard/core/errors.py`)

| Mã | Ý nghĩa ngắn |
| --- | --- |
| `PATH_INVALID` | Không tồn tại / không phải file |
| `FILE_UNREADABLE` | Không đọc được bytes (quyền, khóa file) |
| `UNSUPPORTED_FORMAT` | Magic bytes không khớp định dạng hỗ trợ |
| `FILE_TOO_LARGE` | Vượt `MAX_FILE_BYTES` |
| `TOO_MANY_PIXELS` | Header vượt `MAX_PIXELS` (image bomb) |
| `IMAGE_TOO_SMALL` | Cạnh &lt; `MIN_SIDE` |
| `DECODE_FAILED` | Decoder không giải mã (hỏng/cắt cụt) |
| `INTERNAL` | Lỗi không phân loại |

Chi tiết mitigations: `docs/security-error-handling.md`.

---

## 7. File 0 byte hoặc không phải ảnh

Thử trong thư mục scratch (đã **thực thi** khi soạn doc; đường dẫn host được thay bằng tên file tương đối).

```powershell
$env:PYTHONPATH = '<repo>\src'
cd <scratch>
# empty.bin: file 0 byte
& '<python>' -m imageguard analyze empty.bin --no-persist
# not_image.txt: nội dung ASCII "hello"
& '<python>' -m imageguard analyze not_image.txt --no-persist
# missing.png: không tồn tại
& '<python>' -m imageguard analyze missing.png --no-persist
```

### Kết quả quan sát

**`empty.bin` — exit 1, stdout:**

```json
{
 "code": "UNSUPPORTED_FORMAT",
 "message": "Magic bytes không khớp định dạng hỗ trợ (jpeg/png/bmp/tiff/webp)",
 "detail": {
  "head": ""
 }
}
```

**`not_image.txt` — exit 1, stdout:**

```json
{
 "code": "UNSUPPORTED_FORMAT",
 "message": "Magic bytes không khớp định dạng hỗ trợ (jpeg/png/bmp/tiff/webp)",
 "detail": {
  "head": "68656c6c6f"
 }
}
```

**`missing.png` — exit 1, stdout:**

```json
{
 "code": "PATH_INVALID",
 "message": "Không phải file tồn tại: missing.png",
 "detail": {}
}
```

Trong GUI, file rỗng / quá nhỏ: tab Analyze tắt nút chạy và hiện lỗi — không crash (`tests/test_ui_robustness.py`).

---

## 8. Thư mục `data/` và database

Theo `docs/database.md`:

- SQLite: **`data/imageguard.db`** (gitignored), tạo khi persist phân tích.
- Artifact: **`data/artifacts/<analysis_id>/`** (`*_mask.png`, `*_heat.png`, `report.json`).
- CLI `analyze` **không** ghi DB khi có `--no-persist`; mặc định dùng `Path("data") / "imageguard.db"` từ **thư mục làm việc hiện tại** (`cli.py`).
- Ingest: `data_dir` trong YAML (mặc định `data`) — ví dụ `docs/examples/ingest.example.yaml`; custody log `data/ingest/custody.jsonl`.
- Chạy từ thư mục khác → DB/artifact xuất hiện ở `cwd` đó, không nhất thiết cạnh source tree. Luôn `cd <repo>` trước khi chạy GUI/CLI nếu muốn DB nằm trong repo.
- Schema lệch version: `make_session_factory` báo lỗi — recreate `data/imageguard.db` theo `docs/database.md`.

---

## 9. Chạy test với Qt offscreen

Theo `docs/testing-strategy.md`:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = '<repo>\src'
& '<python>' -m pytest -x -q -p no:cacheprovider
```

Nhiều file test gọi `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` — đặt biến trước khi chạy vẫn là thói quen tốt.

CI (GitHub Actions, `windows-latest`): `ruff check src tests scripts` rồi `pytest -q` với `QT_QPA_PLATFORM=offscreen` — xem tiếp `docs/testing-strategy.md`.

**Trong phiên soạn tài liệu này:** full pytest suite — **NOT_RUN** (chỉ thêm markdown).

---

## 10. Bảng tra nhanh

| Triệu chứng | Hướng đi |
| --- | --- |
| Import `imageguard` fail | `pip install -e ".[dev]"`, `PYTHONPATH=src` |
| GUI Qt plugin | Bỏ `QT_QPA_PLATFORM=offscreen`; cài lại PySide6 trong venv |
| Test UI fail, GUI OK | Thêm `QT_QPA_PLATFORM=offscreen` |
| Không có SIFT | OpenCV sai bản; `verify_env.py` |
| Ảnh 4K+ phân tích &gt; ~47 s keypoint | Kỳ vọng theo `docs/performance.md`; giảm size/method |
| `analyze` exit 1 + JSON `code` | Đọc bảng `ErrorCode` §6 |
| `eval` exit 2 | Thiếu/sai tham số dòng lệnh |
| DB “mất” | Kiểm tra `cwd` — có thể tạo `data/` ở chỗ khác |

---

*Tài liệu đi kèm nhánh `docs/troubleshooting`; không thay thế `docs/user-guide.md` hay `README.md`.*
