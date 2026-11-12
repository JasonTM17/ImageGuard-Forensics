# Contributing to ImageGuard Forensics

Cảm ơn bạn quan tâm đóng góp. Đây là đồ án môn **Xử lý ảnh** — công cụ chỉ báo **dấu hiệu** (indications) liên quan copy-move/splice/nén, **không** phán quyết ảnh giả hay nguyên bản.

## Ranh giới khoa học (bắt buộc đọc)

Mọi thay đổi UI, báo cáo, tài liệu và test phải giữ wording: *dấu hiệu / gợi ý điều tra*; không thêm xác suất giả mạo, không enum hoặc nhãn kiểu `FORGED` / `AUTHENTIC`. Chi tiết: [`README.md`](README.md), [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

## Yêu cầu môi trường

- **Python** ≥ 3.13 (CI bắt buộc **3.14**; job `test-py313` là tín hiệu tương thích).
- **Windows** + PowerShell (khuyến nghị, khớp CI `windows-latest`).
- Không thêm dependency vào `pyproject.toml` trừ khi maintainer đồng ý trước.

## Thiết lập (PowerShell)

```powershell
cd <thư-mục-repo>
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"

# Kiểm tra môi trường
.\.venv\Scripts\python.exe scripts\verify_env.py
```

Gọi trực tiếp `.\.venv\Scripts\python.exe` — không bắt buộc `activate`.

## Chạy ứng dụng

```powershell
# GUI
.\.venv\Scripts\python.exe -m imageguard
# hoặc
.\run_app.bat

# CLI — stdout là ForensicReport JSON
.\.venv\Scripts\python.exe -m imageguard analyze <đường-dẫn-ảnh> `
  [--methods keypoint,block,ela] [--no-persist]
```

Hướng dẫn GUI/CLI: [`docs/user-guide.md`](docs/user-guide.md).

## Chạy test

pytest dùng **offscreen** Qt (bắt buộc trên Windows headless/CI):

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = '<thư-mục-repo>\src'
& '.\.venv\Scripts\python.exe' -m pytest -x -q -p no:cacheprovider
```

Sau khi `pip install -e .`, có thể bỏ `PYTHONPATH` và chạy từ root repo:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -x -q -p no:cacheprovider
```

Chiến lược test: [`docs/testing-strategy.md`](docs/testing-strategy.md).

## Ruff

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

CI (`/.github/workflows/ci.yml`) chạy cùng lệnh trên Python 3.14 (job `test`).

## Nhánh và commit

- Tạo nhánh từ `main`, **một PR một chủ đề** (ví dụ: `feat/ui-…`, `docs/…`, `fix/forensics-…`).
- **Conventional Commits** trên subject: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`, …
- PR nên mô tả phạm vi, lệnh test/ruff đã chạy, và xác nhận không đổi ranh giới khoa học.

## Cần phê duyệt maintainer (không tự ý sửa trong PR thường)

- `configs/thresholds.yaml` và ngưỡng detector.
- Schema artifact / báo cáo JSON và enum **`Verdict`** đóng.
- `pyproject.toml` (phiên bản, dependency).
- Thay đổi metric đánh giá chính thức hoặc con số đo đã công bố trong README.

## Không commit

- `data/` (runtime DB, artifact, ingest log).
- `models/` (weight ONNX tải riêng).
- Báo cáo nội bộ, `.env*`, secrets, ảnh cá nhân.
- Tài liệu kế hoạch nội bộ hoặc đường dẫn máy host trong file được track.

## Tài liệu thêm

- Kiến trúc: [`docs/architecture.md`](docs/architecture.md)
- Bảo mật đầu vào: [`docs/security-error-handling.md`](docs/security-error-handling.md)
- Giới hạn đã đo: [`docs/limitations.md`](docs/limitations.md)

Mẫu PR/issue: [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md), [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
