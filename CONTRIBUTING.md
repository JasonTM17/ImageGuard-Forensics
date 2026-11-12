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

## Đặt tên nhánh

Tạo nhánh từ `main`, **một PR một chủ đề**. Tên nhánh mô tả **rõ chức năng mà nhánh mang lại**:

```text
<type>/<mô-tả-kebab-case-rõ-ràng>
```

`<type>` chỉ thuộc danh sách: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`.

- Mô tả bằng kebab-case, nêu **cái gì được thay đổi** — không dùng id mờ, mã task, số issue hay tên viết tắt không ai đọc hiểu (`feat/abc-123`, `fix/wip`, `docs/update`).
- Ví dụ thật trong repo hiện tại: `docs/readme-english-summary`, `docs/troubleshooting-guide`, `fix/block-vote-direction`, `test/flat-field-noise-sigma-map`, `docs/contributing-branch-naming-and-worktree-workflow`.
- Tiền tố `feature/` cũ (ví dụ `feature/casia-locked-protocol`) không còn dùng — thay bằng `feat/`.
- Nhánh dài tốt hơn nhánh mơ hồ; người review phải đoán được phạm vi chỉ từ tên nhánh.

## Worktree song song (tuỳ chọn)

Khi cần làm nhiều việc cùng lúc mà không trộn thay đổi, dùng `git worktree`: **mỗi worktree một nhánh**, thư mục worktree đặt **ngoài thư mục repo** để không lẫn vào cây nguồn.

```powershell
# Đặt cạnh repo, không nằm trong repo
git worktree add ..\imageguard-wt-<description> -b <type>/<kebab-description>

# Xem và dọn dẹp
git worktree list
git worktree remove ..\imageguard-wt-<description>
```

Quy tắc:

- Mỗi nhánh chỉ thuộc **một** worktree; không checkout cùng nhánh ở hai nơi.
- Chạy test/ruff **trong worktree đó** (đặt `PYTHONPATH` trỏ tới `src` của worktree, không phải repo gốc).
- **Xoá worktree trước khi xoá nhánh** (`git worktree remove` rồi mới `git branch -d`); Git từ chối xoá nhánh đang được checkout trong một worktree khác.

## Kiểm tra cục bộ trước khi commit

Khối lệnh pytest dưới đây là bản sao nguyên văn từ [`docs/testing-strategy.md`](docs/testing-strategy.md) (mục *Chạy*); nguồn chuẩn là tài liệu đó:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = '<repo>\src'
& '<python>' -m pytest -x -q -p no:cacheprovider
```

Ví dụ dùng venv trong repo:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -x -q -p no:cacheprovider
```

Ruff (xem mục [Ruff](#ruff)):

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Gate mỗi phase: full suite xanh; `ruff check` sạch trên phạm vi thay đổi; chỉ merge/push sau khi gate pass trên nhánh feature tương ứng.

## Commit

- **Conventional Commits** trên subject: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`, …
- Subject nêu kết quả thay đổi, không phải tiến trình ("document branch naming", không phải "wip docs").
- Stage đường dẫn cụ thể (`git add <paths>`), không `git add -A` khi worktree còn file rác.
- PR nên mô tả phạm vi, lệnh test/ruff đã chạy, và xác nhận không đổi ranh giới khoa học.

## Cần phê duyệt maintainer (không tự ý sửa trong PR thường)

- `configs/thresholds.yaml` và ngưỡng detector.
- Schema artifact / báo cáo JSON và enum **`Verdict`** đóng.
- `pyproject.toml` (phiên bản, dependency).
- Thay đổi metric đánh giá chính thức hoặc con số đo đã công bố trong README.

## Không commit

Không stage, commit hay push:

- `plans/` — mọi kế hoạch, báo cáo tiến độ, ledger nội bộ.
- Báo cáo, kết quả đo, log, artifact xác minh có dữ liệu chụp được.
- `data/` (runtime DB, artifact, ingest log).
- `models/` (weight ONNX tải riêng).
- `.env*`, secrets, khoá, chứng chỉ, ảnh cá nhân hoặc dữ liệu cá nhân.
- File hướng dẫn cho agent (`AGENTS.md`, `CLAUDE.md`) và thư mục cấu hình agent (`.claude/`, `.agents/`, `.agentkit/`, `.codex/`).
- Đường dẫn máy host (absolute path) trong file được track. Ưu tiên `.git/info/exclude` thay vì sửa `.gitignore` đang track.

## Tài liệu thêm

- Kiến trúc: [`docs/architecture.md`](docs/architecture.md)
- Bảo mật đầu vào: [`docs/security-error-handling.md`](docs/security-error-handling.md)
- Giới hạn đã đo: [`docs/limitations.md`](docs/limitations.md)

Mẫu PR/issue: [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md), [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
