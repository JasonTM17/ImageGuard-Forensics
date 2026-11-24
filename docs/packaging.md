# Packaging — ImageGuard.exe

## Cách 1 — khuyến nghị (dev/demo nhanh)

```bash
python -m imageguard
```

Cần `.venv` đã `pip install -e ".[dev]"` (xem
[`docs/environment.md`](environment.md)).

## Cách 2 — PyInstaller onedir

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_app.ps1
# → dist/ImageGuard/ImageGuard.exe
```

- Launcher: `imageguard_gui.py` (root) — PyInstaller cần script file, không
  nhận `-m`.
- `ImageGuard.spec` cấu hình PyInstaller. Dùng `scripts/build_app.ps1` để chọn
  đúng Python và giới hạn `PATH` của lần build vào thư mục hệ thống Windows
  và Python. Lần QA ngày 2026-11-18 phát hiện `icuuc.dll` từ Poppler trên
  `PATH` bị đóng gói nhầm, làm Qt không nạp được. Wrapper ngăn nguồn DLL đó
  lọt vào bundle và phục hồi `PATH` sau khi build.
- `configs/` bundle kèm qua spec — thresholds đọc được trong exe.
- Onedir (không phải onefile): Qt apps start nhanh hơn, dễ debug.
- DB + artifacts tạo runtime tại `data/` cạnh exe — user có thể xóa sạch.

## Verify sau build

CI dùng wrapper và kiểm tra executable khởi động ở chế độ offscreen, chạy
đủ lâu để tạo SQLite mới bên cạnh file exe. Trước khi demo, mở bản vừa build,
nạp ảnh, chạy phân tích rồi kiểm tra overlay, verdict và lịch sử. Script báo
lỗi nếu PyInstaller trả exit code khác 0.

```powershell
dist\ImageGuard\ImageGuard.exe   # mở app, chạy 1 phân tích mẫu
```

## Trạng thái bằng chứng 2026-11-21

- Journey offscreen trên source của commit pin `main @ 5ec332e` **PASS** (analyze keypoint+block+ELA → verdict, persist, reopen, export Markdown thành công; số liệu chi tiết trong receipt cục bộ tại `plans/261121-2135-.../reports/packaged-runtime-261121/`, gitignored) và UI smoke pytest-qt offscreen **PASS / exit 0** (số test chạy được ghi trong receipt).
- Build stock exe từ `5ec332e` trong lần chạy này **BLOCKED** do preflight RAM free < 4 GiB (4 lần lấy mẫu đều dưới ngưỡng) — chưa có startup smoke trên exe mới ở lần này; CI `package-smoke` vẫn chỉ là signal `continue-on-error`.
- Native GUI journey (có màn hình) trên exe vẫn là **NOT_RUN** (chưa được cấp Computer Use). Không claim "packaged-ready" vượt quá các bằng chứng trên; khởi động trên máy sạch vẫn cần kiểm tra riêng.

## Trạng thái bằng chứng 2026-11-24

- Build stock exe từ `main @ f06d76e` (PyInstaller 6.22.3, wrapper `scripts/build_app.ps1`, dir cô lập) **PASS** — preflight RAM free 8,19 GiB (≥ 4 GiB), build ~105 s, exit 0. Exe SHA-256 `1F365EA8…8FF4C2` (receipt cục bộ `plans/261121-2135…/reports/packaged-runtime-261124-f06d76e/`, gitignored).
- Startup smoke trên exe thật offscreen **PASS**: DB `data\imageguard.db` được tạo cạnh exe sau 1,1 s (≤ 30 s), không exit sớm.
- Exe offscreen nhận ảnh fixture qua argv **PASS**: process alive ≥ 20 s rồi dừng sạch.
- Journey offscreen (source, cùng commit; analyze keypoint+block+ela → persist → reopen → export Markdown) **PASS**: verdict `suspicious_indication`, 66 dòng Markdown, `report.json` artifact tồn tại; UI smoke pytest-qt offscreen **PASS** (145 passed). `configs/thresholds.yaml` SHA-256 trùng pin `3D7948F5…CE23` (ngưỡng không đổi).
- Lần 2026-11-21 build bị BLOCKED duy nhất do RAM free < 4 GiB; lần này cùng rule đã PASS.
- Vẫn giữ nguyên các giới hạn: native GUI journey (có màn hình) trên exe **NOT_RUN** (cần Computer Use); job CI `package-smoke` chỉ là signal `continue-on-error`; **không claim "packaged-ready"** vượt bằng chứng; khởi động trên máy sạch chưa kiểm.

## Giới hạn đã biết

- Bản build kèm toàn bộ PySide6/Qt (~410 MB giải nén, đo 2026-11-16) —
  bình thường cho Qt desktop; thêm ~90 MB nữa nếu kèm weight NAFNet.
- Lần QA ngày 2026-11-16: PyInstaller 6.22.3 + Python 3.13.12
  (Windows Server 2022) build thành công và `dist/ImageGuard/ImageGuard.exe`
  chạy trọn luồng: mở GUI, nạp PNG, phân tích keypoint+ELA đến 100%,
  banner verdict và thẻ kết quả hiển thị đúng. Kết quả này áp dụng cho bundle
  và môi trường của lần QA đó. Nếu gặp lỗi import `QtWidgets`, build lại bằng
  wrapper ở trên hoặc chạy từ source trong lúc kiểm tra môi trường.
- **NAFNet trong bản đóng gói** (đã xác minh 2026-11-16): build với extra
  `lama` — `pip install -e ".[dev,lama]"` trước khi chạy PyInstaller — để
  spec gói kèm `onnxruntime`; copy thư mục `models/` chứa
  `deblurring_nafnet_2025may.onnx` vào cạnh `ImageGuard.exe` (weight không
  đi trong spec/git vì ~92 MB, tải từ OpenCV Zoo, MIT License). Exe nhận
  weight, chạy deblur và xuất PNG kèm sidecar provenance ghi `method:
  nafnet` + `model_sha256` đúng pin. Bundle đã mang sẵn VC++ runtime
  (`msvcp140*`, `vcruntime140*`) từ PySide6. Khởi động trên máy sạch vẫn cần
  kiểm tra riêng;
  môi trường dev cài `lama` trên Windows cũ thiếu redist có thể cần cập
  nhật VC++ 2015-2022 Redistributable. Ba operator cổ điển (unsharp, CLAHE,
  non-local means) không cần weight này.
