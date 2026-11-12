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
- Spec file `ImageGuard.spec` là nguồn cấu hình DUY NHẤT (`--collect-submodules
  imageguard` + `datas=[('configs','configs')]` tương đương CLI cũ); script
  chỉ gọi `pyinstaller --noconfirm --clean ImageGuard.spec`.
- `configs/` bundle kèm qua spec — thresholds đọc được trong exe.
- Onedir (không phải onefile): Qt apps start nhanh hơn, dễ debug.
- DB + artifacts tạo runtime tại `data/` cạnh exe — user có thể xóa sạch.

## Verify sau build

Build thành công chỉ chứng minh đã tạo bundle. Cần kiểm tra executable khởi
động được, mở ảnh, phân tích xong, hiện overlay/verdict và lưu lịch sử trước
khi dùng làm bản demo. Script dừng với lỗi nếu PyInstaller trả exit code khác 0.

```powershell
dist\ImageGuard\ImageGuard.exe   # mở app, chạy 1 phân tích mẫu
```

## Giới hạn đã biết

- Bản build kèm toàn bộ PySide6/Qt (~300 MB giải nén — số cũ, chưa đo lại
  trong worktree này) — bình thường cho Qt desktop.
- Lần QA ngày 2026-11-04: PyInstaller 6.22.3 + PySide6 6.12.0 build
  thành công với Python 3.14.2 và 3.13.7, nhưng executable mới lỗi import
  `QtWidgets` (DLL procedure not found); Qt smoke tối giản cũng lỗi.
  Đây là bằng chứng lịch sử, chưa có xác nhận bản đóng gói hiện tại chạy
  trọn luồng. Dùng cách 1 khi cần chạy source; không suy ra bản `.exe`
  hoạt động từ kết quả pytest hoặc exit code build.
- Spec hiện chỉ bundle `configs/`, không chứa weight NAFNet. Lựa chọn AI
  cần ONNX Runtime và weight cung cấp riêng theo
  [`docs/user-guide.md`](user-guide.md); chưa xác minh NAFNet trong bản đóng
  gói. Ba operator cổ điển (unsharp, CLAHE, non-local means) không cần weight
  này.
