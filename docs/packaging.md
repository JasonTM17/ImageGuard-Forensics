# Packaging — ImageGuard.exe

## Cách 1 — khuyến nghị (dev/demo nhanh)

```bash
python -m imageguard
```

Cần `.venv` đã `pip install -e ".[dev]"` (xem `docs/environment.md`).

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

```powershell
dist\ImageGuard\ImageGuard.exe   # mở app, chạy 1 phân tích mẫu
```

## Giới hạn đã biết

- Bản build kèm toàn bộ PySide6/Qt (~300MB giải nén) — bình thường cho
  Qt desktop.
- PyInstaller 6.22.x + Python 3.14: verified tại thời điểm build; nếu lỗi
  lạ → fallback cách 1.
