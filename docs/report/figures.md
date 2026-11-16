# Hình minh họa báo cáo (synthetic)

Tất cả hình trong thư mục này là **ví dụ minh họa** sinh từ ảnh **tổng hợp nội bộ** (NumPy / synthforge). Không dùng CASIA, CoMoFoD hay ảnh cá nhân. Hình **không** chứng minh độ chính xác hay kết luận ảnh bị giả mạo.

## Tái tạo

Từ thư mục gốc repository, với virtualenv dự án và `src` trên `PYTHONPATH`:

```powershell
$env:PYTHONPATH = 'src'
# Không đặt QT_QPA_PLATFORM=offscreen — tab GUI cần nền tảng Windows có font.
python scripts/make_report_figures.py
```

Tab GUI: nền tảng Qt `windows (default)`, font `Segoe UI`.

Seed cố định: `20261112`. Mỗi PNG được nén/resize để ≤ 307200 byte.

## Danh sách hình

### `copymove_synthetic_keypoint.png` — Copy-move synthetic — keypoint

- **Nội dung:** So sánh ảnh gốc và ảnh forge synthetic; cặp keypoint và mask từ KeypointCopyMove (pipeline thật, seed cố định).
- **Cách sinh:** synthforge + KeypointCopyMove; PNG 287320 byte
- **Kích thước file:** 287320 byte

### `copymove_masks_keypoint_vs_block.png` — Mask keypoint vs block-DCT (hình H4)

- **Nội dung:** Ba khung: ảnh forge synthetic, mask union keypoint (SIFT/ORB qua RANSAC), mask shift-voted block-DCT trên cùng ảnh — so sánh vùng định vị hai method.
- **Cách sinh:** synthforge + KeypointCopyMove + BlockCopyMove; PNG 242668 byte
- **Kích thước file:** 242668 byte

### `freq_block_spectrum.png` — Block-DCT và phổ tần số

- **Nội dung:** Ví dụ cue khám phá: heatmap block-DCT và phổ FFT trên cùng ảnh synthetic forge (không khẳng định độ chính xác).
- **Cách sinh:** BlockCopyMove + FreqView; PNG 212359 byte
- **Kích thước file:** 212359 byte

### `enhance_synthetic_blur_compare.png` — Tăng cường xem thử — blur synthetic

- **Nội dung:** Ba khung: ảnh synthetic bị Gaussian blur, unsharp mask và CLAHE (derive thật). Chỉ minh họa giao diện xem, không phải bằng chứng.
- **Cách sinh:** numpy blur + derive; PNG 221927 byte
- **Kích thước file:** 221927 byte

### `gui_tab_analyze.png` — Giao diện — tab Phân tích

- **Nội dung:** Ảnh chụp native 1280×800 sau khi chạy phân tích trên ảnh synthetic forge (mọi method bật).
- **Cách sinh:** Qt windows (default), QFont(Segoe UI), MainWindow 1280×800 grab; PNG 217614 byte
- **Kích thước file:** 217614 byte

### `gui_tab_enhance.png` — Giao diện — tab Tăng cường

- **Nội dung:** Tab xem thử với banner và blend 50% giữa ảnh blur synthetic và CLAHE (không ghi đè file gốc).
- **Cách sinh:** Qt windows (default), QFont(Segoe UI), MainWindow 1280×800 grab; PNG 304124 byte
- **Kích thước file:** 304124 byte

### `gui_tab_history.png` — Giao diện — tab Lịch sử

- **Nội dung:** Ba dòng lịch sử từ phiên phân tích synthetic (AnalysisService, SQLite tạm).
- **Cách sinh:** Qt windows (default), QFont(Segoe UI), MainWindow 1280×800 grab; PNG 25707 byte
- **Kích thước file:** 25707 byte

### `gui_tab_stats.png` — Giao diện — tab Thống kê

- **Nội dung:** Thống kê phiên trên cùng dữ liệu synthetic (không phải benchmark).
- **Cách sinh:** Qt windows (default), QFont(Segoe UI), MainWindow 1280×800 grab; PNG 23034 byte
- **Kích thước file:** 23034 byte

### `gui_tab_settings.png` — Giao diện — tab Cài đặt

- **Nội dung:** Cây ngưỡng đọc từ cấu hình dự án — minh họa layout báo cáo.
- **Cách sinh:** Qt windows (default), QFont(Segoe UI), MainWindow 1280×800 grab; PNG 80205 byte
- **Kích thước file:** 80205 byte

### `gui_tab_analyze_verdict.png` — Giao diện — tab Phân tích từ bản đóng gói

- **Nội dung:** Bản `.exe` PyInstaller chạy thật: phân tích ảnh synthetic copy-move đến 100%, banner "Có dấu hiệu nghi vấn" và thẻ kết quả từng method. Dùng cho hình H3.
- **Cách sinh:** `dist/ImageGuard/ImageGuard.exe` (PyInstaller 6.22.3, Python 3.13.12), chụp màn hình Windows 2026-11-16

### `h1-layer-diagram.svg` — Sơ đồ layer (hình H1)

- **Nội dung:** Stack UI → Services → Forensics engine → Core domain và tầng hỗ trợ (Enhancement/Ingest/Evaluation/Persistence/IPKit), bám `architecture.md`.
- **Cách sinh:** SVG vector viết tay

### `h2-analysis-flow.svg` — Luồng một lần phân tích (hình H2)

- **Nội dung:** file → safe_open → ImageDocument → method.analyze → fusion.combine → ForensicReport → artifacts + SQLite; nhánh ErrorResult.
- **Cách sinh:** SVG vector viết tay

