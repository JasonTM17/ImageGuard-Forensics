# UI Design — ImageGuard Desktop (PySide6)

## Nguyên tắc

1. **Engine Qt-free**: `imageguard.forensics/*` không import Qt. UI gọi
   service (`AnalysisService`, `HistoryService`, `ReportService`) và engine
   tăng cường Qt-free (`imageguard.enhancement.derive`) — thay UI toolkit
   không đụng engine. Tab Tăng cường gọi `derive` (Qt-free, không DB) thay vì
   `AnalysisService`.
2. **UI thread không bao giờ chạy forensics**: mọi phân tích qua
   `QThreadPool` (maxThreadCount=1 — serialize để giới hạn RAM) +
   `QRunnable`. Rule cứng: worker không chạm QWidget; dữ liệu đi qua signals.
3. **Cancel là cooperative**: `threading.Event`, engine check giữa các stage
   (không kill thread — kill thread giữa native OpenCV call = crash).
4. **QImage phải `.copy()`**: QImage giữ pointer vào buffer ndarray; ndarray
   bị GC → dangling pointer → crash ngẫu nhiên. Mọi bridge numpy→Qt đều copy.
5. **Wording trung thực**: phần lớn nhãn/text UI tập trung `ui/strings.py`
   (một số thông báo dialog viết trực tiếp trong screen) — "dấu hiệu" không
   "kết luận"; verdict không dùng đỏ/xanh tuyệt đối (đỏ = phán có tội, xanh =
   minh oan — cả hai overclaim). Test `test_strings_no_forbidden_wording`
   quét các hằng chuỗi in hoa trong `strings.py`.
6. **Disclaimer nằm trong báo cáo**: câu lưu ý (dấu hiệu, không phải kết luận)
   đi kèm `ForensicReport` và báo cáo Markdown; cửa sổ chính không có thanh
   disclaimer riêng.

## Cấu trúc shell — 5 tab

```
ui/
  app.py           QApplication + build_services() wire DI
  main_window.py   QTabWidget: Analyze | Enhance | History | Stats | Settings
  strings.py       nhãn/text UI chính (integrity-tested)
  workers.py       WorkerSignals + AnalysisWorker; EnhanceSignals + EnhanceWorker
  widgets/
    image_canvas.py    QGraphicsView: zoom(wheel)/pan(drag)/fit + overlay alpha
    verdict_panel.py   VerdictBanner + MethodCard
  screens/
    analyze.py     open(drag-drop/picker) → preview+EXIF → methods → run/cancel
                   → Markdown export sau khi có ForensicReport
    enhance.py     open → operator+params → run/cancel → blend slider → export PNG
    history.py     bảng analyses từ SQLite → reopen/delete/export Markdown
    stats.py       tổng hợp thống kê phiên từ HistoryService
    settings.py    thresholds tree viewer (key|value|nguồn)
```

Thứ tự tab (theo `main_window.py`): Phân tích → Tăng cường → Lịch sử →
Thống kê → Cài đặt. Smoke test: `tests/test_ui_smoke.py` assert `tabs.count() == 5`.

## Data flow

```
AnalyzeScreen._run()
  → AnalysisRequest(path, methods)  (AnalysisWorker tự tạo cancel Event)
  → AnalysisWorker(QRunnable) ──pool──▶ thread
       │  signals.progress(frac, stage)  ← progress_cb từ service
       │  signals.result(ForensicReport|ErrorResult)
       ▼
  _on_result(): banner(verdict+signals) + MethodCard/method
               + overlay dropdown (mask đỏ blend / heatmap colormap)
               + hiện nút Xuất báo cáo (Markdown) → ReportService
```

- `ErrorResult` từ `safe_open` hiển thị trong `lbl_err` — KHÔNG dialog crash.
- Method `NOT_APPLICABLE` (ELA trên PNG) vẫn render `MethodCard` kèm limitation.
- `AnalysisService` persist history; `HistoryService` list/get/delete, delete
  gỡ artifacts trên đĩa.

## Enhance tab

`EnhanceScreen` (`screens/enhance.py`) chỉ để xem: Unsharp mask / CLAHE /
Non-local means (+ NAFNet tùy chọn). Không gọi `AnalysisService`, không ghi
database, không kết luận. `EnhanceWorker` (`workers.py`) chạy trên `make_pool()`
(maxThreadCount=1), gọi `imageguard.enhancement.derive.derive` (Qt-free,
database-free) và check `threading.Event` trước khi emit `result`;
`finished` luôn emit để UI dựng lại trạng thái kể cả khi cancel.

- View: ảnh gốc là base pixmap, ảnh tăng cường là overlay; slider 0–100 chỉ
  đổi opacity của overlay — không ghi pixel, không chạy lại operator.
- Banner `strings.ENHANCE_BANNER` cố định, luôn hiện: ảnh tăng cường là
  bản xem, không phải bằng chứng; phân tích dấu hiệu chỉ gắn với file gốc.
- Xuất: PNG + `.json` sibling (`derive.write_sibling`) ghi `derived`,
  `parent_sha256`/`derived_sha256`/`method`/`params`/`model_sha256` (null với
  operator cổ điển, hash ONNX với NAFNet)/`created_utc`; đích trùng file gốc bị
  từ chối, đích đã tồn tại phải xác nhận ghi đè.

## Stats tab

`StatsScreen` đọc aggregate từ `HistoryService` (số lần phân tích, verdict
phân bố trong DB cục bộ) — không chạy detector, không thay verdict đã lưu.

## Overlay rendering

- **Mask**: `bgr[mask>0] = (0,0,255)` blend 0.6/0.4 với gốc.
- **Heatmap**: `render_heatmap` (colormap) resize về base, alpha=0.55.
- **Pairs** (keypoint): ảnh `pairs_image` làm overlay, alpha=1.0.

## Tests

- `test_ui_smoke.py` — offscreen QPA: shell **5 tab**, disclaimer hiện diện,
  open-preview thành công, file hỏng → lỗi thân thiện, history rỗng,
  wording integrity.
- `test_worker_contract.py` — result/progress signals, cancel cooperative,
  pool serialize (order [0,1]), file corrupt → result signal không failed.
- `test_enhance_derive.py` — engine: JSON sibling keys, `model_sha256` null,
  no absolute path, source-path refusal.
