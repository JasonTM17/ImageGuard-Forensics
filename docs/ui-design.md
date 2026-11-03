# UI Design — ImageGuard Desktop (PySide6)

## Nguyên tắc

1. **Engine Qt-free**: `imageguard.forensics/*` không import Qt. UI chỉ gọi
   `AnalysisService`/`HistoryService` — thay UI toolkit không đụng engine.
2. **UI thread không bao giờ chạy forensics**: mọi phân tích qua
   `QThreadPool` (maxThreadCount=1 — serialize để giới hạn RAM) +
   `QRunnable`. Rule cứng: worker không chạm QWidget; dữ liệu đi qua signals.
3. **Cancel là cooperative**: `threading.Event`, engine check giữa các stage
   (không kill thread — kill thread giữa native OpenCV call = crash).
4. **QImage phải `.copy()`**: QImage giữ pointer vào buffer ndarray; ndarray
   bị GC → dangling pointer → crash ngẫu nhiên. Mọi bridge numpy→Qt đều copy.
5. **Wording trung tính**: mọi text tập trung `ui/strings.py` —
   "dấu hiệu" không "kết luận"; verdict không dùng đỏ/xanh tuyệt đối
   (đỏ = phán có tội, xanh = minh oan — cả hai overclaim). Test
   `test_strings_no_forbidden_wording` enforce.
6. **Disclaimer cố định**: `DisclaimerBar` dưới cùng mọi màn hình, không
   ẩn được.

## Cấu trúc

```
ui/
  app.py           QApplication + build_services() wire DI
  main_window.py   QTabWidget: Analyze / History / Settings + DisclaimerBar
  strings.py       toàn bộ text VN (integrity-tested)
  workers.py       WorkerSignals{progress,result,failed} + AnalysisWorker
  widgets/
    image_canvas.py    QGraphicsView: zoom(wheel)/pan(drag)/fit + overlay alpha
    verdict_panel.py   VerdictBanner + MethodCard + DisclaimerBar
  screens/
    analyze.py     open(drag-drop/picker) → preview+EXIF → methods → run/cancel
    history.py     bảng analyses từ SQLite → reopen/delete(confirm)
    settings.py    thresholds tree viewer (key|value|nguồn)
```

## Data flow

```
AnalyzeScreen._run()
  → AnalysisRequest(path, methods, cancel=Event)
  → AnalysisWorker(QRunnable) ──pool──▶ thread
       │  signals.progress(frac, stage)  ← progress_cb từ service
       │  signals.result(ForensicReport|ErrorResult)
       ▼
  _on_result(): banner(verdict+signals) + MethodCard/method
               + overlay dropdown (mask đỏ blend / heatmap colormap)
```

- `ErrorResult` từ `safe_open` hiển thị trong `lbl_err` — KHÔNG dialog crash.
- Method `NOT_APPLICABLE` (ELA trên PNG) vẫn render card xám kèm limitation.
- History persist qua Phase-6 `HistoryService`; delete gỡ artifacts trên đĩa.

## Overlay rendering

- **Mask**: `bgr[mask>0] = (0,0,255)` blend 0.6/0.4 với gốc.
- **Heatmap**: `render_heatmap` (colormap) resize về base, alpha=0.55.

## Tests

- `test_ui_smoke.py` — offscreen QPA: shell 3 tab, disclaimer hiện diện,
  open-preview thành công, file hỏng → lỗi thân thiện, history rỗng,
  wording integrity.
- `test_worker_contract.py` — result/progress signals, cancel cooperative,
  pool serialize (order [0,1]), file corrupt → result signal không failed.
