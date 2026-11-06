# Kiến trúc ImageGuard Forensics

## Layer diagram

```
┌─────────────────────────────────────────────┐
│ UI (PySide6 Qt Widgets) — Phase 7           │
│  QGraphicsView viewer · worker bridge       │
├─────────────────────────────────────────────┤
│ Services — Phase 6                          │
│  AnalysisService · HistoryService · EvalSvc │
├─────────────────────────────────────────────┤
│ Forensics engine (Qt-free, headless)        │
│  methods/  copy_move_keypoint · block · ela │
│  fusion/   verdict adjudication             │
├─────────────────────────────────────────────┤
│ Core domain                                 │
│  models · enums · errors · limits · config  │
│  io (safe_open) · metadata · preprocess     │
├─────────────────────────────────────────────┤
│ Evaluation — synthforge · metrics · CoMoFoD │
│ Persistence — SQLite + artifact store       │
└─────────────────────────────────────────────┘
```

## Nguyên tắc bất biến

1. **Engine Qt-free** — `src/imageguard/core/`, `forensics/`, `evaluation/` không import PySide6. Test headless bằng pytest.
2. **GUI thread sở hữu widget** — worker chỉ trả `AlgorithmResult`/`ForensicReport` qua signal; không chạm widget.
3. **Ngưỡng tập trung** — mọi constant thuật toán nằm trong `configs/thresholds.yaml` với `source` giải thích nguồn.
4. **Không overclaim** — Verdict 5 trạng thái; ELA/EXIF/noise là exploratory; report có disclaimer + `score_definition`.
5. **Đường dẫn tương đối** — artifact/history lưu relative path + basename; tuyệt đối không lưu absolute path.

## Data flow (phân tích một ảnh)

```
file → safe_open (magic→caps→decode→exif_transpose) → ImageDocument
     → preprocess per-method → method.analyze(gray, cfg) → AlgorithmResult[]
     → fusion.adjudicate() → ForensicReport → artifacts/{report_id}/ + SQLite
```

Folder ingest lives outside `forensics/` and `core/`; it waits for finished
still images, then calls the headless `AnalysisService` and appends custody
events without changing the analysis or verdict pipeline.

## Folder tree

```
src/imageguard/
  core/       io.py metadata.py preprocess.py models.py errors.py limits.py config.py enums.py
  forensics/  keypoint/ block/ auxiliary/ fusion.py registry.py base.py
  services/   analysis.py history.py eval_svc.py
  storage/    db.py artifacts.py history.py
  ui/         app.py main_window.py screens/ widgets/ workers.py strings.py
  evaluation/ synthforge.py metrics.py runner.py comofod.py modelforge.py modelbench.py robustness.py
configs/thresholds.yaml
tests/
docs/      environment.md architecture.md
data/ artifacts/  (runtime, gitignored)
```
