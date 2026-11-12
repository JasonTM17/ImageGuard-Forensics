# Kiến trúc ImageGuard Forensics

## Layer diagram

```
┌─────────────────────────────────────────────┐
│ UI (PySide6 Qt Widgets) — Phase 7           │
│  QGraphicsView viewer · worker bridge       │
├─────────────────────────────────────────────┤
│ Services — Phase 6                          │
│  AnalysisService · HistoryService · EvalSvc │
│  ReportService (Markdown export)            │
├─────────────────────────────────────────────┤
│ Forensics engine (Qt-free, headless)        │
│  methods/  copy_move_keypoint · block · ela │
│  localize/ mask fusion · heatmap            │
│  fusion/   verdict adjudication             │
├─────────────────────────────────────────────┤
│ Core domain                                 │
│  models · enums · errors · limits · config  │
│  io (safe_open) · metadata · preprocess     │
├─────────────────────────────────────────────┤
│ Enhancement (Qt-free derive / NAFNet view)  │
│ Ingest (folder watch · custody log)         │
│ Evaluation — synthforge · metrics · studies │
│ Persistence — SQLite + artifact store       │
└─────────────────────────────────────────────┘
```

## Nguyên tắc bất biến

1. **Engine Qt-free** — `src/imageguard/core/`, `forensics/`, `evaluation/`, `enhancement/`, `ingest/` không import PySide6. Test headless bằng pytest.
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

## Folder tree (theo `git ls-files src/imageguard`)

```
src/imageguard/
  __init__.py  __main__.py  cli.py
  core/        io.py metadata.py preprocess.py models.py errors.py
               limits.py config.py enums.py paths.py
  forensics/   base.py registry.py fusion.py
               keypoint/   detectors.py matchers.py pipeline.py …
               block/      dct_blocks.py pipeline.py voting.py …
               auxiliary/  ela.py noise_map.py freq_view.py splicing.py …
               localize/   fusion_masks.py heatmap.py
  enhancement/ derive.py nafnet.py
  ingest/      folder.py commands.py config.py custody.py http_pull.py
  services/    analysis_service.py history_service.py
               evaluation_service.py report_service.py
  storage/     db.py artifacts.py repository.py models.py history.py
  evaluation/  synthforge.py metrics.py runner.py comofod.py datasets.py
               robustness.py modelforge.py modelbench.py
               blind_kernel.py enhance_ops.py enhance_metrics.py
               enhance_study.py enhance_bootstrap.py nafnet_arm.py
               splicing_study.py
  ui/          app.py main_window.py workers.py strings.py
               screens/  analyze.py enhance.py history.py stats.py settings.py
               widgets/  image_canvas.py verdict_panel.py
configs/thresholds.yaml
tests/
docs/      xem README.md — chỉ mục đầy đủ
data/ artifacts/  (runtime, gitignored)
```
