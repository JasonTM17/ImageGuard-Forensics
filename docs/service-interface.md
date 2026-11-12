# Service Interface (contract cho UI P07)

## AnalysisService

```python
svc.analyze(AnalysisRequest(
    path=Path, methods=["keypoint","block","ela"],
    cancel=threading.Event | None,
    progress_cb=lambda frac, stage: ...,
    include_exif=True,
)) -> ForensicReport | ErrorResult
```

- `methods` mặc định là `DEFAULT_METHODS` = `["keypoint", "block", "ela"]`;
  `include_exif=True` thêm kết quả `audit_exif` vào danh sách method.
- `ForensicReport`: `report_id`, `image_sha256`, `image_name`, `image_width`,
  `image_height`, `created_at`, `tool_version`, `env` (python/opencv/numpy),
  `thresholds_hash`, `overall_verdict` (enum), `methods[]`, `limitations[]`,
  `disclaimer`. Không có field `provenance` gộp — provenance nằm ở các field
  `env`/`thresholds_hash`/`image_sha256`.
- `ErrorResult`: `{code: ErrorCode, message, detail}` — `safe_open` trả về
  thay vì raise; lỗi từng method được cô lập thành `AlgorithmResult` với
  verdict `ERROR`.
- `progress_cb` được gọi từ worker thread — UI phải marshal qua signal.
- Lỗi persistence (DB/artifact) không huỷ kết quả: service thêm cảnh báo vào
  `report.limitations` và vẫn trả `ForensicReport`.

## HistoryService

- `list(limit=100) -> [{id, created_at, verdict, duration_ms, image_name,
  n_methods}]`
- `get(id) -> {id, created_at, verdict, duration_ms, image_name, config_hash,
  methods[{method, verdict, score, score_definition}],
  artifacts[{kind, method, rel_path}]} | None`
- `delete(id) -> bool` — xoá rows (cascade) và gọi
  `ArtifactStore.delete_analysis` để dọn file artifacts. Hiện luôn trả `True`,
  kể cả khi id không tồn tại.
- `stats() -> {n_analyses, by_verdict, avg_duration_ms, label}` — nhãn
  `label` = "thống kê phiên — không phải benchmark", không claim performance
  thuật toán. `avg_duration_ms` có thể là `None` khi chưa có dữ liệu.

## EvaluationService

- `run(dataset, root, split, method_specs, seed=42) -> EvalReport` — dataset
  `"synth"` dùng `SynthAdapter`, còn lại dùng `CoMoFoDAdapter`; ghi
  `reports/eval-<experiment_id>.json` và `.csv` + rows
  `experiments`/`experiment_results`.

## ReportService

- `read_report_json(analysis_id) -> dict | None` — đọc
  `data/artifacts/<analysis_id>/report.json`; trả `None` nếu thiếu/không đọc được.
- `export_markdown(analysis_id, out_path=None) -> Path | None` — render
  Markdown từ `report.json`, ghi `report.md` vào thư mục artifact và (nếu có)
  bản sao tại `out_path`.
