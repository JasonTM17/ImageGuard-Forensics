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

- `ForensicReport`: overall_verdict (enum), methods[], limitations[],
  disclaimer, provenance (env, thresholds_hash, image_sha256).
- `ErrorResult`: {code: ErrorCode, message, detail} — KHÔNG raise.
- progress_cb gọi từ worker thread — UI phải marshal qua signal.

## HistoryService

- `list(limit) -> [{id, created_at, verdict, image_name, n_methods}]`
- `get(id) -> {methods, artifacts[{kind, rel_path}], ...} | None`
- `delete(id) -> bool` — xóa cả artifact files
- `stats() -> {n_analyses, by_verdict, avg_duration_ms}` — nhãn "thống kê
  phiên", không claim performance thuật toán.

## EvaluationService

- `run(dataset, root, split, method_specs, seed) -> EvalReport` — ghi
  `reports/eval-*.json|csv` + rows `experiments`/`experiment_results`.

## ReportService

- `read_report_json(analysis_id) -> dict | None` — export đầy đủ ở P08.
