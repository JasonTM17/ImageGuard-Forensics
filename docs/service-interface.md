# Service Interface (contract cho UI P07)

## AnalysisService

```python
svc.analyze(AnalysisRequest(
    path=Path, methods=["keypoint","block","ela"],
    cancel=threading.Event | None,
    progress_cb=lambda frac, stage: ...,
    include_exif=True,
    reference_path=Path | None,
    deepfake_model_path=Path | None,
    deepfake_model_sha256=str | None,
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
- `deepfake_model_path` chỉ áp dụng cho spec `deepfake`: nạp JSON đã học,
  kiểm tra feature schema/config/checksum rồi giữ quality gate của DIP.
  `deepfake_model_sha256` tùy chọn ghim file được UI/CLI chọn; mismatch trả
  method `ERROR`, không âm thầm quay về heuristic. Không truyền model thì
  giữ nhánh mặc định. Hash model và heuristic baseline nằm trong evidence.
- Lỗi persistence (DB/artifact) không huỷ kết quả: service thêm cảnh báo vào
  `report.limitations` và vẫn trả `ForensicReport`.
  Mỗi lượt lưu giành quyền sở hữu UUID/thư mục artifact mới trước khi ghi.
  Khi ghi thất bại, service thử dọn cả rows của UUID và thư mục artifact,
  giữ `ImageAsset`, ảnh nguồn và các báo cáo trước. Nếu cleanup không hoàn
  tất, limitation nêu UUID, stage lỗi và loại exception để phục hồi thủ công.
  Đây là compensation cho lỗi thông thường, không phải giao dịch nguyên tử
  xuyên SQLite/filesystem hay bảo đảm khi tiến trình bị dừng. Khi dọn file
  thất bại, `report.json` còn sót có thể vẫn được ReportService đọc; không
  coi đó là bằng chứng lượt lưu đã hoàn tất.

## HistoryService

- `list(limit=100) -> [{id, created_at, verdict, duration_ms, image_name,
  n_methods}]`
- `get(id) -> {id, created_at, verdict, duration_ms, image_name, config_hash,
  methods[{method, verdict, score, score_definition, evidence, limitations, duration_ms}],
  artifacts[{kind, method, rel_path}]} | None`
  — `evidence` và `limitations` đọc lại từ SQLite. Trường JSON cũ rỗng,
  sai dạng hoặc không đọc được có giá trị hiển thị mặc định `{}` / `[]`.
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
  bản sao tại `out_path`. Trong artifact store, đích được phép là `report.md`
  của chính phân tích đang xuất. Các đích artifact khác bị từ chối trước khi
  ghi. Bản sao ngoài store được ghi qua file tạm cùng thư mục và thay thế
  nguyên tử để giữ file đích cũ nếu lỗi ghi hoặc publish. Hai bản dùng cùng
  dữ liệu UTF-8. Nếu ghi ra ngoài thất bại, bản `report.md` trong store có thể
  đã được cập nhật và có thể dùng để xuất lại.
