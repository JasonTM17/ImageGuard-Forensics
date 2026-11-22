# Testing Strategy

## Kim tự tháp

- **Unit** (đa số): models/limits/config, metrics R1–R5, matching stages,
  fusion rules, artifact-store guards — nhanh, deterministic, không I/O nặng.
- **Integration**: safe_open pipeline, AnalysisService round-trip
  (analyze→persist→reopen→delete), eval runner end-to-end trên synth set
  nhỏ, UI worker contract.
- **UI smoke** (pytest-qt, offscreen QPA): shell, open-preview, error path,
  disclaimer presence — không pixel-diff.

## Nguyên tắc

1. **Determinism**: mọi fixture sinh bằng seed cố định (`np.random.Generator(PCG64)`);
   synthforge deterministic — cùng seed cùng bytes.
2. **Fixture thật > mock**: test detector trên ảnh forge thật (synthforge),
   không mock AlgorithmResult.
3. **Contract tests**: mọi `AlgorithmResult` có score → `score_definition`;
   banned-wording test cho disclaimer/UI strings.
4. **Windows quirks regression**: unicode path (imencode+write_bytes),
   tên dir dành riêng (aux→auxiliary), console cp1252.
5. **Empty-mask honesty**: R1–R5 có test riêng — không bao giờ để empty
   prediction âm thầm thành F1=1.

## Chạy

PowerShell (Windows, khuyến nghị — không cần activate venv):

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = '<repo>\src'
& '<python>' -m pytest -x -q -p no:cacheprovider
```

Ví dụ dùng venv trong repo:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -x -q -p no:cacheprovider
```

Gate mỗi phase: full suite xanh; `ruff check` sạch trên phạm vi thay đổi;
chỉ merge/push sau khi gate pass trên nhánh feature tương ứng.

## CI (GitHub Actions)

Trên `pull_request` và `push` vào `main`, workflow `ci` chạy trên
`windows-latest`: job bắt buộc tên `test` (Python 3.14) cài `ruff`, `pytest`,
`pytest-qt`, cài editable package, rồi `ruff check src tests scripts` và
`pytest -q` với `QT_QPA_PLATFORM=offscreen`. Job song song `test-py313` lặp
cùng bước trên Python 3.13 (không thay job `test` làm required check).

## Độ phủ kiểm thử (đo ngày 2026-11-22)

Chạy bằng `QT_QPA_PLATFORM=offscreen` và `pytest --cov=src/imageguard --cov-report=term-missing`:

- **Tổng hợp:** **1623 passed, 5 skipped** (`0` warnings; 4 skip do thiếu quyền symlink Windows, 1 skip do `cv2.HOGDescriptor` không có trong bản OpenCV 5.0.0.93); độ phủ mã nguồn **94%** (`9944` statements, `585` missed — tăng từ `92%` / `1598 passed` ở lần đo trước và `90%` / `1547 passed` ở baseline).
- **Nhóm trọng yếu đạt độ phủ cao:** `__main__.py`, `core/io.py`, `core/preprocess.py`, `core/config.py`, `core/models.py`, `core/paths.py`, `enhancement/nafnet.py`, `evaluation/enhance_metrics.py`, `forensics/fusion.py`, `forensics/localize/fusion_masks.py`, `forensics/keypoint/viz.py`, `forensics/auxiliary/ela.py`, `ingest/camera.py`, `ingest/commands.py`, `ingest/custody.py`, `ipkit/_util.py`, `services/analysis_service.py`, `services/evaluation_service.py`, `services/history_service.py`, `storage/db.py`, `storage/models.py`, `storage/repository.py`, `ui/app.py`, `ui/screens/stats.py`, `ui/widgets/image_canvas.py`, `ui/widgets/verdict_panel.py` (100%); `ipkit/edges.py` (99%); `cli.py`, `evaluation/enhance_ops.py`, `evaluation/metrics.py`, `forensics/auxiliary/splicing.py`, `forensics/block/pipeline.py`, `ui/widgets/camera_dialog.py` (98%); `evaluation/nafnet_arm.py`, `evaluation/robustness.py`, `forensics/keypoint/pipeline.py`, `services/report_service.py`, `ui/widgets/sample_dialog.py` (97%); `evaluation/blind_kernel.py` (95%); `core/metadata.py` & `ipkit/features_basic.py` (94%); `forensics/localize/heatmap.py` & `ui/screens/analyze.py` (92%); `enhancement/derive.py` & `ui/screens/history.py` (91%); `evaluation/enhance_study.py`, `evaluation/modelbench.py`, `evaluation/synthforge.py`, `storage/artifacts.py`, `ui/screens/enhance.py` (90%).
