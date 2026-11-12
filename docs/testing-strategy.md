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
