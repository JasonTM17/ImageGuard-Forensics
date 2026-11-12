# ImageGuard Forensics — English summary

Desktop application (Python + PySide6) that analyses **suspicious indications** in digital
images — an Image Processing course project. Focus areas: copy-move (SIFT/ORB keypoint,
block DCT), ELA for JPEG, auxiliary signals (noise/frequency/metadata), conservative
verdict fusion, and a seeded evaluation harness.

This page summarises the Vietnamese [`README.md`](../README.md); the full documentation
set is in Vietnamese. No new numbers or claims are introduced here — all values are
copied from the README and [`limitations.md`](limitations.md).

## Integrity stance

> **Scientific boundary:** the tool only reports *indications that may be related to
> copy-move editing or compression/metadata anomalies*. It does **not** declare an image
> fake and does **not** prove an image authentic. ELA/EXIF/noise are exploratory evidence;
> when there is not enough signal the verdict is `INCONCLUSIVE`. The report is an
> *investigation lead*, not a substitute for expert analysis.

The `Verdict` enum is closed and deliberately excludes `FORGED`/`FAKE`; its only members
are `SUSPICIOUS_INDICATION`, `NO_INDICATION`, `INCONCLUSIVE`, `NOT_APPLICABLE`, `ERROR`.
Auxiliary signals (ELA/EXIF/noise/noise_local/frequency/splicing) never return
`SUSPICIOUS_INDICATION` by contract, so they cannot convict an image on their own.
See [`forensic-report-schema.md`](forensic-report-schema.md) for the JSON output contract.

## Quick start (Windows / PowerShell)

Commands copied verbatim from [`README.md`](../README.md) (Quick start section):

```powershell
cd <thư-mục-repo>
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"

# Chạy GUI (khuyến nghị)
.\.venv\Scripts\python.exe -m imageguard
# hoặc
.\run_app.bat

# Kiểm tra môi trường
.\.venv\Scripts\python.exe scripts\verify_env.py
```

Requirements: Python ≥ 3.13 (developed on 3.14). Activating the venv manually is not
needed if you call `\.venv\Scripts\python.exe` directly.

### Quick analysis (CLI, JSON)

```powershell
.\.venv\Scripts\python.exe -m imageguard analyze <đường-dẫn-ảnh> `
  [--methods keypoint,block,ela] [--no-persist]
```

Stdout is a `ForensicReport` JSON (same structure as the persisted artifact). A non-zero
exit code means a file open/read error (`ErrorResult`), not a forensic verdict.

## Measured limits

The numbers below come from **one official evaluation event** on a filtered cohort of
**103 CASIA image pairs** (author oracle masks, **not** arbitrary camera photos). They
measure *suspicious indications* against the dataset oracle; they do not prove that any
given image is authentic or fake.

| Method | On edited images (103) | On original images (103) |
| --- | --- | --- |
| **Keypoint** | 94 treated as suspicious (9 FN); 9 INCONCLUSIVE on edited images | 22 FP suspicious; **59 INCONCLUSIVE** (within the 81 "nonsuspicious" confusion rows) |
| **Block** | 37 treated as suspicious (66 not suspicious); 2–3 INCONCLUSIVE | 0 FP suspicious; 0 INCONCLUSIVE |

- Block: original FPR **0/103** on this sample — this does *not* claim FPR is always zero
  for every image source.
- Keypoint: edited recall ~**91.26%** on this cohort; original FPR ~**21.36%** (22/103).
- Macro IoU (cue masks, 103 edited images): keypoint ~**0.461**; block ~**0.289**
  (baseline `.01`).
- **Verdict fusion in the app/UI** was not evaluated in that two-method event; GUI users
  see an aggregate result that can differ from the per-method pipeline used for
  measurement.
- Qualification on CoMoFoD dev is **not** treated as evidence of transfer to the CASIA
  cohort above.

Detailed detector-quality investigation and the JSON/CSV artifacts live outside the
public repository; when writing a report, quote numbers from the stored artifacts rather
than extrapolating.

Known limitations beyond these measurements (ELA/JPEG-only scope, geometric
transformations vs block DCT, keypoint degradation under heavy compression, exploratory
splicing cues, resource limits) are listed in [`limitations.md`](limitations.md).

## Documentation

| Document | Content |
| --- | --- |
| [`../README.md`](../README.md) | Original project README (Vietnamese) |
| [`user-guide.md`](user-guide.md) | GUI/CLI operation, verdicts, Enhance, ingest |
| [`ui-design.md`](ui-design.md) | Tabs, workers, overlays, wording |
| [`architecture.md`](architecture.md) | Layers, data flow, directory tree |
| [`algorithms.md`](algorithms.md) | Algorithm overview |
| [`algorithms/02-keypoint.md`](algorithms/02-keypoint.md) | Keypoint pipeline |
| [`algorithms/03-block.md`](algorithms/03-block.md) | Block DCT pipeline |
| [`algorithms/comparison.md`](algorithms/comparison.md) | Method comparison |
| [`testing-strategy.md`](testing-strategy.md) | Test pyramid, determinism, pytest commands |
| [`environment.md`](environment.md) | Verified runtime versions |
| [`performance.md`](performance.md) | Wall-clock timings measured on one machine |
| [`evaluation-methodology.md`](evaluation-methodology.md) | Metrics R1–R5, evaluation runner |
| [`limitations.md`](limitations.md) | Known limitations, no new numbers |
| [`security-error-handling.md`](security-error-handling.md) | `safe_open`, resource limits |
| [`database.md`](database.md) | SQLite, artifact store |
| [`service-interface.md`](service-interface.md) | Analysis / History / Eval services |
| [`forensic-report-schema.md`](forensic-report-schema.md) | `analyze` stdout JSON contract |
| [`dataset-strategy.md`](dataset-strategy.md) | synthforge, CoMoFoD, split |
| [`packaging.md`](packaging.md) | PyInstaller; `.exe` build limits |
| [`ingest.md`](ingest.md) | Directory scanning, custody |
| [`course-coverage.md`](course-coverage.md) | Course topics mapped to source and tests |
| [`report/capstone-skeleton.md`](report/capstone-skeleton.md) | Capstone report skeleton |

## Repository layout (abridged)

- `src/imageguard/` — Qt-free forensics engine, services, PySide6 UI, evaluation, ingest,
  enhancement
- `configs/thresholds.yaml` — centralised algorithm thresholds
- `tests/` — pytest (pytest-qt offscreen)
- `data/` — runtime artifacts and DB, **gitignored**
