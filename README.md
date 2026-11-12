# ImageGuard Forensics

Ứng dụng **desktop** (Python + PySide6) phân tích **dấu hiệu nghi vấn** trên ảnh số — đồ án môn Xử lý ảnh. Trọng tâm: copy-move (keypoint SIFT/ORB, block DCT), ELA cho JPEG, tín hiệu phụ (noise/tần số/metadata), fusion verdict thận trọng và khung đánh giá có seed cố định.

> **Ranh giới khoa học:** Công cụ chỉ báo *dấu hiệu có thể liên quan tới chỉnh sửa copy-move hoặc bất thường nén/metadata* — **không** khẳng định ảnh giả, **không** chứng minh ảnh nguyên bản. ELA/EXIF/noise là bằng chứng khám phá; khi không đủ tín hiệu, verdict là `INCONCLUSIVE`. Báo cáo tổng hợp mang tính *gợi ý điều tra*, không thay thế phân tích chuyên gia.

Khung báo cáo đồ án: [`docs/report/capstone-skeleton.md`](docs/report/capstone-skeleton.md).

## Quick start (Windows / PowerShell)

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

Yêu cầu: Python ≥ 3.13 (phát triển trên 3.14). Không cần kích hoạt venv thủ công nếu gọi trực tiếp `\.venv\Scripts\python.exe`.

### Phân tích nhanh (CLI, JSON)

```powershell
.\.venv\Scripts\python.exe -m imageguard analyze <đường-dẫn-ảnh> `
  [--methods keypoint,block,ela] [--no-persist]
```

Stdout là `ForensicReport` JSON (cùng cấu trúc artifact khi bật persist). Exit khác 0 = lỗi mở/đọc file (`ErrorResult`), không phải verdict forensics.

Chi tiết GUI, lịch sử, xuất Markdown: [`docs/user-guide.md`](docs/user-guide.md).

## Chỉ mục tài liệu (`docs/`)

| Tài liệu | Nội dung |
| --- | --- |
| [`docs/user-guide.md`](docs/user-guide.md) | Vận hành GUI/CLI, verdict, Enhance, ingest |
| [`docs/ui-design.md`](docs/ui-design.md) | Tab, worker, overlay, wording |
| [`docs/architecture.md`](docs/architecture.md) | Layer, data flow, cây thư mục |
| [`docs/testing-strategy.md`](docs/testing-strategy.md) | Kim tự tháp test, determinism, lệnh pytest |
| [`docs/environment.md`](docs/environment.md) | Phiên bản runtime đã xác minh |
| [`docs/packaging.md`](docs/packaging.md) | PyInstaller; giới hạn build `.exe` |
| [`docs/algorithms.md`](docs/algorithms.md) | Tổng quan thuật toán |
| [`docs/algorithms/02-keypoint.md`](docs/algorithms/02-keypoint.md) | Pipeline keypoint |
| [`docs/algorithms/03-block.md`](docs/algorithms/03-block.md) | Pipeline block DCT |
| [`docs/algorithms/comparison.md`](docs/algorithms/comparison.md) | So sánh method |
| [`docs/database.md`](docs/database.md) | SQLite, artifact store |
| [`docs/service-interface.md`](docs/service-interface.md) | Analysis / History / Eval services |
| [`docs/security-error-handling.md`](docs/security-error-handling.md) | safe_open, giới hạn tài nguyên |
| [`docs/dataset-strategy.md`](docs/dataset-strategy.md) | synthforge, CoMoFoD, split |
| [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) | Metric R1–R5, runner |
| [`docs/experiments.md`](docs/experiments.md) | Ghi chép thử nghiệm |
| [`docs/failure-cases.md`](docs/failure-cases.md) | Mẫu lỗi đã quan sát |
| [`docs/ingest.md`](docs/ingest.md) | Quét thư mục, custody |
| [`docs/enhancement-study.md`](docs/enhancement-study.md) | Nghiên cứu tăng cường xem |
| [`docs/modelforge.md`](docs/modelforge.md) | Model-forge / ONNX (design) |
| [`docs/ml-extension.md`](docs/ml-extension.md) | Hướng ML (design-only) |
| [`docs/splicing-study.md`](docs/splicing-study.md) | Splicing cues (exploratory) |
| [`docs/reading-list.md`](docs/reading-list.md) | Tài liệu tham khảo |
| [`docs/report/capstone-skeleton.md`](docs/report/capstone-skeleton.md) | Khung báo cáo đồ án |

## Cấu trúc repo (rút gọn)

- `src/imageguard/` — engine forensics (Qt-free), services, UI PySide6, evaluation, ingest, enhancement
- `configs/thresholds.yaml` — ngưỡng thuật toán tập trung (không chỉnh trong scope tài liệu)
- `tests/` — pytest (+ pytest-qt offscreen)
- `data/` — runtime (artifacts, DB) — **gitignored**

## Giới hạn đã đo (không overclaim)

Các số dưới đây đến từ **một sự kiện đánh giá chính thức** trên cohort **103 cặp** ảnh CASIA đã sàng lọc (oracle mask tác giả, **không** phải ảnh chụp camera tùy ý). Đó là đo *dấu hiệu suspicious* theo oracle dataset, không chứng minh authentic/giả cho ảnh bất kỳ.

| Phương pháp | Trên ảnh đã chỉnh (103) | Trên ảnh gốc (103) |
| --- | --- | --- |
| **Keypoint** | 94 coi suspicious (9 FN); 9 INCONCLUSIVE trên ảnh chỉnh | 22 FP suspicious; **59 INCONCLUSIVE** (trong 81 hàng “nonsuspicious” theo confusion) |
| **Block** | 37 coi suspicious (66 không suspicious); 2–3 INCONCLUSIVE | 0 FP suspicious; 0 INCONCLUSIVE |

- Block: original FPR **0/103** trên mẫu này — *không* khẳng định FPR luôn bằng 0 mọi nguồn ảnh.
- Keypoint: edited recall ~**91,26%** trên cohort này; original FPR ~**21,36%** (22/103).
- Macro IoU (mask cue, 103 ảnh chỉnh): keypoint ~**0,461**; block ~**0,289** (baseline `.01`).
- **Verdict fusion trong app/UI** chưa được đánh giá trong sự kiện hai-method trên; người dùng GUI thấy kết quả tổng hợp khác pipeline đo từng method đơn lẻ.
- Qualification trên CoMoFoD dev **không** được coi là bằng chứng chuyển giao sang cohort CASIA trên.

Điều tra chất lượng detector và artifact JSON/CSV nằm ngoài repo công khai; khi viết báo cáo, trích số từ artifact đã lưu, không suy diễn thêm.
