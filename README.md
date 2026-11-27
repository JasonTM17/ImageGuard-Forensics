# ImageGuard Forensics

Ứng dụng **desktop** (Python + PySide6) phân tích **dấu hiệu nghi vấn** trên ảnh số và luồng thu nhận đa camera. Trọng tâm: phát hiện copy-move (keypoint SIFT/ORB, block DCT), ELA cho JPEG, các tín hiệu phụ (noise/local-MAD, tần số, splicing, và phát hiện dấu hiệu Deepfake/AI 6 trụ cột hỗ trợ Dual-Camera Cross-Sensor), thu nhận trực tiếp từ Webcam/USB/IoT Camera kèm chẩn đoán quang học, module Khôi phục & Hiệu chuẩn Camera (Super-Resolution, Match Histogram, lọc tần số FFT), fusion verdict 5 trạng thái thận trọng và khung đánh giá có seed cố định.

> **Ranh giới khoa học:** Công cụ chỉ báo *dấu hiệu có thể liên quan tới chỉnh sửa copy-move hoặc bất thường nén/metadata* — **không** khẳng định ảnh giả, **không** chứng minh ảnh nguyên bản. ELA/EXIF/noise là bằng chứng khám phá; khi không đủ tín hiệu, verdict là `INCONCLUSIVE`. Báo cáo tổng hợp mang tính *gợi ý điều tra*, không thay thế phân tích chuyên gia.

Khung báo cáo đồ án: [`docs/report/capstone-skeleton.md`](docs/report/capstone-skeleton.md).

### Phần Deepfake hiện làm được gì?

Đây là nhánh **sàng lọc dấu hiệu trên vùng mặt bằng xử lí ảnh cổ điển**;
ngoài pipeline cổ điển đã có các model học sâu tùy chọn GenConViT và CoC
Xception. Điểm 0–1 vẫn là chỉ số chưa hiệu chuẩn, không phải xác suất ảnh giả. Không tìm thấy mặt, ROI quá
nhỏ/thiếu tương phản hoặc không có nền đối chiếu thì không cấp điểm.

Bản sửa 2026-11-27 chuẩn hóa phép đo phổ theo kích thước ROI, yêu cầu nhiều
nhóm dấu hiệu cùng hỗ trợ, phân biệt rõ cờ bất thường với thiếu dữ liệu,
và bỏ việc cộng điểm do mặt đứng yên/ảnh tham chiếu. Chế độ hai camera dùng
khai báo bố cục và ranh giới rõ ràng; ảnh ngang rộng không tự bị chia đôi.

Lượt cải thiện tiếp theo bổ sung kiểm chứng hai mắt cho fallback màu da,
ưu tiên ROI đủ chất lượng và kiểm tra phơi sáng/căn dịch nhỏ khi so mốc.
Dependency được giới hạn `opencv-python>=4.10,<5` để giữ API Haar Cascade;
trên build thiếu cascade, báo cáo hiện rõ khả năng detector bị giảm.

Có thể **học trọng số từ dữ liệu có nhãn của nhóm** bằng logistic regression
trên 13 đặc trưng DIP, không cần GPU hoặc thư viện học sâu. Dùng
`scripts/train_deepfake_model.py`, đánh giá mô hình cố định trên test bằng
`scripts/evaluate_deepfake_cues.py --model ...`, rồi chọn **Nạp mô hình DIP…**
trong giao diện. Mặc định vẫn dùng heuristic; repo không kèm trọng số Deepfake
đã huấn luyện. Quy trình học giữ các frame cùng nguồn trong cùng fold,
chuẩn hóa riêng trên fold train và kiểm tra trùng nguồn/ảnh khi chạy test.

**Chưa có số đo độ chính xác Deepfake trên bộ ảnh thật/giả có nhãn.**
Để thực nghiệm cho đồ án, dùng [`docs/deepfake-evaluation.md`](docs/deepfake-evaluation.md)
và `scripts/evaluate_deepfake_cues.py`: báo coverage, tỷ lệ bật cờ riêng trên
ảnh thật/giả, AUC điểm và toàn bộ lý do mẫu không đủ điều kiện.

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
| [`docs/camera-dialog.md`](docs/camera-dialog.md) | Luồng hộp thoại camera, worker và hủy thao tác |
| [`docs/ui-design.md`](docs/ui-design.md) | Tab, worker, overlay, wording |
| [`docs/architecture.md`](docs/architecture.md) | Layer, data flow, cây thư mục |
| [`docs/testing-strategy.md`](docs/testing-strategy.md) | Kim tự tháp test, determinism, lệnh pytest |
| [`docs/environment.md`](docs/environment.md) | Phiên bản runtime đã xác minh |
| [`docs/performance.md`](docs/performance.md) | Thời gian wall-clock đo trên một máy |
| [`docs/packaging.md`](docs/packaging.md) | PyInstaller; giới hạn build `.exe` |
| [`docs/algorithms.md`](docs/algorithms.md) | Tổng quan thuật toán |
| [`docs/algorithms/02-keypoint.md`](docs/algorithms/02-keypoint.md) | Pipeline keypoint |
| [`docs/algorithms/03-block.md`](docs/algorithms/03-block.md) | Pipeline block DCT |
| [`docs/algorithms/comparison.md`](docs/algorithms/comparison.md) | So sánh method |
| [`docs/database.md`](docs/database.md) | SQLite, artifact store |
| [`docs/service-interface.md`](docs/service-interface.md) | Analysis / History / Eval services |
| [`docs/forensic-report-schema.md`](docs/forensic-report-schema.md) | Hợp đồng JSON stdout của lệnh `analyze` |
| [`docs/security-error-handling.md`](docs/security-error-handling.md) | safe_open, giới hạn tài nguyên |
| [`docs/dataset-strategy.md`](docs/dataset-strategy.md) | synthforge, CoMoFoD, split |
| [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) | Metric R1–R5, runner |
| [`docs/deepfake-evaluation.md`](docs/deepfake-evaluation.md) | Manifest, huấn luyện DIP, grouped CV, test, coverage và chống rò rỉ dữ liệu |
| [`docs/experiments.md`](docs/experiments.md) | Ghi chép thử nghiệm |
| [`docs/failure-cases.md`](docs/failure-cases.md) | Mẫu lỗi đã quan sát |
| [`docs/limitations.md`](docs/limitations.md) | Giới hạn đã biết, không thêm số mới |
| [`docs/ingest.md`](docs/ingest.md) | Quét thư mục, custody |
| [`docs/enhancement-study.md`](docs/enhancement-study.md) | Nghiên cứu tăng cường xem |
| [`docs/modelforge.md`](docs/modelforge.md) | Model-forge / ONNX (design) |
| [`docs/ml-extension.md`](docs/ml-extension.md) | Logistic DIP hiện có và hướng mạng sâu (design-only) |
| [`docs/splicing-study.md`](docs/splicing-study.md) | Splicing cues (exploratory) |
| [`docs/image-processing-toolkit.md`](docs/image-processing-toolkit.md) | Thư viện toán tử Xử lý ảnh số thuần NumPy (`ipkit`) phục vụ hiệu chuẩn & phân tích phụ trợ |
| [`docs/keypoint-calibration-study.md`](docs/keypoint-calibration-study.md) | Nghiên cứu offline false-cue keypoint (không đổi ngưỡng shipped) |
| [`docs/reading-list.md`](docs/reading-list.md) | Tài liệu tham khảo |
| [`docs/course-coverage.md`](docs/course-coverage.md) | Bản đồ đối chiếu thuật toán → mã nguồn, kiểm thử |
| [`docs/report/capstone-skeleton.md`](docs/report/capstone-skeleton.md) | Khung báo cáo kỹ thuật |
| [`docs/report/capstone-draft.md`](docs/report/capstone-draft.md) | Bản nháp báo cáo đầy đủ (Chương 1–7) |
| [`docs/report/figures.md`](docs/report/figures.md) | Hình minh họa synthetic cho báo cáo |
| [`docs/report/slides-outline.md`](docs/report/slides-outline.md) | Dàn ý slide thuyết trình |
| [`docs/demo-script.md`](docs/demo-script.md) | Kịch bản demo trực tiếp và Q&A hội đồng |
| [`docs/glossary.md`](docs/glossary.md) | Thuật ngữ và vị trí trong mã nguồn |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Khắc phục sự cố trên Windows |
| [`docs/fusion-offline-analysis.md`](docs/fusion-offline-analysis.md) | Đếm verdict gộp trên 103 cặp đã lưu (khám phá) |
| [`docs/casia-formal-evidence-summary.md`](docs/casia-formal-evidence-summary.md) | Tóm tắt redact sự kiện CASIA formal + SHA-256 artifact (ngoài repo) |
| [`docs/README.en.md`](docs/README.en.md) | English summary |

## Cấu trúc repo (rút gọn)

- `src/imageguard/` — engine forensics (Qt-free), services, UI PySide6, evaluation, ingest, enhancement
- `configs/thresholds.yaml` — ngưỡng thuật toán tập trung (không chỉnh trong scope tài liệu)
- `tests/` — pytest (+ pytest-qt offscreen)
- `data/` — runtime (artifacts, DB) — **gitignored**

## Giới hạn đã đo (không overclaim)

Các số dưới đây đến từ **một sự kiện đánh giá chính thức** trên cohort **103 cặp** ảnh CASIA đã sàng lọc (oracle mask tác giả, **không** phải ảnh chụp camera tùy ý). Đó là đo *dấu hiệu suspicious* theo oracle dataset, không chứng minh authentic/giả cho ảnh bất kỳ. Receipt aggregate redact và SHA-256 artifact: [`docs/casia-formal-evidence-summary.md`](docs/casia-formal-evidence-summary.md) (artifact đầy đủ giữ **ngoài** repo git).

| Phương pháp | Trên ảnh đã chỉnh (103) | Trên ảnh gốc (103) |
| --- | --- | --- |
| **Keypoint** | 94 coi suspicious (9 FN); 9 INCONCLUSIVE trên ảnh chỉnh | 22 FP suspicious; **59 INCONCLUSIVE** (trong 81 hàng “nonsuspicious” theo confusion) |
| **Block** | 37 coi suspicious; 66 không suspicious *(gồm 3 `INCONCLUSIVE` + 63 `NO_INDICATION` trên ảnh chỉnh — [`docs/fusion-offline-analysis.md`](docs/fusion-offline-analysis.md))* | 0 FP suspicious; 0 INCONCLUSIVE |

- Block: original FPR **0/103** trên mẫu này — *không* khẳng định FPR luôn bằng 0 mọi nguồn ảnh.
- Keypoint: edited recall ~**91,26%** trên cohort này; original FPR ~**21,36%** (22/103).
- Macro IoU (mask cue, 103 ảnh chỉnh): keypoint ~**0,461**; block ~**0,289** (baseline `.01`).
- **Verdict fusion trong app/UI** chưa được đánh giá trong sự kiện hai-method trên; người dùng GUI thấy kết quả tổng hợp khác pipeline đo từng method đơn lẻ.
- Qualification trên CoMoFoD dev **không** được coi là bằng chứng chuyển giao sang cohort CASIA trên.

Khi viết báo cáo, trích số từ [`docs/casia-formal-evidence-summary.md`](docs/casia-formal-evidence-summary.md) hoặc artifact đã lưu cục bộ, không suy diễn thêm.

## Giấy phép

MIT — xem [`LICENSE`](LICENSE).
