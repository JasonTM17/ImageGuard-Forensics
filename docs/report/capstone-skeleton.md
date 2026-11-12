# Capstone Report Skeleton — ImageGuard Forensics

Khung báo cáo đồ án; mỗi mục map với code/docs thật. **Chỉ điền số đo từ
artifact bạn đã chạy** (`reports/*.json`, CSV, hoặc summary đã lưu) — không
chép số từ README/tài liệu khác nếu chưa tự verify file.

**Ran giới wording:** dùng *dấu hiệu nghi vấn* / *suspicious indication*;
tránh “ảnh giả”, “xác thực ảnh thật”, hoặc phần trăm “độ chính xác” không
có định nghĩa metric.

---

## Mục lục (điền số trang khi hoàn thiện)

1. [Tóm tắt (Abstract)](#tóm-tắt-abstract)
2. [Giới thiệu & câu hỏi nghiên cứu](#giới-thiệu--câu-hỏi-nghiên-cứu)
3. [Cơ sở lý thuyết & công trình liên quan](#cơ-sở-lý-thuyết--công-trình-liên-quan)
4. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
5. [Triển khai & quyết định kỹ thuật](#triển-khai--quyết-định-kỹ-thuật)
6. [Đạo đức, phạm vi & giới hạn tuyên bố](#đạo-đức-phạm-vi--giới-hạn-tuyên-bố)
7. [Thực nghiệm & kết quả](#thực-nghiệm--kết-quả)
8. [Phân tích lỗi & INCONCLUSIVE](#phân-tích-lỗi--inconclusive)
9. [Hạn chế & hướng mở rộng](#hạn-chế--hướng-mở-rộng)
10. [Kết luận & demo](#kết-luận--demo)
11. [Phụ lục — tái lập & artifact](#phụ-lục--tái-lập--artifact)

---

## Tóm tắt (Abstract)

*Điền ~150–250 từ sau khi có kết quả cuối.*

- Bài toán: phát hiện **dấu hiệu** copy-move / bất thường JPEG-metadata trên ảnh số.
- Phương pháp: pipeline đa-method (keypoint, block DCT, ELA + tín hiệu phụ),
  fusion verdict 5 trạng thái, desktop PySide6.
- Dữ liệu: nêu rõ split, seed, nguồn — xem [bảng dataset](#bảng-dataset-điền-từ-thực-tế).
- Kết quả chính: **trích một số từ bảng artifact**, không quảng bá “độ chính xác tổng”.
- Kết luận: tool hỗ trợ điều tra có giải thích, không thay thế pháp y kỹ thuật số.

---

## Giới thiệu & câu hỏi nghiên cứu

### Bối cảnh

- Copy-move/splicing trong ảnh số; nhu cầu công cụ **offline**, có mask giải thích.
- Vì sao verdict thận trọng quan trọng trong môn xử lý ảnh / ứng dụng thực tế.

### Phát biểu vấn đề

*Điền 1 đoạn ngắn: ảnh đầu vào → cần đầu ra gì (báo cáo, mask, lịch sử) mà không overclaim.*

### Câu hỏi nghiên cứu (gợi ý — chỉnh cho đúng scope đồ án)

1. Keypoint và block DCT cho **recall / FPR / INCONCLUSIVE** thế nào trên tập
   benchmark đã chọn (oracle mask rõ ràng)?
2. Fusion và UI trình bày verdict có giúp người dùng **đọc dấu hiệu** mà không
   hiểu nhầm là “ảnh giả”?
3. Pipeline có **tái lập** được không (seed, config hash, cùng artifact)?

### Đóng góp (map code)

- Pipeline đa-method + fusion minh bạch — `src/imageguard/forensics/fusion.py`.
- Eval deterministic — `evaluation/synthforge.py`, `evaluation/runner.py`.
- Desktop app + export Markdown — `services/report_service.py`, `ui/screens/analyze.py`.

---

## Cơ sở lý thuyết & công trình liên quan

### Thuật toán (nội bộ)

- SIFT / matching — [`docs/algorithms/02-keypoint.md`](../algorithms/02-keypoint.md).
- Block DCT — [`docs/algorithms/03-block.md`](../algorithms/03-block.md).
- ELA, noise, JPEG cues — [`docs/algorithms.md`](../algorithms.md).

### Tài liệu ngoài

- Điền BibTeX/word từ [`docs/reading-list.md`](../reading-list.md).
- CoMoFoD: Tralic et al., ELMAR-2013 (dataset copy-move có GT).
- *Không* trích CASIA nếu bạn không nêu rõ phiên bản/oracle và licence.

### So sánh ngắn

Bảng: phương pháp cổ điển vs pipeline này (giải thích được, không end-to-end DL).

---

## Kiến trúc hệ thống

- Layer + data flow: [`docs/architecture.md`](../architecture.md).
- ERD SQLite: [`docs/database.md`](../database.md).
- UI 5 tab: [`docs/ui-design.md`](../ui-design.md).

**Hình bắt buộc (tự vẽ, chưa có trong repo — liệt kê để sản xuất):**

| ID | Nội dung hình | Nguồn / ghi chú |
| --- | --- | --- |
| H1 | Sơ đồ layer (UI → Services → Forensics → Storage) | Dựa diagram trong `architecture.md` |
| H2 | Luồng một lần phân tích (file → report artifact) | Sequence hoặc flowchart |
| H3 | Screenshot tab Phân tích (banner + overlay mask) | Chụp từ app thật, ảnh demo không nhạy cảm |
| H4 | Ví dụ mask keypoint vs block trên cùng ảnh synth | Từ `data/synth_*` hoặc artifact lưu |
| H5 | (Tùy chọn) Kiến trúc ingest / custody | Chỉ nếu báo cáo có mục ingest |

Đánh số hình: **Hình 1**, **Hình 2**, … — tham chiếu trong văn bản (“xem Hình 2”).

---

## Triển khai & quyết định kỹ thuật

| Quyết định | Lựa chọn | Lý do | Trade-off |
| --- | --- | --- | --- |
| Threading | `QThreadPool` max 1 | Giới hạn RAM | Chậm hơn parallel |
| Qt bridge | `.copy()` mọi QImage | Tránh dangling | Thêm bộ nhớ |
| Ngưỡng | `configs/thresholds.yaml` | Tái lập, audit | Cần eval khi đổi |
| Aux methods | Chỉ tăng INCONCLUSIVE | Tránh overclaim | Ít “chắc chắn” hơn |

Tham khảo thêm: [`docs/service-interface.md`](../service-interface.md),
[`docs/security-error-handling.md`](../security-error-handling.md).

---

## Đạo đức, phạm vi & giới hạn tuyên bố

- **Phạm vi:** ảnh tĩnh, định dạng trong `safe_open`; không camera stream;
  không khẳng định nguồn gốc pháp lý.
- **Dataset:** tôn trọng licence (CoMoFoD tải tay, không redistribute); CASIA
  chỉ mô tả nếu dùng cohort đã duyệt nội bộ — nêu oracle, không leak đường dẫn
  máy cá nhân.
- **Ứng dụng:** kết quả là **chỉ dẫn kỹ thuật** cho học thuật; không dùng làm
  bằng chứng tội phạm mà không kiểm chứng độc lập.
- **AI enhancement (NAFNet):** chỉ xem, không phân tích; không train trên user data.

---

## Thực nghiệm & kết quả

### Bảng dataset (điền từ thực tế)

| Dataset | Nguồn | Kích thước (n) | Split | Seed | Licence / ghi chú |
| --- | --- | --- | --- | --- | --- |
| synthforge | In-repo generator | *điền* | dev/test theo `dataset-strategy.md` | PCG64 / *ghi seed run* | Nội bộ, không benchmark công bố |
| CoMoFoD | Tải tay | *điền sau verify* | dev/test | `sha256 % 2` | ELMAR-2013, không commit data |
| CASIA (nếu dùng) | Native cohort đã khóa | *103 cặp nếu cùng sự kiện formal* | hold-aside *điền* | *ví dụ 42 nếu cùng receipt* | Chỉ mô tả đã được duyệt scope |

Hướng dẫn split: [`docs/dataset-strategy.md`](../dataset-strategy.md).

### Giao thức đo

- Metric R1–R5, empty-mask: [`docs/evaluation-methodology.md`](../evaluation-methodology.md).
- Lệnh eval: [`docs/user-guide.md`](../user-guide.md) mục Evaluation CLI.

### Chương kết quả — chỉ số từ evidence

*Không điền số ví dụ ở đây trước khi mở JSON thật.*

1. **Dev/regression (synth / CoMoFoD):** bảng F1/IoU macro per method — copy
   từ `reports/<experiment_id>.json`.
2. **Formal / native (nếu có):** bảng confusion suspicious / INCONCLUSIVE /
   recall / FPR — copy từ báo cáo sự kiện đã lưu; ghi rõ cohort ≠ ảnh camera.
3. **Fusion app:** nếu **chưa đo**, ghi `NOT_EVALUATED` — không suy từ method đơn lẻ.
4. **Robustness (tùy chọn):** JPEG q / rotate — `evaluation/robustness.py`.

**Bảng mẫu (điền sau khi có file):**

| Bảng | Nội dung | Artifact |
| --- | --- | --- |
| Bảng 1 | Macro F1/IoU từng method, synth dev | `reports/...json` |
| Bảng 2 | Confusion CASIA (nếu áp dụng) | JSON báo cáo sự kiện |
| Bảng 3 | Thời gian chạy / RAM (nếu đo) | Log hoặc bench script |

Đánh số: **Bảng 1**, **Bảng 2**, …

### README limitations (tham chiếu ngôn ngữ, không thay artifact)

Khi tóm tắt CASIA formal đã công bố nội bộ, có thể nêu định tính (ví dụ keypoint
22 FP trên 103 originals, block 37 TP trên 103 edited) **chỉ sau** đối chiếu file
JSON — xem tóm tắt định tính tại [`README.md`](../../README.md) mục giới hạn.

---

## Phân tích lỗi & INCONCLUSIVE

**Yêu cầu tối thiểu:** ≥ **3** mẫu có verdict hoặc method-level **INCONCLUSIVE**,
mỗi mẫu gồm:

1. Thumbnail hoặc ID ảnh (không dữ liệu nhạy cảm).
2. Method nào INCONCLUSIVE và **vì sao** (ít texture, cancel, aux mâu thuẫn, mask rỗng…).
3. Mask/overlay (nếu có) — giải thích vì sao không đủ để gọi suspicious.

Gợi ý nguồn: [`docs/failure-cases.md`](../failure-cases.md), artifact
`stage_evidence` trong `report.json`, hoặc nhật ký eval.

| Mẫu | Ảnh / ID | Nguyên nhân INCONCLUSIVE (điền) | Hình minh họa |
| --- | --- | --- | --- |
| L1 | | | H? |
| L2 | | | H? |
| L3 | | | H? |

Phân biệt: **FN** (edited nhưng không suspicious) khác **INCONCLUSIVE** (không
đủ bằng chứng để phân loại).

---

## Hạn chế & hướng mở rộng

- Splicing benchmark hạn chế — [`docs/splicing-study.md`](../splicing-study.md).
- ML extension design-only — [`docs/ml-extension.md`](../ml-extension.md).
- Đóng gói `.exe` — [`docs/packaging.md`](../packaging.md) (build vs runtime verify).

---

## Kết luận & demo

- Checklist MVP (GUI analyze, history, export md, eval CLI).
- Demo flow 3–5 phút: mở ảnh → chạy method → đọc banner → overlay → xuất Markdown.
- Lời kết: nhấn lại ranh giới khoa học.

---

## Phụ lục — tái lập & artifact

### Lệnh tái lập (Windows PowerShell — chỉnh `<repo>`)

```powershell
cd <repo>
.\.venv\Scripts\pip install -e ".[dev]"

# Test suite (gate)
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider

# Eval ví dụ
.\.venv\Scripts\python.exe -m imageguard eval --dataset synth `
  --root data/synth_eval --split dev --methods keypoint,block --seed 42

# Một ảnh CLI
.\.venv\Scripts\python.exe -m imageguard analyze path\to\image.jpg --no-persist
```

Chi tiết môi trường: [`docs/environment.md`](../environment.md),
[`docs/testing-strategy.md`](../testing-strategy.md).

### Artifact nộp kèm (gợi ý)

- Snapshot `thresholds_hash` + commit hash git.
- Một `report.json` + `report.md` xuất từ UI.
- JSON eval run đã trích bảng trong chương 7.
- Log pytest (PASS/FAIL, số test) — không claim nếu không chạy.

### Figures checklist (không vẽ sẵn trong repo)

- [ ] H1 Layer diagram
- [ ] H2 Data-flow
- [ ] H3 UI screenshot
- [ ] H4 Method overlay comparison
- [ ] H5 (optional) Ingest

*Trạng thái điền khi nộp: PASS = đã chèn vào báo cáo; NOT_RUN = chưa làm.*
