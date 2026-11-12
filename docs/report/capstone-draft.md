# Báo cáo đồ án — ImageGuard Forensics (bản nháp chương 1–4)

> **Trạng thái:** nháp cho sinh viên chỉnh sửa. Ranh giới khoa học: *dấu hiệu nghi vấn*
> / *suspicious indication* — không khẳng định ảnh giả hay xác thực nguyên bản.

Khung đầy đủ: [`capstone-skeleton.md`](capstone-skeleton.md).

---

## Chương 1 — Đặt vấn đề và phạm vi

### 1.1 Bối cảnh

Ảnh số dễ bị chỉnh sửa cục bộ (copy-move: sao một vùng và dán vào chỗ khác trong
cùng khung hình). Trong môn xử lý ảnh và bài toán điều tra ngoại đời, người dùng
cần công cụ **offline**, có mask/heatmap giải thích, nhưng không được đánh đổi
độ tin cậy khoa học để lấy “kết luận dứt khoát”.

[SINH VIÊN BỔ SUNG: ví dụ tình huống cụ thể tại lớp hoặc use-case cá nhân,
không dùng ảnh nhạy cảm.]

### 1.2 Phát biểu vấn đề

Cho ảnh tĩnh đầu vào (định dạng được `safe_open` chấp nhận), hệ thống cần:

- chạy một hoặc nhiều phương pháp forensics (keypoint, block DCT, và các cue
  bổ trợ);
- hợp nhất thành **verdict năm trạng thái** và báo cáo có `score_definition`;
- lưu lịch sử, xuất Markdown, hỗ trợ đọc overlay trên GUI.

Đầu ra là **chỉ dẫn kỹ thuật** (báo cáo, mask, metadata phương pháp), không phải
phán quyết pháp y.

### 1.3 Câu hỏi nghiên cứu

1. Keypoint (SIFT/ORB) và block DCT cho **recall ảnh mức**, **FPR**, và tỷ lệ
   **INCONCLUSIVE** thế nào trên các tập benchmark có oracle mask rõ ràng
   (synthforge, CoMoFoD dev, cohort CASIA đã sàng lọc)?
2. Fusion verdict và giao diện có giúp người dùng **đọc dấu hiệu** mà không
   hiểu nhầm là “ảnh giả” không? (Fusion app: **chưa đánh giá** trong sự kiện
   formal hai-method — xem [`README.md`](../../README.md).)
3. Pipeline có **tái lập** được không (seed, hash cấu hình, artifact eval)?

### 1.4 Đóng góp (map mã nguồn)

| Đóng góp | Vị trí mã |
| --- | --- |
| Pipeline đa-method + fusion | `src/imageguard/forensics/fusion.py` (mô tả, không chỉnh trong scope báo cáo) |
| Eval có seed | `src/imageguard/evaluation/synthforge.py`, `runner.py` |
| Desktop + export Markdown | `src/imageguard/services/report_service.py`, `ui/screens/analyze.py` |

### 1.5 Đạo đức, phạm vi và giới hạn tuyên bố

- **Phạm vi sản phẩm:** ảnh tĩnh; không camera stream; không khẳng định nguồn
  gốc pháp lý.
- **Wording:** `SUSPICIOUS` = dấu hiệu nghi vấn; `NO_INDICATION` ≠ chứng minh
  authentic; `INCONCLUSIVE` = không đủ bằng chứng — báo riêng, không gộp vào
  “đúng/sai” đơn giản ([`docs/evaluation-methodology.md`](../evaluation-methodology.md)).
- **Dataset ngoài:** CoMoFoD tải tay, không redistribute; CASIA chỉ mô tả
  cohort **103+103 cặp** đã sàng lọc nội bộ với **oracle mask tác giả** — không
  coi là ảnh camera tùy ý ([`docs/dataset-strategy.md`](../dataset-strategy.md),
  [`README.md`](../../README.md)).
- **Ứng dụng học thuật:** kết quả không thay thế kiểm chứng độc lập; tab Enhance
  / NAFNet chỉ xem, không đưa vào verdict phân tích.

**Nguồn:** [`docs/report/capstone-skeleton.md`](capstone-skeleton.md) (mục giới thiệu,
đạo đức); [`README.md`](../../README.md); [`docs/limitations.md`](../limitations.md).

---

## Chương 2 — Cơ sở lý thuyết

### 2.1 Copy-move và khảo sát phương pháp cổ điển

Copy-move tạo hai vùng **cùng nội dung pixel/texture** trong một ảnh. Hai hướng
chính trong đồ án:

- **Keypoint (sparse):** SIFT/ORB, bất biến scale/rotation cục bộ; nhạy JPEG/nhiễu
  và vùng phẳng.
- **Block DCT (dense):** khớp block đã DCT + lượng tử hóa; mạnh trên translate
  texture đều, yếu khi rotate/scale.

Christlein et al. (2012) tóm tắt trade-off benchmark — đọc kèm [`docs/reading-list.md`](../reading-list.md) mục 2.

### 2.2 Block matching và DCT (Fridrich et al.)

Ảnh xám chia block chồng lấn; mỗi block → DCT-II, lượng tử, vector hệ số zigzag.
Block giống nhau → vector gần nhau; **sắp xếp từ điển** giảm độ phức tạp so với
so khớp bậc hai. Cặp block hợp lệ gom theo **bỏ phiếu vector dịch (shift voting)**.
Vùng phẳng lọc bằng ngưỡng độ lệch chuẩn block để giảm false match.

Chi tiết triển khai: [`docs/algorithms/03-block.md`](../algorithms/03-block.md).

**Tài liệu:** Fridrich, Soukal, Lukáš (2003) — [`docs/reading-list.md`](../reading-list.md) #1.

### 2.3 Keypoint: SIFT/ORB, Lowe ratio, RANSAC

**SIFT (Lowe 2004):** scale-space, DoG keypoints, descriptor 128 chiều. **Self-match**
k-NN (k=3), bỏ cặp cùng điểm; **Lowe ratio test** `‖d₁‖/‖d₂‖ < ρ` (ρ cấu hình
trong `thresholds.yaml`) để loại texture lặp ngẫu nhiên.

**Hình học:** `findHomography` / affine với **RANSAC** — mô hình affine 2×3,
inlier theo ngưỡng pixel; thêm **displacement clustering** cho nhiều vùng translate.

Chi tiết: [`docs/algorithms/02-keypoint.md`](../algorithms/02-keypoint.md).

**Tài liệu:** Lowe (2004) #3; OpenCV RANSAC — reading-list #9.

### 2.4 ELA (Error Level Analysis)

`ELA(x,y) = |I(x,y) − R_q(I)(x,y)|` với `R_q` là JPEG re-encode ở chất lượng q.
Vùng có lịch sử nén khác → mức lỗi khác. ELA chỉ mang tính **khám phá**: re-save
xoá vệt; texture tự nhiên cũng gây false cue — trong app **không** đủ để một
mình ra `SUSPICIOUS` ([`docs/algorithms.md`](../algorithms.md)).

**Tài liệu:** Krawetz (2007) #4.

### 2.5 Noise residual và MAD

Residual `r = I − median_k(I)` giữ thành phần cao tần. Ước lượng σ theo block
bằng **MAD**: `σ̂ = 1.4826·median|r|`. Vùng lệch so với median±m·MAD toàn ảnh
→ bản đồ outlier. Denoise/sharpen hợp pháp có thể mô phỏng dấu hiệu tương tự; cue
noise tối đa **INCONCLUSIVE**.

**Tài liệu:** Lukáš, Fridrich, Goljan (2006) #6 (PRNU đầy đủ là hướng mở rộng).

### 2.6 Phân tích tần số (frequency)

FFT2 biên độ (log) phát hiện lưới JPEG 8×8, peak định kỳ khi resampling; clone
đôi khi đổi phổ cục bộ. Metric blockiness so sánh gradient biên block với nội
block. Chỉ cue khám phá — không verdict độc lập ([`docs/algorithms.md`](../algorithms.md)).

**Hình minh họa:** phổ và block-DCT synthetic — xem Hình 2.

### 2.7 So sánh ngắn với học sâu end-to-end

Đồ án chọn pipeline cổ điển có giải thích (mask, evidence JSON) thay vì mạng
ảnh-vào-nhãn-ra — phù hợp phạm vi môn và yêu cầu audit ngưỡng. Hướng ML chỉ
design: [`docs/ml-extension.md`](../ml-extension.md).

**Nguồn:** [`docs/algorithms.md`](../algorithms.md);
[`docs/algorithms/02-keypoint.md`](../algorithms/02-keypoint.md);
[`docs/algorithms/03-block.md`](../algorithms/03-block.md);
[`docs/reading-list.md`](../reading-list.md).

---

## Chương 3 — Thiết kế hệ thống

### 3.1 Kiến trúc phân lớp

```
UI (PySide6) → Services → Forensics engine (Qt-free) → Core I/O
     ↓              ↓              ↓
 Persistence (SQLite + artifacts) ; Evaluation / Ingest / Enhancement (song song)
```

- **Engine Qt-free:** `forensics/`, `core/`, `evaluation/` test headless bằng pytest.
- **UI thread:** không chạy OpenCV; worker qua `QThreadPool` (max 1 thread).
- **Ngưỡng tập trung:** `configs/thresholds.yaml` (không chỉnh trong scope báo cáo).

Sơ đồ đầy đủ: [`docs/architecture.md`](../architecture.md) (ASCII layer diagram).

**Hình 1** — Giao diện tab Phân tích (ảnh synthetic, không dữ liệu nhạy cảm):

![Tab Phân tích](figures/gui_tab_analyze.png)

**Hình 2** — Cue block-DCT và phổ tần số trên ảnh synthetic:

![Block-DCT và phổ](figures/freq_block_spectrum.png)

**Hình 3** — Ví dụ localization keypoint trên forge synthetic:

![Keypoint synthetic](figures/copymove_synthetic_keypoint.png)

Các tab Enhance, History, Settings, Stats: `figures/gui_tab_*.png` — metadata
trong [`figures.md`](figures.md).

[SINH VIÊN BỔ SUNG: sơ đồ layer H1 và luồng sequence H2 nếu giảng viên yêu cầu
— hiện chưa có file vector trong repo; vẽ theo `architecture.md`.]

### 3.2 Luồng dữ liệu một lần phân tích

```
file → safe_open → ImageDocument → preprocess/theo method
  → method.analyze() → AlgorithmResult[]
  → fusion.adjudicate() → ForensicReport
  → artifacts/{id}/ + SQLite (relative paths)
```

Ingest folder gọi cùng `AnalysisService` headless, ghi custody — không đổi pipeline
verdict ([`docs/architecture.md`](../architecture.md)).

### 3.3 Dịch vụ (services)

| Service | Vai trò |
| --- | --- |
| `AnalysisService` | orchestrate methods + fusion + persist |
| `HistoryService` | truy vấn analyses, reopen |
| `ReportService` | xuất Markdown từ `ForensicReport` |
| `EvaluationService` | CLI eval synth / CoMoFoD |

Giao diện: [`docs/service-interface.md`](../service-interface.md).

### 3.4 Lưu trữ

SQLite `data/imageguard.db` (gitignored): `image_assets`, `analyses`,
`method_results`, `artifacts`, `experiments`. File mask/heat/report JSON trong
`data/artifacts/<analysis_id>/`. Chỉ **đường dẫn tương đối** — không lưu absolute
path ([`docs/database.md`](../database.md)).

### 3.5 Giao diện người dùng

Năm tab: Analyze | Enhance | History | Stats | Settings; `DisclaimerBar` cố định;
wording “dấu hiệu” trong `ui/strings.py` ([`docs/ui-design.md`](../ui-design.md)).

Quyết định kỹ thuật GUI (tóm tắt):

| Quyết định | Lựa chọn | Lý do |
| --- | --- | --- |
| Threading | `QThreadPool` max 1 | Giới hạn RAM |
| Qt bridge | `.copy()` mọi `QImage` | Tránh dangling pointer |
| Aux methods | Chỉ tăng INCONCLUSIVE | Tránh overclaim |

### 3.6 Bảo mật và lỗi

`safe_open`: magic byte, giới hạn kích thước/decode; không thực thi payload ảnh
([`docs/security-error-handling.md`](../security-error-handling.md)).

**Nguồn:** [`docs/architecture.md`](../architecture.md); [`docs/database.md`](../database.md);
[`docs/ui-design.md`](../ui-design.md); [`docs/report/figures.md`](figures.md).

---

## Chương 4 — Phương pháp thực nghiệm

### 4.1 Ranh giới đo lường

Metrics đo **dấu hiệu suspicious** và localization cue so với oracle — **không**
chứng minh authentic/giả cho ảnh bất kỳ. `INCONCLUSIVE` báo tách; ELA/noise/freq
không tạo `SUSPICIOUS` độc lập ([`docs/evaluation-methodology.md`](../evaluation-methodology.md)).

### 4.2 Bộ dữ liệu

| Dataset | Vai trò | Ghi chú |
| --- | --- | --- |
| **synthforge** | Dev/regression nội bộ | PCG64 seed, tiers easy/medium/hard; không benchmark công bố ([`docs/dataset-strategy.md`](../dataset-strategy.md)) |
| **CoMoFoD** | Diagnostic / dev ngoài | Tralic et al. ELMAR-2013; tải tay vào `data/comofod_base/`; verify mask trước eval ([`docs/dataset-strategy.md`](../dataset-strategy.md)) |
| **CASIA (cohort formal)** | Sự kiện locked | **103 cặp** ảnh đã sàng lọc: 103 chỉnh + 103 gốc; oracle mask tác giả; **không** coi là cohort ảnh chụp tùy ý ([`README.md`](../../README.md)) |

Split synth/CoMoFoD: theo `base_image_id` (`sha256 % 2`) — mọi biến thể cùng ảnh
gốc cùng phía dev hoặc test; **không hiệu chỉnh ngưỡng trên split test khóa**.

[SINH VIÊN BỔ SUNG: điền n và seed cụ thể của lần chạy synth/CoMoFoD sau khi mở
JSON eval — không chép nếu chưa verify file.]

### 4.3 Metric

**Ảnh mức:** precision, recall, F1, specificity; confusion với lớp dương =
`SUSPICIOUS`; đếm `INCONCLUSIVE` riêng.

**Pixel mức (R1–R5):**

- **R1:** chỉ chấm localization khi GT mask không rỗng.
- **R2:** mask dự đoán rỗng trên GT có vùng → recall/F1/IoU = 0, vẫn trong mẫu số.
- **R3:** mask không rỗng trên ảnh authentic → false cue (area ratio).
- **R4:** authentic sạch → điều khiển, loại khỏi aggregate pixel.
- **R5:** split thiếu lớp → metric phụ thuộc lớp = `N/A`.

Bổ sung ModelBench: **IoU**, **boundary-F1@8** trên ảnh chỉnh có GT
([`docs/evaluation-methodology.md`](../evaluation-methodology.md)).

### 4.4 Giao thức (leakage, manifest, locked event)

1. **Cô lập theo scene:** nhóm theo ID ảnh gốc / pixel gốc; không trộn biến thể
   cùng base giữa dev và test.
2. **Manifest đóng băng:** `scripts/evaluate_detector_quality.py freeze` — hash file,
   mask nhị phân; protocol v2 trong [`docs/evaluation-methodology.md`](../evaluation-methodology.md).
3. **Dev:** chẩn đoán và đề xuất candidate; **locked:** baseline/candidate với
   `--pair-contract`, seed cố định, role receipt; không chỉnh candidate sau khi
   xem output test.
4. **So sánh:** `compare` với tiêu chí **đăng ký trước** (hash criteria trong contract).

Lệnh eval thường: `python -m imageguard eval` — ghi experiment id, config hash,
git SHA ([`docs/user-guide.md`](../user-guide.md)).

### 4.5 Kết quả baseline formal (CASIA 103+103) — chỉ số đã công bố trong repo

Theo [`README.md`](../../README.md) và [`docs/limitations.md`](../limitations.md)
(từ một sự kiện đánh giá formal trên cohort đã nêu):

| Phương pháp | Trên ảnh chỉnh (n=103) | Trên ảnh gốc (n=103) |
| --- | --- | --- |
| Keypoint | 94 suspicious; 9 INCONCLUSIVE | 22 FP suspicious; 59 INCONCLUSIVE (trong nhóm nonsuspicious confusion) |
| Block | **37** suspicious; 2–3 INCONCLUSIVE | **0** FP suspicious |

- Keypoint edited recall ~**91,26%** trên cohort này; original FPR ~**21,36%** (22/103).
- Macro IoU mask (103 chỉnh): keypoint ~**0,461**; block ~**0,289**.
- **Fusion UI:** `NOT_EVALUATED` trong sự kiện hai-method trên.

[SINH VIÊN BỔ SUNG: trích lại từ JSON sự kiện đã lưu khi nộp bản chính thức —
README chỉ là tóm tắt định tính.]

### 4.6 Nghiên cứu candidate (block) — kết quả **FAIL**

Mục tiêu đã đăng ký (ví dụ trên cùng cohort formal): cải thiện **recall ảnh mức**
(block coi `SUSPICIOUS` trên ảnh chỉnh) ít nhất **+5** so với baseline, đồng thời
giữ FPR gốc trong ngưỡng đã duyệt.

| Vai trò | Số ảnh chỉnh được coi suspicious (trên 103) | Ghi chú |
| --- | --- | --- |
| Baseline (config khóa) | **37/103** | Khớp bảng formal trong [`README.md`](../../README.md) |
| Candidate | **38/103** | Cải thiện **+1**, không đạt ngưỡng **+5** |

**Kết luận sự kiện:** `compare` **rejected** — candidate **không được chấp nhận**;
ngưỡng production **không** đổi theo kết quả này.

[SINH VIÊN BỔ SUNG: dán `comparison.json` / receipt hash từ `evaluate_detector_quality.py compare`.]

**Nguồn:** [`docs/evaluation-methodology.md`](../evaluation-methodology.md);
[`scripts/evaluate_detector_quality.py`](../../scripts/evaluate_detector_quality.py);
[`README.md`](../../README.md); [`docs/dataset-strategy.md`](../dataset-strategy.md).

---

## Chương 5–7 — TODO (chỉ khung, chưa viết kết quả đầy đủ)

> **Quy tắc:** mọi số trong các chương sau phải dán từ `reports/*.json`, CSV eval,
> hoặc log đã lưu — không suy từ README nếu chưa mở file. Đánh PASS/FAIL/NOT_RUN
> trung thực.

### Chương 5 — Kết quả thực nghiệm (TODO)

- [ ] **Bảng 1:** macro F1/IoU từng method — synth dev — artifact: `reports/…json`
- [ ] **Bảng 2:** CoMoFoD dev (nếu có run) — artifact: …
- [ ] **Bảng 3:** nhân bậc formal CASIA (đã liệt kê sơ §4.5) — đối chiếu JSON gốc
- [ ] **Bảng 4:** thời gian/RAM (nếu đo) — log bench hoặc NOT_RUN
- [ ] Fusion app verdict metrics: `NOT_EVALUATED` hoặc artifact riêng

**Nguồn gợi ý:** [`docs/report/capstone-skeleton.md`](capstone-skeleton.md) mục
“Thực nghiệm & kết quả”; [`docs/experiments.md`](../experiments.md).

### Chương 6 — Thảo luận và phân tích lỗi (TODO)

- [ ] ≥3 ca INCONCLUSIVE: ID ảnh, method, lý do (texture, cancel, mask rỗng…)
- [ ] Phân biệt FN vs INCONCLUSIVE
- [ ] So sánh keypoint vs block theo [`docs/failure-cases.md`](../failure-cases.md)
- [ ] Robustness JPEG/rotate: `evaluation/robustness.py` — NOT_RUN nếu chưa chạy

### Chương 7 — Kết luận và demo (TODO)

- [ ] Checklist MVP (GUI, history, export md, eval CLI)
- [ ] Script demo 3–5 phút: mở ảnh → chạy method → đọc banner → overlay → export
- [ ] Hạn chế & hướng mở rộng: [`docs/splicing-study.md`](../splicing-study.md),
  [`docs/packaging.md`](../packaging.md)
- [ ] Phụ lục tái lập: [`docs/environment.md`](../environment.md),
  [`docs/testing-strategy.md`](../testing-strategy.md)

---

## Phụ lục A — Tóm tắt (Abstract) [TODO]

Điền 150–250 từ sau khi hoàn thiện chương 5–7; không quảng bá “độ chính xác tổng”.
