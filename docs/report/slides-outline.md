# Dàn ý trình bày 10 phút — ImageGuard Forensics

- **Thời lượng:** khoảng 10 phút, 14 slide (~40–45 giây/slide), không tính Q&A.
- **Đối tượng:** giảng viên và sinh viên môn Xử lý ảnh.
- **Nguyên tắc số liệu:** mọi con số **về kết quả đo** dưới đây được chép từ `README.md`, `docs/limitations.md`,
  `docs/performance.md` hoặc `docs/algorithms/comparison.md` và ghi nguồn ngay cạnh.
  Không có số nào được suy ra, làm tròn thêm hay lấy từ nguồn ngoài bốn tài liệu đó.
- **Trạng thái tài liệu:** dàn ý. Các lệnh nêu trong bài là lệnh có sẵn trong repo,
  **chưa** được chạy lại cho buổi trình bày.

## Bảng phân bổ thời gian

| Slide | Nội dung | Thời lượng |
| --- | --- | --- |
| 1 | Tiêu đề | 20 s |
| 2 | Ranh giới khoa học | 45 s |
| 3 | Kiến trúc và luồng dữ liệu | 40 s |
| 4 | Copy-move bằng keypoint | 55 s |
| 5 | Copy-move bằng block DCT | 50 s |
| 6 | ELA và EXIF | 40 s |
| 7 | Tín hiệu phụ và fusion | 40 s |
| 8 | Giao diện và lịch sử | 40 s |
| 9 | Kết quả CASIA | 70 s |
| 10 | Hiệu năng | 45 s |
| 11 | Candidate bị từ chối | 40 s |
| 12 | Giới hạn | 60 s |
| 13 | Tái lập và kiểm thử | 35 s |
| 14 | Kết luận | 30 s |

---

## Slide 1 — ImageGuard Forensics: dấu hiệu, không phải phán quyết

**Bullets**

- Công cụ desktop phân tích **dấu hiệu nghi vấn** trên ảnh số — đồ án môn Xử lý ảnh
  (`README.md`).
- Python 3.14 + PySide6 + OpenCV/NumPy; engine forensics chạy headless, GUI chỉ là lớp hiển thị
  (`README.md`, `docs/architecture.md`).
- Câu hỏi dẫn của bài: ảnh này có dấu hiệu copy-move hoặc bất thường nén/metadata nào đáng xem lại?
  (`docs/algorithms.md`).

**Speaker note:** Đồ án không trả lời “ảnh thật hay ảnh giả”, mà trả lời “có dấu hiệu nào cần
kiểm tra lại không”. Mục tiêu của bài là chỉ ra cách các dấu hiệu đó được tính và được kiểm chứng.

**Hiển thị:** `docs/report/figures/gui_tab_analyze.png`.

---

## Slide 2 — Ranh giới khoa học: indication, không phải verdict

**Bullets**

- Công cụ chỉ báo dấu hiệu có thể liên quan tới chỉnh sửa copy-move hoặc bất thường nén/metadata —
  **không** khẳng định ảnh giả, **không** chứng minh ảnh nguyên bản (`README.md`).
- Khi không đủ tín hiệu, verdict là `INCONCLUSIVE`; báo cáo tổng hợp chỉ là gợi ý điều tra
  (`README.md`).
- ELA/EXIF/noise là bằng chứng khám phá và không bao giờ tự đưa ảnh lên `SUSPICIOUS_INDICATION`
  (`README.md`, `docs/algorithms.md`).

**Speaker note:** Đây là ràng buộc thiết kế chứ không phải câu nói giảm nhẹ: detector copy-move
quyết định verdict, các cue phụ chỉ khoanh vùng đáng xem.

**Hiển thị:** `README.md`, mục “Ranh giới khoa học”.

---

## Slide 3 — Kiến trúc: một engine, nhiều lối vào

**Bullets**

- Ba lớp: engine forensics Qt-free (headless) → services (Analysis/History/Eval/Report) → UI PySide6
  (`docs/architecture.md`).
- Luồng một ảnh: `safe_open` → preprocess → `method.analyze` → `fusion.adjudicate` →
  `ForensicReport` → artifact + SQLite (`docs/architecture.md`).
- Ngưỡng thuật toán tập trung trong `configs/thresholds.yaml`; artifact lưu đường dẫn tương đối,
  không lưu đường dẫn tuyệt đối (`docs/architecture.md`).

**Speaker note:** Tách engine khỏi Qt cho phép CLI, GUI và test dùng chung đúng một pipeline.
Nhờ vậy kết quả trong bài có thể tái lập ngoài giao diện.

**Hiển thị:** `docs/architecture.md` (sơ đồ lớp và data flow).

---

## Slide 4 — Copy-move bằng keypoint (SIFT/ORB)

**Bullets**

- Pipeline: detect keypoint/descriptor → self-match + Lowe ratio → RANSAC affine → displacement
  clustering → mask ứng viên (`docs/algorithms/02-keypoint.md`).
- Mask là hợp của cả hai phía; `score` là số cặp verified, **không** phải xác suất
  (`docs/algorithms/02-keypoint.md`).
- Vùng phẳng/texture lặp hoặc copy quá nhỏ → `INCONCLUSIVE`; JPEG nén mạnh làm descriptor suy giảm
  (`docs/limitations.md`).

**Speaker note:** Keypoint mạnh ở twin copy-move có rotate/scale, nhưng chủ động bỏ qua khi ảnh
không đủ texture thay vì đoán bừa.

**Hiển thị:** `docs/report/figures/copymove_synthetic_keypoint.png`.

---

## Slide 5 — Copy-move bằng block DCT

**Bullets**

- Dense block B×B: DCT + lượng tử → lexicographic sort → shift voting → rasterize mask
  (`docs/algorithms/03-block.md`).
- Flat-block filter (loại block `std < ε`) chống false positive ở vùng trời/tường trơn
  (`docs/algorithms/03-block.md`).
- Chỉ bắt copy dạng translate; copy + rotate/scale/flip → **thường** `NO_INDICATION`
  (`docs/algorithms/03-block.md`, `docs/limitations.md`).

**Speaker note:** Block bắt được vùng ít góc mà keypoint bỏ sót, nhưng đổi hình học là mất dấu vì
hệ số DCT thay đổi hoàn toàn.

**Hiển thị:** `docs/report/figures/freq_block_spectrum.png` (heatmap block-DCT và phổ FFT).

---

## Slide 6 — ELA và EXIF: cue khám phá

**Bullets**

- ELA so ảnh với bản JPEG re-encode ở nhiều mức quality, lấy heatmap cực đại
  (`docs/algorithms.md`).
- ELA **chỉ áp dụng cho JPEG**; re-save xoá vệt, cạnh sắc/texture tự nhiên cũng cho ELA cao
  (`docs/limitations.md`).
- EXIF audit đọc Software/DateTime/Make-Model để tạo `findings`; metadata là dữ kiện khai báo,
  không tham gia verdict (`docs/algorithms.md`).

**Speaker note:** ELA khoanh vùng “đáng xem”, không bao giờ tự kết luận. Vì vậy trong app nó tối đa
chỉ đóng góp `INCONCLUSIVE`.

**Hiển thị:** `docs/algorithms.md`, mục ELA và EXIF audit.

---

## Slide 7 — Tín hiệu phụ và fusion thận trọng

**Bullets**

- Bốn nhóm cue phụ: noise σ-MAD, `noise_local` hai tầng, frequency (spectrum/blockiness),
  splicing (noise × ELA) — tất cả là exploratory (`docs/algorithms.md`).
- Fusion: `SUSPICIOUS_INDICATION` cần ít nhất một detector copy-move; nếu mọi method chạy được đều
  `NO_INDICATION` thì kết quả là `NO_INDICATION`; còn lại `INCONCLUSIVE`
  (`src/imageguard/forensics/fusion.py`).
- Cue phụ chỉ tạo nhãn corroboration để reviewer chú ý, không nâng verdict (`docs/algorithms.md`).

**Speaker note:** Fusion ở đây là luật thận trọng, không phải mô hình học máy: càng thiếu bằng chứng,
kết quả càng dễ là `INCONCLUSIVE`.

**Hiển thị:** lệnh CLI trong `README.md`:
`python -m imageguard analyze <ảnh> --methods keypoint,block,ela`.

---

## Slide 8 — Giao diện: phân tích, lịch sử, xuất báo cáo

**Bullets**

- Tab Analyze hiển thị overlay mask/heatmap; worker trả `ForensicReport` hoặc `ErrorResult` qua signal;
  `AlgorithmResult` nằm trong `report.methods` — worker không chạm widget (`docs/ui-design.md`).
- History đọc bảng `analyses` từ SQLite; mở lại, xoá phiên và xuất Markdown qua `ReportService`
  (`docs/ui-design.md`, `docs/user-guide.md`).
- Tab Enhance chỉ để xem; pixel enhanced **không** được đưa vào fusion verdict
  (`docs/limitations.md`).

**Speaker note:** Kết quả không chỉ nằm trên màn hình: mỗi phiên còn có artifact JSON và bản xuất
Markdown để đối chiếu lại.

**Hiển thị:** `docs/report/figures/gui_tab_history.png` và
`docs/report/figures/gui_tab_enhance.png`.

---

## Slide 9 — Kết quả đo trên cohort CASIA

**Bullets**

- Sự kiện đánh giá chính thức trên **103 cặp** CASIA đã sàng lọc — receipt redact
  `docs/casia-formal-evidence-summary.md`; oracle mask tác giả —
  không phải ảnh chụp camera tùy ý (`README.md`).
- Keypoint: **94/103** ảnh đã chỉnh bị coi suspicious, **9** `INCONCLUSIVE`; trên ảnh gốc có
  **22/103** FP và **59/103** `INCONCLUSIVE` (`README.md`).
- Block: **37/103** ảnh đã chỉnh bị coi suspicious; **66** không suspicious (gồm **3** `INCONCLUSIVE`
  và **63** `NO_INDICATION` — [`docs/fusion-offline-analysis.md`](../fusion-offline-analysis.md));
  trên ảnh gốc **0/103** FP suspicious (`README.md`, `docs/limitations.md`).
  Bảng so sánh method không có cột `INCONCLUSIVE` trên ảnh chỉnh (`docs/algorithms/comparison.md`).
- Keypoint recall ~**91,26%**, original FPR ~**21,36%**; macro IoU mask **0,461** (keypoint) so với
  **0,289** (block), baseline **.01** (`README.md`, `docs/algorithms/comparison.md`).

**Speaker note:** Phải đọc verdict rate và IoU cùng nhau: recall cao không đồng nghĩa localization
tốt, còn FPR bằng 0 chỉ đúng trên mẫu này. Đây là kết quả đo dấu hiệu theo oracle dataset, không
chứng minh ảnh thật/giả.

**Hiển thị:** `README.md`, mục “Giới hạn đã đo (không overclaim)”.

---

## Slide 10 — Hiệu năng: chi phí thực tế

**Bullets**

- Đo median qua 3 lần lặp, ảnh synth 256/512/1024/2048 px, seed **424242**, trên một máy
  — **một lần chạy**, JSON tạm **chưa archive** trong repo (`docs/performance.md`).
- Tại 2048 px: keypoint **47.603493499998876 s** (chậm nhất trong **method đơn lẻ**),
  block **6.8929092999969725 s**; `analysis_service` **51.22097900000517 s** (hàng **tổng hợp**)
  (`docs/performance.md`).
- Tại 256 px: keypoint **0.028280600003199652 s**, `analysis_service`
  **0.5253770999988774 s** (`docs/performance.md`).
- Đỉnh bộ nhớ: **NOT_RUN** — script cố ý bỏ qua vì `tracemalloc` không đại diện cho OpenCV
  (`docs/performance.md`).

**Speaker note:** Nút thắt là keypoint ở ảnh lớn; block chậm ở ảnh nhỏ nhưng tăng chậm hơn nhiều
theo kích thước. Số này là thời gian trên một máy, không phải claim độ chính xác.

**Hiển thị:** `docs/performance.md`, bảng `median_s`; lệnh tái lập `scripts/bench_methods.py`.

---

## Slide 11 — Một candidate bị từ chối

**Bullets**

- Trước sự kiện, tiêu chí đã đăng ký: cải thiện recall ảnh mức (block coi suspicious trên ảnh chỉnh)
  ít nhất **+5** so với baseline, FPR giữ (`docs/report/capstone-draft.md`, mục 4.6); cơ chế
  candidate gắn với hợp đồng đánh giá (`docs/evaluation-methodology.md`).
- `compare` kết luận **rejected**: candidate không được chấp nhận, ngưỡng production không đổi
  (`docs/report/capstone-draft.md`, mục 4.6).
- Kết quả âm tính được giữ nguyên trong báo cáo thay vì chỉnh ngưỡng sau khi xem output test
  (`docs/evaluation-methodology.md`).

**Speaker note:** Đây là slide quan trọng nhất về phương pháp: một cải tiến không đạt tiêu chí vẫn
được ghi lại trung thực. Không có con số nào được thêm vào slide này vì nguồn số của sự kiện đó nằm
ngoài bốn tài liệu được phép trích.

**Hiển thị:** `docs/report/capstone-draft.md`, mục 4.6 “Nghiên cứu candidate (block) — kết quả FAIL”.

---

## Slide 12 — Giới hạn: điều gì chưa chứng minh được

**Bullets**

- Đây là prototype exploratory của đồ án; cohort CASIA đã sàng lọc với oracle mask **không** đại diện
  cho ảnh camera tùy ý (`README.md`, `docs/limitations.md`).
- Verdict fusion trong app/UI **chưa** được đánh giá trong sự kiện hai-method; kết quả GUI có thể
  khác pipeline đo từng method (`README.md`).
- Block **0/103** FP chỉ đúng trên mẫu này; keypoint **59/103** `INCONCLUSIVE` trên ảnh gốc là
  abstain có chủ đích khi texture kém (`README.md`, `docs/limitations.md`).
- Qualification trên CoMoFoD dev không được coi là bằng chứng chuyển giao; arm NAFNet E3
  **N=24** đang **NOT_RUN** (`README.md`, `docs/limitations.md`).

**Speaker note:** Không có “độ chính xác tổng” nào được tuyên bố. Tất cả chỉ là kết quả đo theo
cohort kèm giới hạn, và các giới hạn này được nêu ngay trong bài chứ không giấu ở phụ lục.

**Hiển thị:** `docs/limitations.md`.

---

## Slide 13 — Tái lập: hình, bảng và test đều có nguồn

**Bullets**

- Ảnh synth và fixture dùng seed cố định; cùng seed cho cùng bytes (`docs/testing-strategy.md`).
- Hình trong bài sinh bằng `scripts/make_report_figures.py`; bảng hiệu năng sinh bằng
  `scripts/bench_methods.py` (`docs/report/figures.md`, `docs/performance.md`).
- Kiểm thử headless: `$env:PYTHONPATH = '<repo>\src'; & '<python>' -m pytest -x -q -p no:cacheprovider`,
  kèm `ruff check` (`docs/testing-strategy.md`).

**Speaker note:** Số trong bài được chép từ tài liệu đã lưu, còn hình có script sinh lại. Nhờ vậy
người chấm có thể kiểm tra thay vì tin vào slide.

**Hiển thị:** lệnh `python scripts/make_report_figures.py` (mục “Tái tạo” của
`docs/report/figures.md`).

---

## Slide 14 — Kết luận

**Bullets**

- Đã có: engine Qt-free cho keypoint/block/ELA/cue phụ, CLI + GUI, artifact JSON và khung đánh giá
  (`README.md`, `docs/architecture.md`).
- Kết quả đo: keypoint mạnh về verdict recall trên cohort, block bảo thủ; localization vẫn là điểm
  yếu (`README.md`).
- Hướng tiếp: đánh giá fusion trong app, cải thiện localize vùng generative-edit, thêm detector cục
  bộ (`docs/algorithms/comparison.md`, `docs/ml-extension.md`).

**Speaker note:** Thông điệp cuối: một công cụ hỗ trợ điều tra có kỷ luật — biết mình đo được gì và
không đo được gì — hữu ích hơn một công cụ tuyên bố phát hiện ảnh giả.

**Hiển thị:** `docs/report/figures/copymove_synthetic_keypoint.png` (đóng vòng từ slide 4).

---

## Bảng truy xuất số liệu

Mỗi dòng dưới đây là một con số xuất hiện trong bài, kèm nguồn duy nhất được phép trích.

| Số liệu | Giá trị trong bài | Nguồn |
| --- | --- | --- |
| Số cặp CASIA | 103 cặp | `README.md` |
| Keypoint, ảnh chỉnh | 94/103 suspicious; 9 `INCONCLUSIVE` | `README.md` |
| Keypoint, ảnh gốc | 22/103 FP; 59/103 `INCONCLUSIVE` | `README.md` |
| Block, ảnh chỉnh | 37/103 suspicious; 66 không suspicious (3 `INCONCLUSIVE` + 63 `NO_INDICATION`) | `README.md`, `docs/fusion-offline-analysis.md` |
| Block, ảnh gốc | 0/103 FP; 0 `INCONCLUSIVE` | `README.md` |
| Keypoint recall / FPR | ~91,26% / ~21,36% | `README.md` |
| Macro IoU | 0,461 (keypoint); 0,289 (block); baseline .01 | `README.md`, `docs/algorithms/comparison.md` |
| Số lần lặp benchmark | 3 | `docs/performance.md` |
| Kích thước benchmark | 256, 512, 1024, 2048 px | `docs/performance.md` |
| Seed ảnh benchmark | 424242 | `docs/performance.md` |
| keypoint 2048 px | 47.603493499998876 s | `docs/performance.md` |
| block 2048 px | 6.8929092999969725 s | `docs/performance.md` |
| `analysis_service` 2048 px | 51.22097900000517 s | `docs/performance.md` |
| keypoint 256 px | 0.028280600003199652 s | `docs/performance.md` |
| `analysis_service` 256 px | 0.5253770999988774 s | `docs/performance.md` |
| Đỉnh bộ nhớ | `NOT_RUN` | `docs/performance.md` |
| Arm NAFNet E3 | N=24, `NOT_RUN` | `docs/limitations.md` |

## Những gì bài này không tuyên bố

- Không nói ảnh là giả hay nguyên bản; không dùng từ “phát hiện forgery”.
- Không đưa ra phần trăm độ chính xác ngoài các tỷ lệ đã có trong `README.md` và
  `docs/algorithms/comparison.md`.
- Không suy FPR = 0 cho mọi nguồn ảnh từ kết quả 0/103 trên mẫu CASIA.
- Không coi kết quả CoMoFoD dev là bằng chứng chuyển giao sang cohort CASIA.
- Không trình bày số của candidate bị từ chối vì nguồn số đó nằm ngoài bốn tài liệu được phép trích.
