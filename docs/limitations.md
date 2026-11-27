# Giới hạn đã biết

Tài liệu này tóm tắt **giới hạn đã ghi trong repo** — không thêm con số mới. Đồ án chỉ báo *dấu hiệu nghi vấn*, không khẳng định ảnh giả hay nguyên bản ([`README.md`](../README.md)).

## Kết quả đo trên cohort CASIA đã sàng lọc (103 cặp)

Nguồn số: bảng trong [`README.md`](../README.md), tóm tắt bằng chứng redact [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md), và khung đánh giá ([`evaluation-methodology.md`](evaluation-methodology.md), [`experiments.md`](experiments.md)). Oracle mask tác giả; **không** chứng minh cho ảnh camera tùy ý.

### Cách chọn cohort và giả định oracle (examiner-facing)

Theo bản ghi sự kiện formal (chi tiết + SHA-256 trong [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md)):

- Intake cố định **144** cặp; **7** cặp quarantine (geometry hoặc alias RGB chính xác); **137** survivors.
- Formal chọn **103** cặp (rank 1–103); **34** cặp rank 104–137 **held-aside** — không chạy detector trong sự kiện.
- Oracle: chiếu mask tác giả **blue-complement** (đỏ/xanh lá = foreground, xanh dương = pristine), giữ geometry/orientation.
- **Giả định đã chấp nhận, chưa được chứng minh độc lập:** annotate tác giả đúng vùng; các scene đủ tách biệt — screening crop/RGB/dHash **không** chứng minh độc lập viewpoint hoặc relatedness sau biến đổi.

| Phương pháp | Trên ảnh đã chỉnh (103) | Trên ảnh gốc (103) |
| --- | --- | --- |
| **Keypoint** | 94 coi suspicious; 9 INCONCLUSIVE trên ảnh chỉnh | 22 FP suspicious; **59 INCONCLUSIVE** |
| **Block** | 37 coi suspicious; 3 INCONCLUSIVE | **0** FP suspicious; **0** INCONCLUSIVE |

Ý nghĩa:

- Keypoint trên ảnh gốc: nhiều kết quả **INCONCLUSIVE** (59/103) — abstain là hành vi có chủ đích khi texture kém ([`docs/algorithms/02-keypoint.md`](algorithms/02-keypoint.md)).
- Block trên ảnh gốc: **0/103** FP suspicious trên mẫu này — README nhấn mạnh *không* suy ra FPR luôn bằng 0 mọi nguồn ảnh.
- Block trên ảnh chỉnh: không phải mọi ảnh đều suspicious (66 không suspicious trong cohort).

Macro IoU (mask cue, 103 ảnh chỉnh): keypoint ~0,461; block ~0,289 — baseline trong tài liệu so sánh ([`docs/algorithms/comparison.md`](algorithms/comparison.md)).

**Verdict fusion** trong app/UI **chưa** được đánh giá trong sự kiện đo hai method đơn lẻ trên; kết quả tổng hợp GUI có thể khác pipeline đo từng method ([`README.md`](../README.md)).

Qualification trên CoMoFoD dev **không** được coi là bằng chứng chuyển sang cohort CASIA trên ([`README.md`](../README.md)).

## Giới hạn thuật toán (theo tài liệu thiết kế)

| Chủ đề | Tóm tắt | Tài liệu |
| --- | --- | --- |
| ELA chỉ JPEG | PNG/không JPEG → cue không áp dụng hoặc không đứng một mình | [`docs/algorithms/comparison.md`](algorithms/comparison.md), [`docs/algorithms/03-block.md`](algorithms/03-block.md) |
| Block DCT vs rotate/scale | Biến đổi hình học lớn → hệ số DCT đổi; copy-move rotate/scale có thể **NO_INDICATION** | [`docs/algorithms/03-block.md`](algorithms/03-block.md) |
| Keypoint vs JPEG/nhiễu | Descriptor suy giảm sau nén nặng; vùng phẳng → INCONCLUSIVE | [`docs/algorithms/02-keypoint.md`](algorithms/02-keypoint.md) |
| Splicing cues | Suy giảm trên non-JPEG; exploratory | [`docs/algorithms/comparison.md`](algorithms/comparison.md), [`docs/splicing-study.md`](splicing-study.md) |

## Enhance / AI chỉ xem

Tab Enhance và operator **NAFNet** tạo ảnh xem — **không** gọi `AnalysisService`, **không** đưa pixel enhanced vào fusion verdict ([`docs/user-guide.md`](user-guide.md), [`docs/ui-design.md`](ui-design.md), [`docs/enhancement-study.md`](enhancement-study.md)).

Study cổ điển đã ghi trên cohort thiết kế; arm **NAFNet (E3)** trên nhánh *easy* 256×256, **N=24**: trạng thái **NOT_RUN** (một ảnh 256 thử đơn lẻ không thay thế study đầy đủ) — [`docs/enhancement-study.md`](enhancement-study.md).

## Dataset và đánh giá

- Metric R1–R5 và `INCONCLUSIVE` tách khỏi ma trận nhị phân — xem [`docs/evaluation-methodology.md`](evaluation-methodology.md).
- Artifact đánh giá chi tiết nằm ngoài repo công khai; khi trích số trong báo cáo đồ án, lấy từ [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md) hoặc artifact đã lưu cục bộ, không suy diễn ([`README.md`](../README.md)).

## Camera và deepfake khám phá

Mốc cùng phiên và burst 4–6 khung chỉ thêm tín hiệu trên khung gốc. Chúng không
có cohort đo trên camera thật, không nhận diện một người, và không đổi hợp đồng
`deepfake_cues`: không bao giờ `SUSPICIOUS_INDICATION`. Burst đọc trong một phiên
mở camera với khoảng cách danh nghĩa 0.75 giây/khung; người thật đứng yên vẫn có
thể cho điểm tĩnh cao, còn video phát lại có chuyển động vẫn có thể né cue.
Driver camera có thể buffer frame nên cadence là danh nghĩa, không phải timestamp
phần cứng. Manifest cũ thiếu sha256 bị bỏ qua vì không thể chứng minh thuộc đúng
primary PNG; hash ngăn ghép nhầm/artifact chỉnh lẻ, không chống một người dùng
local cố ý sửa đồng bộ cả file và manifest. Mốc tham chiếu là bản PNG lossless
hết hạn sau 8 giờ; expiry và hash chỉ giảm nguy cơ dùng nhầm mốc cũ/bị sửa,
không chứng minh mốc được chụp live hay cùng một người. Cả
hai cue chỉ áp dụng thay đổi điểm/verdict khi ROI được phát hiện là mặt. Ảnh đã
tăng cường vẫn không được đưa vào detector.

Giới hạn đo được của trụ FFT (cohort synthetic 20 seed, 2026-11-24, không phải
camera thật): sau khi hiệu chuẩn `peak_score` về `(peak_ratio - 4.0)/14.0` để
khung webcam tự nhiên (peak ~8) không còn `freq_cue = 1`, các lưới upsampling
biên hài thực (NN 2×, peak ~6.9–7.4) rơi vào dải mù 4–15 và chỉ được chấm điểm
thấp riêng lẻ ở trụ này — có thể still bị phát hiện qua các trụ σ/LBP khác.
`grid_score` (độ tập trung đỉnh HF, `fft_hf_conc8`) bắt được lưới tuần hoàn kiểu
GAN/Diffusion (chu kỳ 3px, `fft_hf_conc8` ~0.046) an toàn với mặt (~0.003) và
nhiễu thuần (~0.003), nhưng có thể bật trên texture tuần hoàn tự nhiên (vải
dệt, nan chớp). Khi chưa định vị được vùng mặt, đỉnh phổ và biên của nền lặp
cần thêm tín hiệu kết cấu để bật cờ. Quy tắc này vẫn chỉ phục vụ sàng lọc.

## Deepfake ML (`deepfake_ml`, GenConViT-ED)

Method ML đầu tiên, khác hẳn `deepfake_cues`: classifier train trên
DFDC/FF++/Celeb-DF/DeepfakeTIMIT nên CÓ THỂ trả `SUSPICIOUS_INDICATION` khi
`fake_frac ≥ fake_thresh` (mặc định 0.7). Giới hạn:

- Chỉ áp dụng khi phát hiện được khuôn mặt đủ lớn (YuNet); ảnh không có mặt
  → `NOT_APPLICABLE`. Nhiều mặt: chỉ chấm mặt lớn nhất.
- `fake_frac` là tỉ lệ sigmoid heuristic của classifier, KHÔNG phải xác suất
  đã hiệu chuẩn. Ngưỡng 0.7/0.3 là provisional — chỉ mới kiểm chứng tách
  real/fake trên cohort mẫu nhỏ (12 video từ repo tác giả, nguồn không độc
  lập với model), chưa hiệu chuẩn trên dữ liệu thực địa. Cohort smoke 12 video mẫu
  (2026-11-27): pipeline chạy end-to-end đúng; 2 video fake theo tên file đạt
  0.90/0.96 fake_frac; video khớp pattern tên gốc FF++ (`28__talking_angry_couch`)
  đạt 0.12 — nhất quán orientation đúng; 5/12 video đạt ≤0.12. Tuy nhiên cohort
  không có ground-truth đáng tin nên KHÔNG chứng minh độ chính xác.
- Face detection dùng YuNet thay dlib của pipeline gốc — sai lệch tiền xử
  lý đã biết; mặt nghiêng/che khuất có thể bị bỏ sót → `NOT_APPLICABLE`
  thay vì cảnh báo.
- Generator/codec/nén mạng xã hội ngoài phân phối train (2023-2024) có thể
  trượt — deepfake mới hơn có thể không bị gắn cờ, ảnh thật nén nặng có thể
  bị điểm cao. `NO_INDICATION` không chứng minh ảnh thật.
- Weights gốc CC BY-NC 4.0 — chỉ dùng phi thương mại (docs/third-party-models.md).
- `SUSPICIOUS_INDICATION` từ method này promote overall verdict qua fusion —
  vẫn là dấu hiệu cần đối chiếu chuyên gia, không phải bằng chứng giả mạo.

## Bảo mật đầu vào (không phải độ chính xác forensics)

Giới hạn tài nguyên `safe_open` (100 MB, 80 MP) và SSRF khi bật ingest HTTP — [`docs/security-error-handling.md`](security-error-handling.md), [`docs/ingest.md`](ingest.md). DNS rebinding vẫn là rủi ro còn lại khi dùng HTTP snapshot.

`scripts/validate_camera_burst.py` cung cấp positive control bằng các file frozen identical để kiểm tra logic end-to-end; đây vẫn không thay thế cohort replay thật qua màn hình/thiết bị phát lại.
