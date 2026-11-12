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

## Bảo mật đầu vào (không phải độ chính xác forensics)

Giới hạn tài nguyên `safe_open` (100 MB, 80 MP) và SSRF khi bật ingest HTTP — [`docs/security-error-handling.md`](security-error-handling.md), [`docs/ingest.md`](ingest.md). DNS rebinding vẫn là rủi ro còn lại khi dùng HTTP snapshot.
