# Capstone Report Skeleton — ImageGuard Forensics

Khung báo cáo đồ án; mỗi mục map với code/docs thật. Điền số đo TỪ
`reports/` artifacts — không chép số từ tài liệu người khác.

## 1. Giới thiệu & động lực
- Bài toán copy-move/splicing; vì sao cần tool offline giải thích được.
- Phát biểu đóng góp: pipeline đa-method + verdict thận trọng + eval
  deterministic + desktop app.

## 2. Cơ sở lý thuyết
- SIFT (scale space, ratio test) — `docs/algorithms/02-keypoint.md`.
- DCT block matching — `docs/algorithms/03-block.md`.
- ELA, noise residual, JPEG blockiness — `docs/algorithms.md`.
- Fusion rules minh bạch — `forensics/fusion.py` docstring.

## 3. Kiến trúc hệ thống
- Diagram: `docs/architecture.md`; ERD: `docs/database.md`.
- Điểm nhấn: engine Qt-free, verdict enum 5 trạng thái, empty-mask R1–R5.

## 4. Cài đặt & quyết định kỹ thuật
- Bảng: quyết định | lựa chọn | lý do | trade-off.
- Ví dụ: serialized worker vì RAM; `.copy()` vì QImage dangling; aux
  signals tối đa INCONCLUSIVE vì tránh overclaim.

## 5. Đánh giá
- Dataset: synthforge (deterministic) + CoMoFoD.
- Bảng số: F1/IoU macro per method, từ `reports/*.json` THẬT.
- Robustness battery: `evaluation/robustness.py` — detection rate theo
  JPEG q/rotate/scale.
- Failure analysis: ít nhất 3 ảnh INCONCLUSIVE, giải thích nguyên nhân.

## 6. Hạn chế & hướng mở rộng
- Splicing chưa benchmark; resampling-EM cắt scope; ML ở
  `docs/ml-extension.md` (design-only).

## 7. Kết luận & demo
- Checklist MVP; demo flow từ `docs/user-guide.md`.

## Phụ lục
- Thresholds snapshot + config_hash; repro commands; test suite output.
