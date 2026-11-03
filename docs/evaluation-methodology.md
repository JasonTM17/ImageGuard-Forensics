# Evaluation Methodology (draft — P05 sẽ hoàn thiện)

## Nguyên tắc

1. Không claim số chưa đo. Mọi metric ghi: N mẫu, dataset, seed, config hash.
2. `INCONCLUSIVE` là kết quả hợp lệ — tách "không có dấu hiệu" khỏi "không đủ bằng chứng".
3. Aux signals (ELA/EXIF) không thể tự kết luận — exploratory only.

## Verdict fusion (bảng luật chính thức)

| Điều kiện | Overall verdict |
|---|---|
| ≥1 method conclusive trả SUSPICIOUS_INDICATION | SUSPICIOUS_INDICATION (+ signals liệt kê) |
| Tất cả methods chạy được = NO_INDICATION (≥1) | NO_INDICATION |
| Còn lại (mâu thuẫn, thiếu dữ kiện, toàn N/A/ERROR) | INCONCLUSIVE |

NOT_APPLICABLE loại khỏi xét; ERROR → warning, không tính quyết định.

## Two-threshold band (per method)

`score ≥ t_susp` → SUSPICIOUS; `score < t_clean` → NO_INDICATION; giữa →
INCONCLUSIVE. Ngưỡng trong `configs/thresholds.yaml` kèm `source`.

## Metrics dự kiến (P05)

- R1: image-level precision/recall/F1 trên tập có nhãn forged/authentic.
- R2: pixel-level IoU/F1 trên mask GT (quy ước empty-mask: GT rỗng +
  pred rỗng = perfect; GT rỗng + pred non-empty = FP).
- R3: per-transform breakdown (translate/rotate/scale/jpeg).
- R4: INCONCLUSIVE rate — đo độ thận trọng, không phạt.
- R5: runtime per method trên kích thước chuẩn.

## Dataset

- synthforge (in-repo, deterministic seed) — dev/test nội bộ.
- CoMoFoD (external, manual download, không redistribute) — eval công bố.
- Split theo base-image để tránh leakage giữa biến thể cùng nguồn.
