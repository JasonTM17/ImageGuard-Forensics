# Nghiên cứu hiệu chuẩn false-cue keypoint (offline)

## Mục tiêu

Detector copy-move keypoint đang dùng `verified_pairs ≥ t_susp=10` làm tín hiệu nghi vấn. Nhiều ảnh gốc vẫn vượt ngưỡng do false cue; nhiều ảnh khác nằm vùng INCONCLUSIVE. Tài liệu này ghi **một nghiên cứu offline** (không đổi `thresholds.yaml`, không đổi verdict shipped) nhằm xem **một đặc trưng đơn giản** có tách forged/original tốt hơn đếm `n_verified` thuần hay không.

**Không** phải đánh giá CASIA hay cohort niêm phong. **Không** thay đổi ngưỡng production.

## Tái hiện

```powershell
$env:PYTHONPATH = 'src'
$env:QT_QPA_PLATFORM = 'offscreen'
$env:IMAGEGUARD_DATA = '<thư-mục-gốc-chứa-các-dataset-bên-dưới>'
python scripts/keypoint_calibration_study.py --out-json reports/keypoint-calibration-study.json
```

Hoặc: `python scripts/keypoint_calibration_study.py --data-root <thư-mục-data>`.

Seed cố định: `20261112`. Pipeline: `preprocess_keypoint` → `detect` → `self_match` → `geometric_verify` (giống `keypoint/pipeline.py`). AUC = thống kê hạng Mann–Whitney (không sklearn).

### Dataset (chỉ đọc)

| Khóa trong script | Thư mục con | Số file F / O |
|-------------------|-------------|---------------|
| `tuning` | `modelforge_audit_train_fixed` | 12 / 12 |
| `holdout` | `modelforge_audit_holdout_fixed` | 6 / 6 |
| `synth_eval` | `synth_eval` | 6 / 6 |
| `comofod` | `comofod_base` | 200 / 200 |

Chọn ngưỡng phụ (`spatial_spread`, `median_pair_distance_px`) **chỉ** trên split tuning. Holdout, synth_eval và CoMoFoD chỉ báo cáo với ngưỡng đã khóa từ tuning.

### Đặc trưng phụ (từ inlier geometric verify)

1. **spatial_spread** — diện tích bbox mọi keypoint inlier (hai đầu cặp) chia cho diện tích ảnh làm việc.
2. **median_pair_distance_px** — trung vị khoảng cách pixel giữa hai keypoint trong mỗi cặp inlier.

Cổng thử nghiệm (AND với `n_verified ≥ 10`): `n_clusters ≥ 2`, `spatial_spread ≥ x`, `median_pair_distance_px ≥ x` (tune `x` trên tuning).

## Rank AUC trên tuning (n=24, rất nhỏ)

| Mô hình điểm / cổng | AUC |
|---------------------|-----|
| Chỉ `n_verified` | 0,590 |
| + cổng `n_clusters ≥ 2` (điểm = 0 nếu không đạt) | 0,503 |
| + cổng `spatial_spread ≥ 0,224` (tune) | 0,590 |
| + cổng `median_pair_distance_px ≥ 74,53` (tune) | **0,653** |

Trên tuning, **chỉ median khoảng cách cặp** cải thiện AUC so với baseline; cổng cluster làm giảm AUC; spread tune không đổi AUC (cùng điểm số hạng trên mẫu này). Kết luận mang tính **khám phá** — 24 ảnh không đủ để khẳng định shipping.

## Tỉ lệ flag `n_verified ≥ 10` (Wilson 95%)

Đếm: forged_flagged / n_forged, original_flagged / n_original. Ngoặc: Wilson 95% cho **tỉ lệ flag trong split đó** (giả định hàng độc lập chưa chứng minh).

### Split tuning (12F + 12O)

| Cổng | Forged flag | Original flag |
|------|-------------|---------------|
| Baseline | 3/12 (0,09–0,53) | 0/12 (0,00–0,24) |
| `n_clusters ≥ 2` | 1/12 (0,02–0,35) | 0/12 (0,00–0,24) |
| `spread ≥ 0,224` | 3/12 (0,09–0,53) | 0/12 (0,00–0,24) |
| `median_dist ≥ 74,53 px` | 3/12 (0,09–0,53) | 0/12 (0,00–0,24) |

### Split holdout (6F + 6O) — dùng ngưỡng tune, không tune lại

| Cổng | Forged flag | Original flag |
|------|-------------|---------------|
| Baseline | 3/6 (0,19–0,81) | 0/6 (0,00–0,39) |
| `n_clusters ≥ 2` | 2/6 (0,10–0,70) | 0/6 (0,00–0,39) |
| `spread ≥ 0,224` | 2/6 (0,10–0,70) | 0/6 (0,00–0,39) |
| `median_dist ≥ 74,53 px` | 3/6 (0,19–0,81) | 0/6 (0,00–0,39) |

### synth_eval (6F + 6O)

| Cổng | Forged flag | Original flag |
|------|-------------|---------------|
| Baseline | 4/6 (0,30–0,90) | 0/6 (0,00–0,39) |
| `n_clusters ≥ 2` | 0/6 (0,00–0,39) | 0/6 (0,00–0,39) |
| `spread ≥ 0,224` | 0/6 (0,00–0,39) | 0/6 (0,00–0,39) |
| `median_dist ≥ 74,53 px` | 3/6 (0,19–0,81) | 0/6 (0,00–0,39) |

### CoMoFoD (200F + 200O) — kiểm tra bổ sung, không tune

| Cổng | Forged flag | Original flag |
|------|-------------|---------------|
| Baseline | 168/200 (0,78–0,88) | **89/200 (0,38–0,51)** |
| `n_clusters ≥ 2` | 112/200 (0,49–0,63) | 89/200 (0,38–0,51) |
| `spread ≥ 0,224` | 147/200 (0,67–0,79) | 88/200 (0,37–0,51) |
| `median_dist ≥ 74,53 px` | 141/200 (0,64–0,76) | **45/200 (0,17–0,29)** |

CoMoFoD cho thấy false flag trên ảnh gốc **cao** ở baseline; cổng `median_pair_distance_px` **giảm** original flag (89→45) nhưng vẫn còn 45/200; cổng `n_clusters` **không** giảm false flag trên gốc trong tập này.

## Artefact JSON

Chạy script ghi `reports/keypoint-calibration-study.json` (gitignore). Nội dung: counts từng ảnh trên tuning/holdout, meta seed, ngưỡng tune, bảng split đầy đủ.

## Kiểm tra tự động

`tests/test_keypoint_calibration_study.py`: AUC đồng chơi, Wilson validation, ảnh synthetic → đặc trưng hữu hạn.

## Tuyên bố

- **Không** có thay đổi `configs/thresholds.yaml` hay logic verdict shipped.
- **Không** đọc `data/external/**`, CASIA, hay `plans/**`.
- Số liệu bảng trên lấy từ lần chạy script tạo artefact `keypoint-calibration-study.json` trong workspace (study_seed=20261112).
