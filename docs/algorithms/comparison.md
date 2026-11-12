# Method Comparison — bằng chứng formal CASIA và exploratory nội bộ

Tài liệu này có **hai loại bằng chứng không cùng giá trị**:

1. **Formal (mục 1):** sự kiện CASIA trên cohort đã screen — **evaluation duy nhất
   được chấm điểm** của classical detectors.
2. **Exploratory (mục 2):** đo trên dataset sinh nội bộ (synthforge + modelforge)
   bằng `src/imageguard/evaluation/robustness.py` +
   `src/imageguard/evaluation/modelbench.py`. Đây là mô tả hành vi pipeline trên
   ảnh synth, **không** phải ước lượng độ chính xác và **không** được đọc như
   accuracy của app.

Protocol chung: [`docs/evaluation-methodology.md`](../evaluation-methodology.md).

## 1. CASIA formal criterion-B (paired, screened cohort)

Sự kiện đánh giá formal trên **103 cặp** CASIA đã screen (native oracle, không
phải ảnh chụp camera). Block baseline ngưỡng `.01`; keypoint cùng cấu hình ở cả
baseline và candidate. Kết quả lưu trong artifact đánh giá cục bộ (không nằm
trong repo). Đây là đo *dấu hiệu suspicious* theo oracle dataset, không chứng
minh authentic/giả cho ảnh bất kỳ. Nguồn công khai của các số dưới đây:
[`README.md`](../../README.md) và [`docs/limitations.md`](../limitations.md).

| method | edited recall (TP / 103) | original FP / 103 | original INCONCLUSIVE / 103 |
|---|---:|---:|---:|
| keypoint (SIFT) | 94 / 103 | 22 / 103 | 59 / 103 |
| block (DCT baseline) | 37 / 103 | 0 / 103 | 0 / 103 |

Đọc bảng:

- **Block phải đọc kèm recall 37/103:** block chỉ coi 37/103 ảnh đã chỉnh là
  `SUSPICIOUS` (66/103 không suspicious) — con số **0/103 original FP** chỉ có
  nghĩa trong cặp này, *không* khẳng định FPR luôn bằng 0 với mọi nguồn ảnh.
- Keypoint: recall verdict 94/103 đi cùng 22/103 original `SUSPICIOUS` và **59/103
  original `INCONCLUSIVE`**; recall cao không đồng nghĩa localization đúng.
- Macro IoU trên 103 ảnh đã chỉnh: keypoint ≈ 0.461; block ≈ 0.289 (cùng sự kiện).
  **Verdict rate và IoU phải đọc cùng nhau** — localization không chứng minh vùng giả.
- Block trên ảnh đã chỉnh: 37 `SUSPICIOUS`, 3 `INCONCLUSIVE`, 63 `NO_INDICATION`
  (tổng 103; đếm từ hàng đã lưu bằng `scripts/fusion_offline_summary.py`, xem
  [`docs/fusion-offline-analysis.md`](../fusion-offline-analysis.md)).

**Verdict fusion trong app/UI chưa được chấm điểm riêng** trong sự kiện hai-method
trên; kết quả tổng hợp GUI có thể khác pipeline đo từng method. Đây là bằng chứng
formal chính cho classical detectors; dev CoMoFoD không thay thế sự kiện này.

## 2. Exploratory — synthetic / small-sample (không phải accuracy estimate)

> **Mọi số trong mục này là exploratory.** Chúng KHÔNG phải benchmark chuẩn quốc
> tế, KHÔNG phải ước lượng accuracy hay FPR, và không được cộng gộp với bảng
> CASIA ở mục 1. Nhiều ô có n=3 hoặc n rất nhỏ; phần còn lại của battery có thể là
> `INCONCLUSIVE` — pipeline conservative by design, nên đây không phải kiểm định
> lý thuyết.

### Copy-move detectors (synthforge medium, 256², seed 99, n=3/distortion)

| distortion | keypoint det-rate (exploratory) |
|---|---:|
| none | 0.667 |
| jpeg q90 | 0.667 |
| jpeg q70 | 0.667 |
| rotate 5° | 0.667 |
| scale 0.9 | 0.667 |

Với n=3 mỗi distortion, bảng này chỉ mô tả hành vi định tính; không suy ra tỷ lệ
phát hiện cho ảnh thật. Tái lập: `generate_set(n=3, seed=99, tier=medium,
size=256²)` rồi `run_battery` trên `*_F.png` với method `keypoint`
(deterministic). Artifact JSON SHA-256:
`1257279da963ad6ebcad5a0c497eca8ed022a1b129ce2d93ce24eb35769c030e`.

### Model-edited (modelforge LaMa, n=12, 384², exploratory) — ma trận đầy đủ tại [modelforge.md](../modelforge.md)

| kind | keypoint | block | ý nghĩa |
|---|---|---|---|
| splice_keep | **1.0**, IoU 0.319 (bF1 0.598) | 0.0 | twin sống qua Poisson blend; verdict cao + IoU vừa |
| move | 0.0, IoU 0.000 (3/3 `INCONCLUSIVE`) | 0.0 | gốc bị LaMa xoá → twin phá (holdout 512²: 2 `SUSPICIOUS`, IoU 0.042; mẫu nhỏ) |
| erase / foreign | 0.0 | 0.0 | không twin → đúng luật miss |

Các verdict ở đây là trên ảnh synth n=12; không phải accuracy trên ảnh thật.

## Đặc tính từng method

| method | mạnh | yếu | cost (đo 512², CPU) |
|---|---|---|---|
| keypoint (SIFT) | twin blended/rotated/scaled; mask + pairs | vùng phẳng/texture kém → INCONCLUSIVE; CASIA formal: 94/103 edited recall, 22/103 original FP | ~0.1–0.5s |
| block (DCT) | không cần feature point; baseline kinh điển; CASIA formal: 37/103 edited recall, 0/103 original FP | miss clone đã blend; chậm; flat-guard bắt buộc | ~5–10s |
| ELA | JPEG history cue | chỉ JPEG; re-save phá; không standalone | ~0.1s |
| noise σ-MAD | tín hiệu σ-separation trên model-edit — exploratory (N=12, n=3/kind, seed=2026, config_hash=`df80dcb9a6d32070`, bench JSON SHA-256 `e70d58546d87e515c437563ff0c6b6253d33e47a4669ec2e7a3f7e712c6855ed`); p per-sample không đủ cơ sở thống kê ở n=3/kind nên không trích làm bằng chứng | localize yếu (IoU≈0.005) | ~0.02s |
| noise_local (two-tier) | localize model-fill: IoU 0.335–0.665, bF1 0.559–0.888 (exploratory) | false-positive vùng phẳng tự nhiên; vẫn aux | ~0.05s |
| freq cues | spectrum/blockiness artifact | mô tả hơn phán đoán | ~0.02s |
| splicing cues | cross-cue noise×ELA | suy giảm trên non-JPEG | ~0.05s |

**Cost:** cột cost là ước lượng thô ở 512², không phải số đo chuẩn hoá và có thể
lệch so với bảng `median_s`. Ở ảnh lớn chi phí tăng mạnh —
[`docs/performance.md`](../performance.md) ghi `median_s` của `keypoint` là
**1.6158136000012746 s** tại 1024 px và **47.603493499998876 s** tại 2048 px
(một máy, seed 424242, 3 lần lặp); `analysis_service` tại 2048 px là
51.22097900000517 s.

## Takeaway cho báo cáo

- Không method nào "tốt nhất" — keypoint thắng ở twin-copy-move, aux cues
  thắng ở "có gì đó lạ" trên model-edit; fusion rule tận dụng điều đó.
- Khoảng trống lớn nhất: **localize vùng generative-edited** — holdout
  σ-separation chỉ exploratory (N=6, n=2/kind, seed=7777, config_hash=`df80dcb9a6d32070`,
  bench JSON SHA-256 `33c56479c9743eb020a35144ae89a139c50aa802d942dc36ce62b40707354326`);
  p per-sample ở n=2/kind không đủ cơ sở để trích, nên vẫn cần detector cục bộ hơn →
  [`docs/ml-extension.md`](../ml-extension.md).
