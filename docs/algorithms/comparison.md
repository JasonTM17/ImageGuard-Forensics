# Method Comparison — số đo thật từ benchmark nội bộ

Mọi số trong bảng đo từ `src/imageguard/evaluation/robustness.py` + `src/imageguard/evaluation/modelbench.py`
trên dataset sinh nội bộ (synthforge + modelforge). KHÔNG phải benchmark
chuẩn quốc tế — xem [`docs/evaluation-methodology.md`](../evaluation-methodology.md) cho protocol.

## Copy-move detectors (synthforge medium, 256², seed 99, n=3/distortion)

| distortion | keypoint det-rate |
|---|---|
| none | 0.667 |
| jpeg q90 | 0.667 |
| jpeg q70 | 0.667 |
| rotate 5° | 0.667 |
| scale 0.9 | 0.667 |

Tái lập: `generate_set(n=3, seed=99, tier=medium, size=256²)` rồi `run_battery` trên
`*_F.png` với method `keypoint` (deterministic). Artifact JSON SHA-256:
`1257279da963ad6ebcad5a0c497eca8ed022a1b129ce2d93ce24eb35769c030e`. Phần còn lại
có thể là `INCONCLUSIVE` — pipeline conservative by design; không coi đây là kiểm
định lý thuyết.

## CASIA formal criterion-B (paired, screened cohort)

Sự kiện đánh giá formal trên **103 cặp** CASIA đã screen (native oracle, không
phải ảnh chụp camera). Block baseline ngưỡng `.01`; keypoint cùng cấu hình ở cả
baseline và candidate. Kết quả lưu trong artifact formal (không commit plans/).

| method | edited recall (TP / 103) | original FP / 103 | original INCONCLUSIVE / 103 |
|---|---:|---:|---:|
| keypoint (SIFT) | 94 / 103 | 22 / 103 | 59 / 103 |
| block (DCT baseline) | 37 / 103 | 0 / 103 | 0 / 103 |

Keypoint: macro IoU trên 103 edited ≈ 0.461; block ≈ 0.289 (cùng sự kiện). **Verdict
rate và IoU phải đọc cùng nhau** — ví dụ keypoint có recall verdict cao nhưng vẫn
22/103 original `SUSPICIOUS` và localization không đồng nghĩa chứng minh vùng giả.
Đây là bằng chứng formal chính cho classical detectors; dev CoMoFoD không thay thế
sự kiện này.

## Model-edited (modelforge LaMa, n=12, 384² train sau sửa ground-truth) — ma trận đầy đủ tại [modelforge.md](../modelforge.md)

| kind | keypoint | block | ý nghĩa |
|---|---|---|---|
| splice_keep | **1.0**, IoU 0.319 (bF1 0.598) | 0.0 | twin sống qua Poisson blend; verdict cao + IoU vừa |
| move | 0.0 (3 inc), IoU 0.000 | 0.0 | gốc bị LaMa xoá → twin phá (holdout 512²: 2 sus, IoU 0.042) |
| erase / foreign | 0.0 | 0.0 | không twin → đúng luật miss |

## Đặc tính từng method

| method | mạnh | yếu | cost (đo 512², CPU) |
|---|---|---|---|
| keypoint (SIFT) | twin blended/rotated/scaled; mask + pairs | vùng phẳng/texture kém → INCONCLUSIVE; CASIA formal: 22/103 original FP | ~0.1–0.5s |
| block (DCT) | không cần feature point; baseline kinh điển; CASIA formal: 0/103 original FP | miss clone đã blend; chậm; flat-guard bắt buộc | ~5–10s |
| ELA | JPEG history cue | chỉ JPEG; re-save phá; không standalone | ~0.1s |
| noise σ-MAD | train model-edit σ-separation MWU p per sample (N=12, n=3/kind, seed=2026, config_hash=`df80dcb9a6d32070`, bench JSON SHA-256 `e70d58546d87e515c437563ff0c6b6253d33e47a4669ec2e7a3f7e712c6855ed`): erase_0000=8.803304263477716e-13, erase_0004=0.4566456368698131, erase_0008=0.00635903098503261; move_0001=1.321699692882183e-06, move_0005=6.551271422897137e-08, move_0009=0.001601635814580065 | localize yếu (IoU≈0.005) | ~0.02s |
| noise_local (two-tier) | localize model-fill: IoU 0.335–0.665, bF1 0.559–0.888 | false-positive vùng phẳng tự nhiên; vẫn aux | ~0.05s |
| freq cues | spectrum/blockiness artifact | mô tả hơn phán đoán | ~0.02s |
| splicing cues | cross-cue noise×ELA | suy giảm trên non-JPEG | ~0.05s |

## Takeaway cho báo cáo

- Không method nào "tốt nhất" — keypoint thắng ở twin-copy-move, aux cues
  thắng ở "có gì đó lạ" trên model-edit; fusion rule tận dụng điều đó.
- Khoảng trống lớn nhất: **localize vùng generative-edited** — holdout σ-separation MWU p per sample (N=6, n=2/kind, seed=7777, config_hash=`df80dcb9a6d32070`, bench JSON SHA-256 `33c56479c9743eb020a35144ae89a139c50aa802d942dc36ce62b40707354326`): erase_0000=5.870311664222097e-25, erase_0004=4.51494454293609e-28, move_0001=2.589512653945499e-10, move_0005=6.254540685818551e-08; vẫn cần detector cục bộ hơn →
  [`docs/ml-extension.md`](../ml-extension.md).
