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

Đúng lý thuyết: SIFT bất biến rotation/scale; texture JPEG-robust giữ
detection qua re-encode. Phần còn lại INCONCLUSIVE (không phải clean)
— pipeline conservative by design.

## Model-edited (modelforge LaMa, n=12, 384² train sau sửa ground-truth) — ma trận đầy đủ tại [modelforge.md](../modelforge.md)

| kind | keypoint | block | ý nghĩa |
|---|---|---|---|
| splice_keep | **1.0**, IoU 0.006 (bF1 0.139) | 0.0 | twin sống qua Poisson blend |
| move | 0.0 (3 inc), IoU 0.000 | 0.0 | gốc bị LaMa xoá → twin phá (holdout 512²: 2 sus, IoU 0.042) |
| erase / foreign | 0.0 | 0.0 | không twin → đúng luật miss |

## Đặc tính từng method

| method | mạnh | yếu | cost (đo 512², CPU) |
|---|---|---|---|
| keypoint (SIFT) | twin blended/rotated/scaled; mask + pairs | vùng phẳng/texture kém → INCONCLUSIVE | ~0.1–0.5s |
| block (DCT) | không cần feature point; baseline kinh điển | miss clone đã blend; chậm; flat-guard bắt buộc | ~5–10s |
| ELA | JPEG history cue | chỉ JPEG; re-save phá; không standalone | ~0.1s |
| noise σ-MAD | train model-edit σ-separation MWU p per sample (N=12, n=3/kind, seed=2026, config_hash=`df80dcb9a6d32070`): erase_0000=8.803304263477716e-13, erase_0004=0.4566456368698131, erase_0008=0.00635903098503261; move_0001=1.728408115408002e-05, move_0005=1.5301133820660006e-06, move_0009=0.0016727292286712369 | localize yếu (IoU≈0.005) | ~0.02s |
| noise_local (two-tier) | localize model-fill: IoU 0.335–0.665, bF1 0.559–0.888 | false-positive vùng phẳng tự nhiên; vẫn aux | ~0.05s |
| freq cues | spectrum/blockiness artifact | mô tả hơn phán đoán | ~0.02s |
| splicing cues | cross-cue noise×ELA | suy giảm trên non-JPEG | ~0.05s |

## Takeaway cho báo cáo

- Không method nào "tốt nhất" — keypoint thắng ở twin-copy-move, aux cues
  thắng ở "có gì đó lạ" trên model-edit; fusion rule tận dụng điều đó.
- Khoảng trống lớn nhất: **localize vùng generative-edited** — holdout σ-separation MWU p per sample (N=6, n=2/kind, seed=7777, config_hash=`df80dcb9a6d32070`): erase_0000=5.870311664222097e-25, erase_0004=4.51494454293609e-28, move_0001=5.114573614120509e-10, move_0005=9.85708489286173e-07; vẫn cần detector cục bộ hơn →
  [`docs/ml-extension.md`](../ml-extension.md).
