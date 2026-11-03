# Method Comparison — số đo thật từ benchmark nội bộ

Mọi số trong bảng đo từ `evaluation/robustness.py` + `modelbench.py`
trên dataset sinh nội bộ (synthforge + modelforge). KHÔNG phải benchmark
chuẩn quốc tế — xem `docs/evaluation-methodology.md` cho protocol.

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

## Model-edited (modelforge LaMa, n=12, 384²) — ma trận đầy đủ tại docs/modelforge.md

| kind | keypoint | block | ý nghĩa |
|---|---|---|---|
| splice_keep | **1.0**, IoU 0.265 | 0.0 | twin sống qua Poisson blend |
| move | 0.333 | 0.0 | gốc bị LaMa xoá → twin phá |
| erase / foreign | 0.0 | 0.0 | không twin → đúng luật miss |

## Đặc tính từng method

| method | mạnh | yếu | cost (đo 512², CPU) |
|---|---|---|---|
| keypoint (SIFT) | twin blended/rotated/scaled; mask + pairs | vùng phẳng/texture kém → INCONCLUSIVE | ~0.1–0.5s |
| block (DCT) | không cần feature point; baseline kinh điển | miss clone đã blend; chậm; flat-guard bắt buộc | ~5–10s |
| ELA | JPEG history cue | chỉ JPEG; re-save phá; không standalone | ~0.1s |
| noise σ-MAD | model-edit σ-separation p<0.001 (erase/move) | localize yếu (IoU≈0.005) | ~0.02s |
| noise_local v3 | localize model-fill: IoU 0.17–0.22, bF1 0.33–0.45 | false-positive vùng phẳng tự nhiên; vẫn aux | ~0.05s |
| freq cues | spectrum/blockiness artifact | mô tả hơn phán đoán | ~0.02s |
| splicing cues | cross-cue noise×ELA | suy giảm trên non-JPEG | ~0.05s |

## Takeaway cho báo cáo

- Không method nào "tốt nhất" — keypoint thắng ở twin-copy-move, aux cues
  thắng ở "có gì đó lạ" trên model-edit; fusion rule tận dụng điều đó.
- Khoảng trống lớn nhất: **localize vùng generative-edited** — cue
  σ-separation tồn tại (p<0.001) nhưng cần detector cục bộ hơn →
  `docs/ml-extension.md`.
