# Experiment Protocol (ghi trước — chống HARKing)

## Factorial design

method {keypoint-SIFT, keypoint-ORB, block-DCT}
× transform {translate, rotate, scale, distortion}
× postproc {none, JPEG q60/80/90, blur, noise}
× tier {easy, medium, hard}

Output: per-condition tables (method × transform × postproc) + N +
n_excluded. CẤM một con số headline gộp.

## Hypotheses (ghi trước khi chạy)

- H1: keypoint > block trên rotate/scale (SIFT bất biến affine, DCT không).
- H2: block ≥ keypoint trên vùng ít góc nhưng có texture đều.
- H3: JPEG postproc làm suy giảm cả hai, block chịu tệ hơn (lượng tử trôi).
- H4: ELA không tương quan mask GT đủ mạnh để localize — chỉ exploratory.

## Protocol

1. `generate_set(n, seed=42)` → `data/synth_eval/` (regenerate được, manifest
   ghi seed).
2. `python -m imageguard eval --dataset synth --root data/synth_eval
   --split dev --methods keypoint,block --seed 42`.
3. Ghi `experiment_id` vào report/notes; calibration chỉ trên dev.
4. Test split chỉ chạy MỘT lần sau khi chốt thresholds.

## Số đo hiện có (đo thật, không phải claim)

- `synthforge-dev-*`: keypoint trên 6-sample dev, tier easy — xem
  `reports/eval-*.json` (gitignored, regenerate được).
