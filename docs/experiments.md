# Experiment Protocol (ghi trước — chống HARKing)

## Factorial design

method {keypoint (SIFT/ORB chọn qua config thresholds), block (DCT)}
× transform {translate, rotate, scale} (synthforge affine 2D; distortion riêng qua battery `robustness.py`)
× postproc {none, jpeg80, jpeg60, blur, noise} (synthforge; jpeg90 dùng trong modelforge và robustness)
× tier {easy, medium, hard}

Output: per-condition tables (method × transform × postproc) + N +
n_excluded. CẤM một con số headline gộp.

## Hypotheses (ghi trước khi chạy)

- H1: keypoint > block trên rotate/scale (SIFT bất biến scale/rotate nhưng
  không bất biến affine đầy đủ; DCT không).
- H2: block ≥ keypoint trên vùng ít góc nhưng có texture đều.
- H3: JPEG postproc làm suy giảm cả hai, block chịu tệ hơn (lượng tử trôi).
- H4: ELA không tương quan mask GT đủ mạnh để localize — chỉ exploratory.

## Protocol

1. `python -m imageguard.evaluation.synthforge --n 6 --seed 42 --out data/synth_eval --tier medium`
   (hoặc gọi `generate_set(n, seed=42)`) → `data/synth_eval/` (regenerate được, manifest ghi seed).
2. `python -m imageguard eval --dataset synth --root data/synth_eval --split dev --methods keypoint,block --seed 42`.
3. Ghi `experiment_id` vào report/notes; calibration chỉ trên dev.
4. Test split chỉ chạy MỘT lần sau khi chốt thresholds.

## Số đo hiện có

- Copy-move dev: `data/synth_eval`, seed 42, 6 scored rows (3 ground-truth manipulated / 3 authentic controls):
  keypoint F1 0.50, recall 1/3, macro pixel IoU 0.258; block recall 0/3.
  Đây là bộ medium nhỏ, không phải benchmark công bố. Kiểm lại 2026-11-12
  bằng đúng hai lệnh trên: 6 scored rows, `dataset_version`
  `3289e7504d69`, `config_hash` `df80dcb9a6d32070`.
- LaMa edits and clean controls: xem [modelforge.md](modelforge.md) để đọc kết quả, split sizes và giới hạn.
- Classical enhancement, evaluation only: xem [enhancement-study.md](enhancement-study.md). PSNR/SSIM trên bộ synth đó không phải detection gain.
- Không chạy locked test split trên synthforge.
- CoMoFoD: bản local gitignored tại `data/comofod_base` (200 base sets;
  `provenance.json` ghi 199 strict-binary eligible, base `099` loại vì mask
  không nhị phân thuần; `_B.png` binary GT, `_M.png` mask mềm, không có `_C` trong
  bản này). Dataset không có trong repo nên các số 200/199 chưa kiểm lại được
  từ checkout này; script study loại base `099` với lý do
  `099_mask_convention_unresolved`. Dev qualification trên CoMoFoD không thay
  thế sự kiện CASIA formal
  (xem [`algorithms/comparison.md`](algorithms/comparison.md)).
