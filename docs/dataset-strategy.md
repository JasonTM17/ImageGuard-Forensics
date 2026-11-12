# Dataset Strategy

## synthforge (in-repo)

- Procedural, PCG64-seeded, deterministic — unit/integration tests từ ngày 1.
- Mỗi mẫu: `*_O.png` (original), `*_F.png` (forged), `*_M.png` (binary mask,
  255=vùng dán), `meta.json` {seed, shape, src_polygon, dst_offset, rotate_deg,
  scale, postproc, tier}.
- Tiers (`synthforge._rand_spec`): easy — không rotate/scale, postproc
  `none`|`jpeg80`; medium — rotate ±25°, cùng tập postproc; hard — thêm scale
  0.85–1.15. Vùng dán chiếm 15–30% mỗi cạnh ở cả ba tier.
- KHÔNG phải benchmark công bố — chỉ dev/regression nội bộ.

## CoMoFoD (external)

- Tralic et al., ELMAR-2013 — copy-move dataset có GT mask.
- **Manual download** (điều khoản phát hành): giải nén vào `data/comofod_base/`
  (gitignored, không redistribute trong repo). Bản hiện có: 200 base sets;
  `provenance.json` tại root — `strict_binary_eligible_sets`=199 (base `099`
  excluded: mask có giá trị trung gian, chưa dùng cho strict-binary eval).
  `data/` không nằm trong repo nên hai số 200/199 chưa kiểm lại được từ
  checkout này; `scripts/evaluate_detector_quality.py` luôn loại base `099`
  với lý do `099_mask_convention_unresolved`.
- Naming trong `comofod_base`: `NNN_F.png` forged, `NNN_O.png` original,
  `NNN_B.png` binary mask, `NNN_M.png` soft mask (không có `_C` trong bản đã
  kiểm); adapter vẫn nhận `_C` nếu có ở bản khác.
- **Verify gate bắt buộc** trước batch eval đầu:
  1. `python -m imageguard eval --dataset comofod --root data/comofod_base` chỉ
     chạy sau khi `CoMoFoDAdapter.verify_masks()` (trong `src/imageguard/evaluation/comofod.py`)
     đã chạy: hàm đếm số cặp đã kiểm trên split `dev`, liệt kê các mask lệch
     kích thước, và trả vài giá trị mask quan sát được cho từng ảnh để người
     đọc đối chiếu. Hàm không tự khẳng định convention và không đối chiếu
     manifest với số file trên đĩa; adapter dựng manifest bằng cách quét chính
     các file có trong thư mục (không hardcode số lượng).
  2. Mask convention KHÔNG được đoán — đọc readme dataset / mở bằng mắt.
- Adapter: discovery-based, thiếu dataset → skip+log, không fail suite.

## Split & leakage

- Split dev/test theo `base_image_id` (sha256 % 2) — deterministic, mọi biến
  thể của một ảnh gốc nằm cùng phía. Dev = hiệu chuẩn ngưỡng; test = khóa.
- `fp_area_ratio_mean` (trung bình tỉ lệ diện tích mask dự đoán trên ảnh
  authentic) đo false-positive; `n_clean_correct` đếm authentic không bị tô
  mask.

## Exclusion rules

- Forged file mà GT mask rỗng → data anomaly → exclude + `n_excluded` + log.
- Mask dims ≠ image dims → reject + log, KHÔNG silent resize.
