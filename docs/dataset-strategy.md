# Dataset Strategy

## synthforge (in-repo)

- Procedural, PCG64-seeded, deterministic — unit/integration tests từ ngày 1.
- Mỗi mẫu: `*_O.png` (original), `*_F.png` (forged), `*_M.png` (binary mask,
  255=vùng dán), `meta.json` {seed, src_polygon, dst_offset, transform, postproc}.
- Tiers: easy (translate ± jpeg80), medium (+rotate, postproc đa dạng),
  hard (+scale, vùng nhỏ, postproc nặng).
- KHÔNG phải benchmark công bố — chỉ dev/regression nội bộ.

## CoMoFoD (external)

- Tralic et al., ELMAR-2013 — copy-move dataset có GT mask.
- **Manual download** (điều khoản phát hành): đặt vào `data/comofod/`
  (gitignored, không redistribute trong repo).
- Naming: `NNN_F.png` forged, `NNN_O.png` original, `NNN_B.png` binary mask,
  `NNN_C.png` color mask (một số bản `_M`).
- **Verify gate bắt buộc** trước batch eval đầu:
  1. `python -m imageguard eval --dataset comofod --root data/comofod` chỉ
     chạy sau khi `CoMoFoDAdapter.verify_masks()` (trong `src/imageguard/evaluation/comofod.py`)
     xác nhận: dims mask==image, giá trị mask đúng convention
     (`binary_white_is_forged`), manifest đếm đúng file thực.
  2. Mask convention KHÔNG được đoán — đọc readme dataset / mở bằng mắt.
- Adapter: discovery-based, thiếu dataset → skip+log, không fail suite.

## Split & leakage

- Split dev/test theo `base_image_id` (sha256 % 2) — deterministic, mọi biến
  thể của một ảnh gốc nằm cùng phía. Dev = hiệu chuẩn ngưỡng; test = khóa.
- `fp_area_ratio` đo false-positive trên authentic; `n_clean_correct` đếm
  authentic không bị tô mask.

## Exclusion rules

- Forged file mà GT mask rỗng → data anomaly → exclude + `n_excluded` + log.
- Mask dims ≠ image dims → reject + log, KHÔNG silent resize.
