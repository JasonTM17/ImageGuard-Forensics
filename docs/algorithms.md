# Algorithms — ImageGuard Forensics

Index tài liệu thuật toán + công thức nền. Chi tiết từng method xem file riêng.

| Method | File | Trạng thái |
|---|---|---|
| Copy-move keypoint (SIFT/ORB) | [02-keypoint.md](algorithms/02-keypoint.md) | P02 ✓ |
| Copy-move block (DCT) | [03-block.md](algorithms/03-block.md) | P03 ✓ |
| ELA | file này, mục dưới | P04 ✓ |
| EXIF audit | file này, mục dưới | P04 ✓ |
| Noise inconsistency | file này, mục dưới | P08 ✓ |
| Local noise / model-fill | file này, mục dưới | P08 ✓ |
| Frequency analysis | file này, mục dưới | P08 ✓ |
| Splicing cues | file này, mục dưới | P08 ✓ |

## ELA — Error Level Analysis

`ELA(x,y) = |I(x,y) − R_q(I)(x,y)|` với `R_q` = JPEG re-encode ở quality q.

- Vùng nén lần đầu: error phân bố theo JPEG quantization đầu.
- Vùng chèn sau (đã qua lịch sử nén khác / chưa nén): error khác biệt.
- Sweep q∈{75,85,90,95} → heat = max qua các map, amplify ×15.

**Vì sao chỉ exploratory:** re-save xoá vệt; cạnh sắc/texture tự nhiên cũng
cho ELA cao; flat region ELA thấp tự nhiên. ELA không bao giờ trả
SUSPICIOUS — tối đa INCONCLUSIVE.

## EXIF audit

Checklist: Software (editor hints), DateTime vs DateTimeOriginal, thiếu
Make/Model, tag_count=0 → `findings[]`. Metadata là dữ kiện khai báo —
có finding thì `INCONCLUSIVE`, không có thì `NO_INDICATION`; không bao giờ
`SUSPICIOUS`, không score.

## Phương pháp bổ trợ L2 (P08)

### Noise inconsistency
- Residual `r = I − median_k(I)` (k=5) — giữ high-frequency nhiễu.
- Ước lượng σ nhiễu theo block: `σ_b = std(r)` trong block; outlier khi
  `|σ_b − median(σ)| > 3·MAD(σ)` với `MAD = median|σ − median(σ)|`.
- Vùng chèn từ ảnh khác → σ nội bộ lệch median±m·MAD toàn ảnh (m=3.0)
  → outlier map.
- Điểm yếu đã biết: denoising/sharpening hợp pháp cũng tạo σ bất thường; aux signal tối đa INCONCLUSIVE.

### Local noise / model-fill (`noise_local`)
- Phát hiện vùng residual có σ thấp bất thường (model-fill mịn hơn nền).
- Bộ lọc 2 tầng: (1) `σ < q·median` (q=0.6) và (2) component-mean `σ < q2·median` (q2=0.45) với `min_region_blocks=6`.
- Không bao giờ trả SUSPICIOUS — chỉ là aux cue (tối đa INCONCLUSIVE).

### Frequency analysis
- FFT2 magnitude spectrum (log-scale): JPEG grid artifacts → peaks định kỳ
  ở tần số 1/8 chu kỳ/px (lưới 8×8); resampling → periodic peaks; clone →
  không dấu vết trực tiếp nhưng vùng dán đôi khi đổi spectrum cục bộ.
- Blockiness metric: chênh lệch gradient biên 8×8 trung bình so với nội block.

### Splicing cues
- Corroboration giữa noise outlier và ELA difference: tỉ lệ
  `(noise_outlier ∩ ELA_hot)/ELA_hot ≥ overlap_frac` (0.15) → `INCONCLUSIVE`
  (cue kết hợp); mask là phần giao hai vùng.
