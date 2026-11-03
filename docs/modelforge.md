# ModelForge + ModelBench — ảnh chỉnh sửa bằng model local & benchmark

## Câu hỏi nghiên cứu

> Khi ảnh bị chỉnh sửa bằng **model sinh nội dung mới** (inpainting deep
> learning — LaMa) thay vì copy-paste pixel thô, ImageGuard còn phát hiện
> và localize được không?

Đây là câu hỏi quan trọng với capstone vì copy-move detector truyền thống
được thiết kế để tìm "anh em sinh đôi" (twin) — vùng bị model XOÁ và SINH
MỚI không còn twin. Kết quả định lượng phía dưới là đo thật, không suy diễn.

## Thành phần

| File | Vai trò |
|---|---|
| `evaluation/modelforge.py` | Sinh dataset edit bằng model: 4 kind × 4 shape × PNG/JPEG |
| `evaluation/modelbench.py` | Benchmark detector trên set đó: IoU, boundary-F1@8, σ-separation MWU |
| `models/lama_fp32.onnx` | LaMa (208MB, **không commit**) — tải: `huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx`, SHA-256 `1faef530…b68d6` |
| `reports/modelbench_*.md` | Report xuất tự động (gitignore) |

## Backend editor

- **`lama`** — LaMa ONNX qua onnxruntime CPU (~30s/inpaint ở 512²). Fourier
  convolutions nhìn toàn cảnh → vùng fill hợp ngữ cảnh, seam mềm.
- **`classic`** — `cv2.inpaint` Telea + `seamlessClone`: baseline nhanh,
  seam dễ thấy hơn — dùng cho test nhanh và so sánh.

## Các kiểu edit (`kind`)

| kind | Mô tả | Twin tồn tại? | Detector lý thuyết |
|---|---|---|---|
| `erase` | Inpaint xoá object | Không | keypoint/block: thiết kế không bắt |
| `move` | seamlessClone đến đích + inpaint xoá gốc | Không (gốc đã xoá) | keypoint thất bại đúng luật |
| `splice_keep` | seamlessClone, giữ nguyên bản gốc | Có (đã blend mép) | keypoint nên bắt được |
| `splice_foreign` | seamlessClone vùng ảnh khác vào | Không | splicing cues mới đúng miền |

Mỗi mẫu có shape ngẫu nhiên `rect|ellipse|polygon|blob` và một nửa lưu
JPEG q90 (re-save thực tế + cho ELA có tín hiệu).

## Kết quả đo thật — `modelforge_lama12` (n=12, seed 2026, 384²)

Detection matrix (det-rate = tỉ lệ SUSPICIOUS_INDICATION):

| kind | keypoint | block | noise | freq | splicing | ELA |
|---|---|---|---|---|---|---|
| erase | 0.0 (IoU≈0) | 0.0 | 3× INCONC | 1× INCONC | 0.0 | 1× INCONC |
| move | **0.333** | 0.0 | 3× INCONC | 2× INCONC | 0.0 | — |
| splice_keep | **1.0** (IoU 0.265, bF1 0.515) | 0.0 | 3× INCONC | 0.0 | 0.0 | 3× INCONC |
| splice_foreign | 0.0 | 0.0 | 3× INCONC | 1× INCONC | 0.0 | 1× INCONC |

### Phát hiện quan trọng nhất — Noise-σ separation

| kind | σ_in (median) | σ_out | MWU p-value | Ý nghĩa |
|---|---|---|---|---|
| erase (3 mẫu) | 2.3–4.3 | ~10 | **< 0.0002 cả 3** | Vùng LaMa-fill MỊN hơn nền rõ có ý nghĩa |
| move (3 mẫu) | 5.0–7.2 | ~11 | **< 0.0001 cả 3** | Vùng xoá-gốc cũng để lại dấu vết |
| splice_keep | 8.7–9.5 | ~11 | 0.02–0.13 | Clone giữ noise riêng → khó tách hơn |
| splice_foreign | 9–18 | ~11 | 0.03–0.79 | Tuỳ noise của nguồn ngoài |

**Diễn giải:** model inpainting sinh content ít nhiễu hơn ảnh thật →
residual-σ trong vùng chỉnh sửa thấp hơn có ý nghĩa thống kê (p<0.001).
Cue này **tồn tại thật và đo được** — vấn đề là detector phù hợp.

### Vòng cải tiến detector (đo trên cùng set, ba phiên bản)

| detector | ý tưởng | erase IoU | move IoU | kết luận |
|---|---|---|---|---|
| `noise_inconsistency` (MAD toàn ảnh) | outlier 2 phía vs median global | 0.000 | 0.006 | median≈nền, vùng nhỏ không lệch đủ |
| local-MAD v1 | median láng giềng 5×5 block | 0.000 | 0.000 | láng giềng hỗn hợp → MAD phình → z chìm; chỉ bắt seam-ring mép (n_low≈0) |
| Otsu v2 | tự tách 2 mode σ | 0.004–0.105 | 0.085–0.191 | vùng edit ~1% lưới → Otsu cắt giữa mode nền |
| **`noise_local` v3** | σ < 0.6·median + component ≥6 block | **0.169** | **0.215** | ✓ tỉ lệ cứng một phía đúng vật lý model-fill |

`noise_local` cũng đạt bF1@8: erase 0.332, move 0.449 — mask chạm đúng
vùng chỉnh sửa (không phải nhiễu ngẫu nhiên). Vẫn giữ verdict tối đa
INCONCLUSIVE theo aux contract.

### Kiểm chứng mức ngẫu nhiên (chance-level) và false-positive

| sample | flag-rate | recall GT | IoU | chance IoU |
|---|---|---|---|---|
| erase_0000 | 11.8% | 78.9% | 0.299 | ~0.036 |
| move_0001 | 17.2% | 51.4% | 0.191 | ~0.057 |
| erase_0004 | 9.2% | 86.4% | 0.127 | ~0.012 |
| move_0005 | 12.5% | 44.4% | 0.187 | ~0.046 |
| erase_0008 | 15.5% | 61.5% | 0.082 | ~0.019 |
| move_0009 | 14.1% | 61.3% | 0.267 | ~0.051 |

IoU quan sát **cao hơn mức ngẫu nhiên 4–8×** → localization là tín hiệu
thật, không phải may mắn do mask rộng. Tuy nhiên:

- **False-positive control:** trên 12 ảnh GỐC chưa chỉnh, `noise_local`
  vẫn flag 10–18% diện tích (vùng trời/phẳng/mịn tự nhiên có σ thấp).
  → precision bị giới hạn cấu trúc; detector chỉ nói "vùng này mịn bất
  thường" chứ không phân biệt được mịn-do-model vs mịn-tự-nhiên.
- Đây là lý do verdict tối đa INCONCLUSIVE là **bắt buộc**: cue thật
  nhưng không đặc hiệu.

### Held-out validation (seed 7777, `modelforge_holdout/`)

Ngưỡng q=0.6 được chọn từ bộ seed 2026 — kiểm chứng trên bộ seed khác
hoàn toàn (`reports/modelbench_holdout.md`, n=6):

| kind | train IoU | holdout IoU | holdout bF1@8 |
|---|---|---|---|
| erase | 0.169 | **0.162** | 0.311 |
| move | 0.215 | **0.172** | 0.378 |

σ-separation tái lập trên holdout: σ_in 2.9–7.5 vs σ_out ~10.5–11,
p<0.001 cả 4 mẫu erase/move → **q=0.6 không overfit**, cue ổn định
theo cơ chế model-fill (không phụ thuộc seed).
- Hướng cải thiện thực tế: kết hợp seam-ring evidence (đã giữ riêng
  trong `evidence["seam_ring_blocks"]`), hoặc PRNU-fingerprint mismatch —
  natural flat có noise pattern nhất quán camera, model-fill thì không.

### Kết luận trả lời câu hỏi ban đầu

1. **Clone giữ gốc (kể cả blend Poisson mềm)** → keypoint SIFT vẫn bắt
   twin (3/3) và localize tốt (bF1 0.515).
2. **Edit bằng model sinh mới (erase/move/foreign)** → conclusive detectors
   không kết luận được (đúng thiết kế), NHƯNG hệ thống không "mù": noise
   σ-separation có ý nghĩa thống kê mạnh → verdict INCONCLUSIVE +
   corroboration là câu trả lời đúng của hệ thống, và đường nâng cấp rõ ràng.
3. `move` 0.333: một mẫu twin sót lại sau inpaint — minh hoạ LaMa không
   phải lúc nào cũng xoá sạch.

## Reproduce

```bash
pip install -e ".[lama]"          # onnxruntime
# tải models/lama_fp32.onnx theo link + sha256 ở bảng trên
python -m imageguard.evaluation.modelforge --n 12 --seed 2026 \
    --out modelforge_lama12 --backend lama --size 384
python -m imageguard.evaluation.modelbench --set modelforge_lama12 \
    --out reports/modelbench_lama12
```

## Attribution

LaMa: Suvorov et al., WACV 2022, arXiv:2109.07161, Apache-2.0 —
github.com/advimman/lama. ONNX export: Carve/LaMa-ONNX (Apache-2.0),
train trên Places2 (CC-BY 4.0).
