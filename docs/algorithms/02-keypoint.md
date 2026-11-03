# 02 — Copy-Move Detection bằng Keypoint (SIFT/ORB)

## Bản chất

Copy-move: một vùng ảnh được copy và dán chỗ khác **trong cùng ảnh**. Vùng đích
mang texture giống hệt vùng nguồn → các điểm đặc trưng (keypoint) ở hai vùng có
descriptor gần như trùng. Detector = tìm cụm keypoint match + chứng minh hình học.

## Pipeline

```
gray(≤2048px) → detect kps+desc (SIFT/ORB)
  → self-match: knn k=3 + Lowe ratio + spatial filter
  → geometric verify: global affine RANSAC → displacement clustering
  → mask builder (Gaussian blob σ∝kp.size → threshold → morphology)
  → verdict (t_susp / t_clean / k_min)
```

## Toán trọng tâm

### SIFT

- Scale-space: `L(x,y,σ) = G(x,y,σ) * I(x,y)` — ảnh blur Gaussian ở nhiều σ.
- DoG: `D(x,y,σ) = L(x,y,kσ) − L(x,y,σ)` — cực trị DoG = keypoint (xấp xỉ
  LoG, rẻ hơn tính trực tiếp).
- Descriptor 128-d: vùng 16×16 quanh keypoint → 16 ô 4×4 → mỗi ô histogram
  8 bin hướng gradient → chuẩn hoá → bất biến scale/rotation/illumination.

### Lowe ratio test (Lowe 2004)

Với k=3 self-match `[self, d1, d2]` (bỏ self d≈0):

```
match hợp lệ ⟺ ‖d1‖ / ‖d2‖ < ρ   (ρ=0.8)
```

Descriptor đặc trưng thì nearest-neighbour (twin copy) gần hơn hẳn runner-up.
Texture lặp → d2 gần d1 → ratio fail → tự loại nhiễu.

### RANSAC affine

Model affine 2×3 (6 dof, cần ≥3 điểm không thẳng hàng): `x' = Ax + t`.
Số vòng lặp lý thuyết: `N = log(1−p) / log(1−w^n)` với p=0.99 xác suất thành
công, w = tỉ lệ inlier, n=3 điểm tối thiểu. Inlier: residual < ε=3px.

## Chiến lược 2 pass (đánh đổi)

1. **Global affine RANSAC** trên toàn bộ pairs — bắt một vùng copy với
   transform affine bất kỳ (translate/rotate/scale đều là affine).
2. **Displacement clustering** trên outliers — bắt NHIỀU vùng copy độc lập
   kiểu translate (mỗi vùng cùng một (dx,dy)); mỗi cluster → RANSAC riêng.

Giới hạn: nhiều vùng copy với affine *khác nhau triệt để* chỉ bắt được một
phần — ghi `limitations`.

## Kết quả & thất bại điển hình

| Tình huống | Kết quả | Lý do |
|---|---|---|
| Ảnh phẳng/mờ | INCONCLUSIVE | kps < k_min — insufficient texture, đúng hành vi |
| Texture lặp tự nhiên (gạch, cửa sổ) | thường INCONCLUSIVE | coverage > 50% → hạ verdict |
| Copy-move + JPEG nặng | suy giảm | descriptor trôi sau re-encode; blob texture sống tốt hơn nhiễu 1px |
| Copy vùng rất nhỏ | INCONCLUSIVE | ít keypoint trong vùng → ít inlier |
| Rotate/scale lớn | tùy mức | affine RANSAC chịu được; biến dạng phiếm diện thì không |

`score` = số cặp verified (inliers), KHÔNG phải xác suất. Mask = union cả hai
phía — keypoint không chứng minh được hướng copy.
