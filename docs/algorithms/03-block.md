# 03 — Copy-Move Detection bằng Block Matching (Fridrich/Popescu)

## Bản chất

Chia ảnh thành block overlap B×B (stride 1). Vùng copy → hai tập block có
nội dung giống nhau → feature vector giống nhau → match. Khác keypoint:
**dense** (mọi vị trí), phát hiện được cả vùng ít góc — nhưng chỉ translate.

## Pipeline

```
gray(≤768px) → mọi block B×B → DCT + lượng tử → feature K hệ số zigzag
  → loại flat blocks (std < ε) → lexicographic sort → k hàng xóm match
  → shift-vector histogram → giữ shift ≥ min_votes → rasterize mask
```

## Toán trọng tâm

- **DCT-II**: `F(u,v) = α(u)α(v) ΣΣ f(x,y)·cos[(2x+1)uπ/2B]·cos[(2y+1)vπ/2B]`
  với `α(0)=√(1/B)`, `α(u)=√(2/B)` — ma trận hoá `F = D·block·Dᵀ`.
- **Lượng tử hoá**: `F_q = round(F/Q)`, Q tăng theo `u+v` — hai block giống
  nhau sau JPEG/nhiễu nhẹ vẫn rơi cùng bin (robustness qua mất thông tin có
  chủ đích).
- **Lexicographic sort**: `O(N_b log N_b)` thay vì `O(N_b²)` — block giống
  nhau đứng cạnh nhau sau sort, chỉ duyệt k hàng xóm.
- **Shift voting**: cặp thật chia sẻ cùng (dx,dy); cặp nhiễu shift rải rác
  → histogram vote là bộ lọc hình học thô, miễn phí.

## Flat-block filter (quan trọng)

Vùng phẳng (bầu trời, tường trơn) → mọi block gần-identical → "match" loạn
khắp ảnh → false positive kinh điển của trường phái. Loại block `std < ε`
(ε=5) khỏi matching — chúng không mang texture nhận diện.

## Giới hạn

| Tình huống | Kết quả | Lý do |
|---|---|---|
| Copy + rotate/scale/flip | NO_INDICATION | hệ số DCT đổi hoàn toàn — không bất biến hình học |
| Copy + JPEG mạnh | suy giảm | lượng tử hoá giảm ảnh hưởng nhưng không miễn nhiễm |
| Ảnh lớn | chậm | `N_b ≈ (H−B+1)(W−B+1)` — 512² → 247k blocks (~10s đo trên máy dev) |
| Vùng copy < B | INCONCLUSIVE | vùng nhỏ không chứa trọn block |

## score

`matched_area_fraction` = diện tích mask / diện tích ảnh. KHÔNG phải xác suất.
