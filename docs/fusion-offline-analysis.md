# Phân tích fusion offline (khám phá)

Tài liệu này mô tả **kết quả đếm verdict** khi áp dụng luật `fusion.combine` lên các hàng detector **đã lưu** trong sự kiện CASIA formal (103 cặp gốc/chỉnh), **không** chạy lại detector và **không** chỉnh ngưỡng. Mục đích: minh bạch hóa output fusion trên cùng artifact đã dùng cho báo cáo tuning-free — **không** phải held-out mới và **không** bao gồm 34 cặp held-aside.

## Lệnh tái lập

Chạy từ root repo (placeholder đường dẫn; file thật nằm ngoài git trong thư mục kế hoạch đánh giá):

```powershell
$env:PYTHONPATH='<worktree>\src'
& '<python>' scripts/fusion_offline_summary.py --report <path-to-report.json> --format markdown
```

Ví dụ `--format both` in JSON + bảng markdown.

## Kết quả trên `report.json` sự kiện formal CASIA (103 cặp)

Nguồn số: output của `scripts/fusion_offline_summary.py` với `--report` trỏ tới `report.json` baseline sự kiện CASIA formal b-v1 (206 ảnh: 103 gốc + 103 chỉnh; mỗi ảnh có hàng keypoint và block).

### Keypoint đơn lẻ

| Cohort | n | suspicious | inconclusive | no_indication |
| --- | ---: | --- | --- | --- |
| originals | 103 | 22 (0.145–0.302) | 59 (0.476–0.664) | 22 (0.145–0.302) |
| edits | 103 | 94 (0.842–0.953) | 9 (0.047–0.158) | 0 (0.000–0.036) |

### Block đơn lẻ

| Cohort | n | suspicious | inconclusive | no_indication |
| --- | ---: | --- | --- | --- |
| originals | 103 | 0 (0.000–0.036) | 0 (0.000–0.036) | 103 (0.964–1.000) |
| edits | 103 | 37 (0.273–0.455) | 3 (0.010–0.082) | 63 (0.515–0.700) |

### Fused (keypoint + block) — cùng luật `combine` trong app

| Cohort | n | suspicious | inconclusive | no_indication |
| --- | ---: | --- | --- | --- |
| originals | 103 | 22 (0.145–0.302) | 59 (0.476–0.664) | 22 (0.145–0.302) |
| edits | 103 | 95 (0.854–0.960) | 8 (0.040–0.146) | 0 (0.000–0.036) |

Số trong ngoặc: khoảng Wilson 95% cho **tỉ lệ** verdict đó trong cohort (mỗi ảnh đếm một lần).

Trên cohort này, fusion hai method detector khiến **một** ảnh chỉnh thêm vào nhóm suspicious so với keypoint đơn lẻ (94 → 95); originals không đổi so với keypoint đơn lẻ vì block không đẩy suspicious trên gốc trong artifact này.

## Giới hạn và caveat

- Artifact chỉ chứa `copy_move_keypoint` và `copy_move_block`. Các method phụ (ELA, EXIF, noise, frequency, splicing) **không** có trong `report.json`; danh sách method mặc định trên GUI có thể **abstain nhiều hơn** (thêm `INCONCLUSIVE`) so với bảng fused ở trên.
- Đây là phân tích **khám phá** trên cùng 103 cặp đã dùng cho báo cáo chất lượng detector; **không** thay thế test giữ lại (34 cặp) và **không** mở rộng sang ảnh ngoài cohort CASIA đã sàng lọc.
- Verdict là **dấu hiệu nghi vấn / không dấu hiệu / không kết luận**, không phải nhãn “giả” hay “thật” pháp y.

## Examiner không nên suy diễn

- **Không** coi tỉ lệ suspicious trên ảnh gốc là “false positive rate” thực tế hay độ chính xác trên ảnh tùy ý.
- **Không** coi tỉ lệ suspicious trên ảnh chỉnh là sensitivity toàn cục hay tỷ lệ “bắt được giả mạo”.
- Khoảng Wilson chỉ mô tả **độ không chắc của tỉ lệ mẫu 103** trong cohort đã định nghĩa — không ngoại suy sang dân số ảnh khác.
- So sánh fused vs từng method ở đây **không** chứng minh fusion “tốt hơn” ngoài cohort và protocol đã khóa.

Tham chiếu bối cảnh: [`docs/limitations.md`](limitations.md), [`docs/evaluation-methodology.md`](evaluation-methodology.md).
