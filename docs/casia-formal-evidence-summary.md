# Tóm tắt bằng chứng sự kiện CASIA formal (đã redact)

Tài liệu này là **bản tổng hợp công khai** của một sự kiện đánh giá detector đã khóa trên cohort CASIA-CMFD đã sàng lọc. Các artifact JSON, mask PNG, báo cáo đầy đủ và receipt thực thi **không** nằm trong repo git; chúng được giữ ngoài kho công khai để tránh lộ dữ liệu nghiên cứu cục bộ. Mọi **con số** dưới đây được sao chép nguyên văn từ bản ghi sự kiện đã lưu cục bộ (2026-11-08, timezone Asia/Bangkok), không suy diễn thêm.

**Ranh giới:** Sự kiện đo *dấu hiệu suspicious* và localization cue mask so với **oracle mask tác giả** đã chiếu — **không** chứng minh ảnh camera tùy ý là giả hay nguyên bản. **Verdict fusion** trong app/UI **không** được đánh giá trong sự kiện hai-method này.

## Xây dựng cohort và giả định oracle

| Bước | Số liệu / quy tắc (từ bản ghi intake + contract) |
| --- | --- |
| Cohort cố định trước khi decode | **144** cặp original/edited |
| Cách ly (quarantine) | **7** cặp — lệch geometry hoặc alias RGB chính xác giữa original ID khác nhau (theo bản ghi intake) |
| Survivors sau sàng lọc | **137** cặp hoàn chỉnh, RGB gốc phân biệt theo bản ghi |
| Chọn formal | **103** cặp (rank 1–103 theo digest cố định `imageguard-casia-formal-103-v1` + seed **42** + original ID) |
| Held-aside | **34** cặp (rank 104–137) — **không** chạy inference detector trong sự kiện |
| Oracle đã duyệt | Chiếu mask tác giả `casia-busternet-blue-complement-v1`: đỏ/xanh lá = foreground, xanh dương = pristine; giữ geometry và hướng ảnh |
| **Chưa** được chứng minh trong intake | Đúng vùng annotate của tác giả; độc lập scene giữa các cặp (crop/viewpoint/rotate có thể còn liên quan); screening RGB/dHash **không** chứng minh độc lập viewpoint |

Người dùng đã chấp nhận tường minh population **103** cặp, oracle blue-complement union, giới hạn scene-screening đã nêu và tiêu chí criterion-B trước khi inference.

## Sự kiện và phạm vi đo

- Một sự kiện: `casia-formal-b-261108-v1`, seed **42**, thứ tự method: keypoint rồi block.
- Baseline và candidate chạy **tuần tự** trên cùng checkout sạch; **206** ảnh được chấm mỗi role (103 gốc + 103 chỉnh), **103** nhóm original-ID.
- **0** loại trừ, lỗi hoặc NOT_APPLICABLE trên các hàng đã chọn; **34** cặp held-aside không inference.
- Thời gian wall-clock (bản ghi thực thi): baseline **212.522** s, candidate **273.662** s, tổng **498.587** s; không timeout.

## Kết quả aggregate (baseline `.01` vs candidate `.007`, keypoint giống nhau ở cả hai role)

Trên **103** ảnh chỉnh và **103** ảnh gốc; confusion đếm INCONCLUSIVE như nonsuspicious nhưng vẫn báo riêng.

| Metric | Block baseline `.01` | Block candidate `.007` | Keypoint (cả hai role) |
| --- | ---: | ---: | ---: |
| Edited suspicious /103, TP | 37 | 38 | 94 |
| Edited nonsuspicious /103, FN | 66 | 65 | 9 |
| Original suspicious /103, FP | 0 | 0 | 22 |
| Original nonsuspicious /103, TN | 103 | 103 | 81 |
| Edited recall | 35.92% | 36.89% | 91.26% |
| Original FPR | 0/103 | 0/103 | 21.36% |
| Original INCONCLUSIVE | 0 | 0 | 59 |
| Edited INCONCLUSIVE | 3 | 2 | 9 |
| Macro IoU, 103 hàng chỉnh | .2891367884 | .2891367884 | .4607202509 |
| Micro IoU | .3017894382 | .3017894382 | .4378979955 |

Một verdict đổi giữa baseline và candidate: nhóm `Au_ind_00041`, ảnh chỉnh liên quan — INCONCLUSIVE → SUSPICIOUS; IoU giữ `.5820336868`.

## Cổng khoa học criterion-B — **FAIL**

Tiêu chí đã đăng ký (hash criteria trong contract): block candidate phải cải thiện **ít nhất +5** TP ảnh mức trên ảnh chỉnh so với baseline, đồng thời giữ các giới hạn FPR, IoU và abstention đã khóa.

- Quan sát: block candidate **+1** TP (37 → 38), **dưới** ngưỡng **+5**.
- Comparator báo `FAIL`; nhãn thất bại gồm `tp_delta_min` và `primary_correction_count`.
- Các giới hạn khác (FP, IoU, abstention gốc) vẫn trong biên đã duyệt — **không** đủ để chấp nhận candidate.
- **Quyết định cuối:** scientific gate **FAIL**; ngưỡng production mặc định **không** đổi theo sự kiện này.

## SHA-256 artifact và config (theo bản ghi sự kiện)

Giá trị dưới đây là danh tính file đã lưu cục bộ; nội dung file **không** redistribute trong repo.

| Vai trò | SHA-256 |
| --- | --- |
| Baseline `report.json` | `3a04da5f759f7e7d1be94fceb61f3e38d232772a8524b3003b33ce62a092e018` |
| Candidate `report.json` | `1381265fc96d37a6aebf084d3e23a69a3d5b0e0145d0d7f8873950e660706f74` |
| Comparison (receipt aggregate) | `cc5570e2b66d964fc7209753245b3221ef5ee0538d967a3b1fb91773d8a3064f` |
| Approved runtime criteria | `059fbdf54146fb6a45a905ed67b175826eab9b82afb1e925683b27402e1f1ebb` |
| Pair lock | `dd1b68643207f2f4cbf7e8b0f31eb53c563a370a863a9dd0a68c84de4ca8b75c` |
| Event freeze | `316e9567f1c0d29fe270e28b47fe2e58f01bbbe3db0b76c21d385d6764e1783c` |
| Manifest seal | `f56884c805cd64c9a0b118347366f91674158569343eb809369853c48e011a33` |
| Protocol | `dc5eae1c4568a6d54ba1a0cc19e90479a018dde8dd6d882bfceba9eb1f2bdb77` |

Runtime ghi nhận: Windows, Python **3.14.2**, OpenCV **5.0.0.93**, NumPy **2.5.4**, Pillow **13.0.0**, PyYAML **6.0.3**.

## Đối chiếu với tài liệu công khai

- Tóm tắt định tính cho người đọc repo: [`README.md`](../README.md), [`limitations.md`](limitations.md).
- Phân tích fusion khám phá trên cùng artifact baseline: [`fusion-offline-analysis.md`](fusion-offline-analysis.md).
- Bản nháp báo cáo: [`report/capstone-draft.md`](report/capstone-draft.md).

Qualification CoMoFoD dev **không** được coi là bằng chứng chuyển sang cohort này.
