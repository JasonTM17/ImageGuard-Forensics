# Giới hạn đã biết

Tài liệu này tóm tắt **giới hạn đã ghi trong repo** — không thêm con số mới. Đồ án chỉ báo *dấu hiệu nghi vấn*, không khẳng định ảnh giả hay nguyên bản ([`README.md`](../README.md)).

## Kết quả đo trên cohort CASIA đã sàng lọc (103 cặp)

Nguồn số: bảng trong [`README.md`](../README.md), tóm tắt bằng chứng redact [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md), và khung đánh giá ([`evaluation-methodology.md`](evaluation-methodology.md), [`experiments.md`](experiments.md)). Oracle mask tác giả; **không** chứng minh cho ảnh camera tùy ý.

### Cách chọn cohort và giả định oracle (examiner-facing)

Theo bản ghi sự kiện formal (chi tiết + SHA-256 trong [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md)):

- Intake cố định **144** cặp; **7** cặp quarantine (geometry hoặc alias RGB chính xác); **137** survivors.
- Formal chọn **103** cặp (rank 1–103); **34** cặp rank 104–137 **held-aside** — không chạy detector trong sự kiện.
- Oracle: chiếu mask tác giả **blue-complement** (đỏ/xanh lá = foreground, xanh dương = pristine), giữ geometry/orientation.
- **Giả định đã chấp nhận, chưa được chứng minh độc lập:** annotate tác giả đúng vùng; các scene đủ tách biệt — screening crop/RGB/dHash **không** chứng minh độc lập viewpoint hoặc relatedness sau biến đổi.

| Phương pháp | Trên ảnh đã chỉnh (103) | Trên ảnh gốc (103) |
| --- | --- | --- |
| **Keypoint** | 94 coi suspicious; 9 INCONCLUSIVE trên ảnh chỉnh | 22 FP suspicious; **59 INCONCLUSIVE** |
| **Block** | 37 coi suspicious; 3 INCONCLUSIVE | **0** FP suspicious; **0** INCONCLUSIVE |

Ý nghĩa:

- Keypoint trên ảnh gốc: nhiều kết quả **INCONCLUSIVE** (59/103) — abstain là hành vi có chủ đích khi texture kém ([`docs/algorithms/02-keypoint.md`](algorithms/02-keypoint.md)).
- Block trên ảnh gốc: **0/103** FP suspicious trên mẫu này — README nhấn mạnh *không* suy ra FPR luôn bằng 0 mọi nguồn ảnh.
- Block trên ảnh chỉnh: không phải mọi ảnh đều suspicious (66 không suspicious trong cohort).

Macro IoU (mask cue, 103 ảnh chỉnh): keypoint ~0,461; block ~0,289 — baseline trong tài liệu so sánh ([`docs/algorithms/comparison.md`](algorithms/comparison.md)).

**Verdict fusion** trong app/UI **chưa** được đánh giá trong sự kiện đo hai method đơn lẻ trên; kết quả tổng hợp GUI có thể khác pipeline đo từng method ([`README.md`](../README.md)).

Qualification trên CoMoFoD dev **không** được coi là bằng chứng chuyển sang cohort CASIA trên ([`README.md`](../README.md)).

## Giới hạn thuật toán (theo tài liệu thiết kế)

| Chủ đề | Tóm tắt | Tài liệu |
| --- | --- | --- |
| ELA chỉ JPEG | PNG/không JPEG → cue không áp dụng hoặc không đứng một mình | [`docs/algorithms/comparison.md`](algorithms/comparison.md), [`docs/algorithms/03-block.md`](algorithms/03-block.md) |
| Block DCT vs rotate/scale | Biến đổi hình học lớn → hệ số DCT đổi; copy-move rotate/scale có thể **NO_INDICATION** | [`docs/algorithms/03-block.md`](algorithms/03-block.md) |
| Keypoint vs JPEG/nhiễu | Descriptor suy giảm sau nén nặng; vùng phẳng → INCONCLUSIVE | [`docs/algorithms/02-keypoint.md`](algorithms/02-keypoint.md) |
| Splicing cues | Suy giảm trên non-JPEG; exploratory | [`docs/algorithms/comparison.md`](algorithms/comparison.md), [`docs/splicing-study.md`](splicing-study.md) |

## Enhance / AI chỉ xem

Tab Enhance và operator **NAFNet** tạo ảnh xem — **không** gọi `AnalysisService`, **không** đưa pixel enhanced vào fusion verdict ([`docs/user-guide.md`](user-guide.md), [`docs/ui-design.md`](ui-design.md), [`docs/enhancement-study.md`](enhancement-study.md)).

Study cổ điển đã ghi trên cohort thiết kế; arm **NAFNet (E3)** trên nhánh *easy* 256×256, **N=24**: trạng thái **NOT_RUN** (một ảnh 256 thử đơn lẻ không thay thế study đầy đủ) — [`docs/enhancement-study.md`](enhancement-study.md).

## Dataset và đánh giá

- Metric R1–R5 và `INCONCLUSIVE` tách khỏi ma trận nhị phân — xem [`docs/evaluation-methodology.md`](evaluation-methodology.md).
- Artifact đánh giá chi tiết nằm ngoài repo công khai; khi trích số trong báo cáo đồ án, lấy từ [`casia-formal-evidence-summary.md`](casia-formal-evidence-summary.md) hoặc artifact đã lưu cục bộ, không suy diễn ([`README.md`](../README.md)).

## Camera và deepfake khám phá

`deepfake_cues` dùng xử lí ảnh cổ điển, chưa có trọng số mạng phân loại hay
benchmark Deepfake thật/giả trong repo. Da mịn tự nhiên, trang điểm, khử
nhiễu camera, ánh sáng lệch, nền nhiều texture và nén ảnh có thể tạo dấu
hiệu tương tự chỉnh sửa. Đối xứng HOG/Hu không loại trừ được ảnh hưởng của
tư thế; bộ tìm mặt Haar/màu da có thể bỏ sót hoặc định vị nhầm, đặc biệt với
góc nghiêng, che khuất và điều kiện chiếu sáng khác. Màu da không phải bộ
xác nhận khuôn mặt. Phương pháp chỉ đo một ROI được chọn.

Fallback phiên bản v3 kiểm tra hai mắt nên giảm lỗi trên vật thể gradient/vân
đã tái hiện, nhưng có thể bỏ sót mắt bị che và mặt nghiêng. OpenCV phải có
Haar cascade hoạt động; dependency hiện giới hạn `<5`, và evidence hiện rõ
nếu custom build chỉ còn fallback. Không suy số đo trên đối chứng tổng hợp
thành độ chính xác tìm mặt trên mọi màu da/độ tuổi/điều kiện ảnh.

Nhánh logistic DIP tùy chọn cần nhóm tự huấn luyện từ ảnh có nhãn. Nó giữ
quality gate, chỉ hỗ trợ ảnh đơn và không chứa trọng số có sẵn trong repo.
Tham số/threshold có thể overfit dev; grouped OOF không thay thế test độc lập.
Tách theo source_group không tự chứng minh tách danh tính hoặc generator;
nhãn và quan hệ nguồn phải do nhóm xác minh. Sigmoid 0–1 chưa hiệu chuẩn
xác suất; classifier không định vị pixel, heatmap chỉ là diagnostic của DIP.

Bản sửa 2026-11-27 không cấp điểm nếu thiếu mặt, mặt quá nhỏ/ít tương phản
hoặc thiếu nền độc lập. `NO_INDICATION` chỉ có nghĩa không bật cờ trên ROI
đủ điều kiện. `INCONCLUSIVE` có thể do có dấu hiệu hoặc do thiếu dữ liệu;
cần đọc riêng `cue_flag`, `cue_unavailable` và `unavailable_reason`.

FFT đã bỏ cách dùng tỷ lệ top-8 bin cố định để ra cờ, vì tỷ lệ đó thay đổi
mạnh theo diện tích ROI. Phép đo mới dùng năng lượng top 1% bin và tham
chiếu nhiễu trắng; các đối chứng nhiễu/lưới tổng hợp kiểm tra lỗi số học,
không chứng minh độ chính xác nhận diện. Vải, nan chớp, màn hình và texture
thật tuần hoàn vẫn có thể làm tăng điểm. Các ngưỡng phổ và kết hợp tín hiệu
đều chưa hiệu chuẩn trên bộ ảnh Deepfake có nhãn.

Hai camera cần khai báo bố cục và ranh giới cụ thể; ảnh rộng không tự bị
chia đôi. Metadata PNG có thể bị mất khi xuất lại hoặc bị người dùng sửa,
không xác thực nguồn. Khung chụp từ hai thiết bị hiện được đọc tuần tự,
không bảo đảm đồng bộ phần cứng. Mỗi miền được xử lí độc lập, nhưng chỉ
ROI được chọn có kết quả; đây không phải bộ so khớp danh tính đa camera.

Burst 4–6 khung đo độ tĩnh của ROI, **không cộng điểm hoặc đổi verdict**:
người thật có thể đứng yên, còn video phát lại vẫn có thể chuyển động.
Driver có thể buffer frame; khoảng cách đọc danh nghĩa 0.75 giây không
phải timestamp phần cứng. Mốc tham chiếu có thể tạo cờ chênh lệch độc lập,
nhưng không cộng vào điểm không gian; thay đổi góc mặt/ánh sáng/tiêu cự cũng
có thể làm tăng chênh lệch. Không dùng ảnh dual làm mốc tham chiếu mặt.

Reference nay từ chối clipping, tương phản thay đổi quá lớn và cấu trúc không
tương ứng; chỉ căn dịch nguyên nhỏ, không giải quyết mọi thay đổi tư thế,
biểu cảm hoặc tiêu cự. NCC chỉ kiểm tra tương ứng cấu trúc, không xác thực
cùng người. Các ngưỡng clipping/NCC/căn dịch vẫn là ngưỡng thăm dò.

Hash và thời hạn mốc 8 giờ giảm nguy cơ dùng nhầm dữ liệu; không chứng minh
mốc live/cùng người và không chống người sửa đồng bộ ảnh cùng manifest.
Ảnh tăng cường chỉ dùng quan sát, không thay thế ảnh gốc khi phân tích.

Khi đánh giá, phải báo tỷ lệ mẫu đủ điều kiện riêng cho thật/giả cùng tỷ lệ
bật cờ và lý do từ chối đo. Xem [giao thức Deepfake](deepfake-evaluation.md).

## Bảo mật đầu vào (không phải độ chính xác forensics)

Giới hạn tài nguyên `safe_open` (100 MB, 80 MP) và SSRF khi bật ingest HTTP — [`docs/security-error-handling.md`](security-error-handling.md), [`docs/ingest.md`](ingest.md). DNS rebinding vẫn là rủi ro còn lại khi dùng HTTP snapshot.

`scripts/validate_camera_burst.py` cung cấp positive control bằng các file frozen identical để kiểm tra logic end-to-end; đây vẫn không thay thế cohort replay thật qua màn hình/thiết bị phát lại.
