# Glossary — Thuật ngữ Xử lý ảnh trong ImageGuard Forensics

Giải thích ngắn các thuật ngữ tiếng Anh được dùng trong mã nguồn và tài liệu của
đồ án. Mỗi mục ghi đường dẫn thật nơi thuật ngữ hoặc phần cài đặt liên quan xuất
hiện (đã kiểm tra bằng script) và key cấu hình tương ứng trong
`configs/thresholds.yaml` khi có. Đây là tài liệu học tập, không phải đặc tả mới
hay lời giải thích thay thế tài liệu thuật toán chi tiết.

## 1. Keypoint và matching

### SIFT — Scale-Invariant Feature Transform

SIFT phát hiện keypoint trong không gian scale (Gaussian/DoG) và mô tả chúng
bằng descriptor bất biến với scale, rotation và thay đổi sáng cục bộ. Trong repo,
SIFT là backend chính của detector copy-move; ORB là fallback.

- Xuất hiện: `src/imageguard/forensics/keypoint/detectors.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.backend` (`sift`), `keypoint.nfeatures` (`0` = để OpenCV
  quyết định).

### Keypoint — điểm đặc trưng

Keypoint là điểm ảnh được detector chọn vì ổn định dưới các biến đổi cục bộ, kèm
vị trí, scale và hướng. Detector copy-move so khớp keypoint của ảnh với chính nó;
nếu tổng số keypoint quá ít thì kết quả là thiếu bằng chứng, không phải "không có
dấu hiệu".

- Xuất hiện: `src/imageguard/forensics/keypoint/detectors.py`,
  `src/imageguard/forensics/keypoint/pipeline.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.k_min` (`50`) — dưới ngưỡng này trả `INCONCLUSIVE`;
  `keypoint.min_match_dist` (`20` px) loại các cặp keypoint quá sát nhau.

### Descriptor — vector mô tả đặc trưng

Descriptor là vector số mô tả vùng lân cận của một keypoint; hai vùng giống nhau
cho descriptor gần nhau nên có thể so khớp bằng khoảng cách. Tài liệu của repo mô
tả SIFT dùng descriptor 128 chiều, còn ORB dùng descriptor nhị phân.

- Xuất hiện: `src/imageguard/forensics/keypoint/matchers.py`,
  `src/imageguard/forensics/keypoint/detectors.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.backend` quyết định loại descriptor và norm so khớp.

### Lowe ratio test — kiểm tra tỉ số Lowe

Lowe ratio test chỉ giữ một match khi khoảng cách tới láng giềng gần nhất nhỏ hơn
hẳn khoảng cách tới láng giềng thứ hai, cụ thể là `d1/d2 < ρ`. Cách này loại các
match mơ hồ do texture lặp, vì khi đó hai khoảng cách xấp xỉ nhau.

- Xuất hiện: `src/imageguard/forensics/keypoint/matchers.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.ratio_rho` (`0.8`, khoảng cho phép `[0.6, 0.9]`).

### RANSAC — RANdom SAmple Consensus

RANSAC lặp lại việc lấy một mẫu điểm tối thiểu để fit model rồi đếm số inlier,
nhờ đó ước lượng được model ngay cả khi dữ liệu có outlier. Trong repo, RANSAC fit
biến đổi affine từ các cặp keypoint và giữ các cặp có residual dưới ngưỡng.

- Xuất hiện: `src/imageguard/forensics/keypoint/geo_verify.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.ransac_residual` (`3.0` px) là dung sai inlier;
  `keypoint.min_cluster` (`3`) là số cặp tối thiểu cho một cụm.

### Affine transform — biến đổi affine

Biến đổi affine ánh xạ điểm theo `x' = A·x + t` với ma trận 2×3 (6 bậc tự do),
bao gồm tịnh tiến, quay, scale và shear. Repo dùng nó làm model hình học cho một
vùng copy-move, với các tham số được kiểm chứng bằng RANSAC.

- Xuất hiện: `src/imageguard/forensics/keypoint/geo_verify.py`,
  `docs/algorithms/02-keypoint.md`
- Config: `keypoint.ransac_residual` (dung sai inlier của affine RANSAC).

## 2. Block và DCT

### DCT — Discrete Cosine Transform

DCT chuyển một block ảnh sang miền hệ số cosine; với ảnh tự nhiên, năng lượng
thường tập trung ở các hệ số tần số thấp. Repo dùng DCT-II cho từng block và giữ
một số hệ số đầu theo thứ tự zigzag làm feature.

- Xuất hiện: `src/imageguard/forensics/block/dct_blocks.py`,
  `docs/algorithms/03-block.md`
- Config: `block.block_size` (`16` px), `block.n_coeffs` (`24`).

### Block matching — so khớp block

Block matching chia ảnh thành các block overlap B×B với stride 1, trích feature
rồi tìm các block giống nhau để lộ vùng copy-move. Cách này dày (dense) nhưng chỉ
nhạy với tịnh tiến, không bất biến với rotate/scale.

- Xuất hiện: `docs/algorithms/03-block.md`,
  `src/imageguard/forensics/block/voting.py`
- Config: `block.block_size` (`16`), `block.sim_thresh` (`0.95`),
  `block.min_shift_votes` (`8`), `block.flat_std_eps` (`5.0`).

### JPEG quantization — lượng tử hoá JPEG

JPEG quantization chia hệ số DCT cho bước lượng tử `Q` rồi làm tròn, tạo mất mát
có chủ đích và để lại dấu vết theo bảng lượng tử. Trong repo, khái niệm này xuất
hiện khi giải thích phân bố lỗi ELA; pipeline block cũng lượng tử hoá feature DCT
của nó bằng một bảng `Q` tăng theo tần số.

- Xuất hiện: `docs/algorithms.md`,
  `src/imageguard/forensics/block/dct_blocks.py`
- Config: `ela.qualities` (`[75, 85, 90, 95]`) là các quality re-encode, không
  phải bảng lượng tử JPEG.

## 3. ELA và noise

### ELA — Error Level Analysis

ELA re-encode ảnh JPEG ở một quality rồi đo `|I − R_q(I)|`; vùng có lịch sử nén
khác thường lộ mức lỗi khác với phần còn lại. Repo sweep nhiều quality, lấy giá trị
lớn nhất, khuếch đại rồi chỉ trả tối đa `INCONCLUSIVE`.

- Xuất hiện: `src/imageguard/forensics/auxiliary/ela.py`,
  `docs/algorithms.md`
- Config: `ela.qualities`, `ela.amplify` (`15`), `ela.indication_p99` (`0.2`).

### Noise residual — phần dư nhiễu

Noise residual là phần ảnh còn lại sau khi trừ một bản làm mịn, ví dụ
`r = I − medianBlur(I, k)`; nó giữ thành phần tần số cao dùng để ước lượng nhiễu.
Repo đo độ lệch chuẩn σ của residual theo block rồi so với median/MAD toàn ảnh để
tìm vùng bất thường.

- Xuất hiện: `src/imageguard/core/preprocess.py`,
  `src/imageguard/forensics/auxiliary/noise_map.py`, `docs/algorithms.md`
- Config: `noise.median_k` (`5`), `noise.block` (`16`),
  `noise.outlier_mad` (`3.0`).

## 4. Toán tử xử lý ảnh cổ điển

### CLAHE — Contrast Limited Adaptive Histogram Equalization

CLAHE cân bằng histogram theo từng tile cục bộ và giới hạn độ tương phản để tránh
khuếch đại nhiễu. Repo dùng CLAHE nhẹ trước khi detect keypoint và trong các phép
tăng cường xem.

- Xuất hiện: `src/imageguard/core/preprocess.py`,
  `src/imageguard/evaluation/enhance_ops.py`
- Config: không có key riêng; tham số `clipLimit=2.0` và
  `tileGridSize=(8, 8)` nằm trong code.

### Canny và Sobel — phát hiện cạnh

Sobel là toán tử gradient dùng hai kernel 3×3 để tính đạo hàm theo x và y; độ lớn
gradient đánh dấu cạnh. Canny là pipeline phát hiện cạnh gồm làm mịn, gradient,
non-maximum suppression và hysteresis threshold. Repo cài Sobel bằng NumPy trong
toolkit học tập, còn Canny chỉ dùng trong evaluation để lấy profile cạnh phục vụ
ước lượng kernel mờ.

- Xuất hiện: `src/imageguard/ipkit/edges.py` (Sobel),
  `src/imageguard/evaluation/blind_kernel.py` (Canny),
  `docs/course-coverage.md`
- Config: không có key riêng; Canny dùng ngưỡng `35` và `105` trong code.

### Morphology — hình thái học

Morphology là nhóm toán tử erode/dilate/open/close/gradient/tophat dựa trên một
structuring element, dùng để nối hoặc loại các vùng nhỏ trong mask. Repo cài đặt
bằng NumPy trong toolkit và dùng morphology của OpenCV để làm sạch mask
keypoint/block.

- Xuất hiện: `src/imageguard/ipkit/morphology.py`,
  `src/imageguard/forensics/keypoint/mask_builder.py`,
  `src/imageguard/forensics/block/voting.py`
- Config: không có key riêng; kích thước kernel được tính trong code.

## 5. Đánh giá

### IoU — Intersection over Union

IoU đo mức chồng lấn giữa mask dự đoán và mask ground truth:
`TP / (TP + FP + FN)`. Repo tính IoU theo từng ảnh rồi tổng hợp macro/micro, với
quy tắc R1–R5 xử lý mask rỗng.

- Xuất hiện: `src/imageguard/evaluation/metrics.py`,
  `docs/evaluation-methodology.md`
- Config: không có key riêng.

### F1 — F1 score

F1 là trung bình điều hoà của precision và recall: `2·P·R / (P + R)`, được dùng ở
mức ảnh và mức pixel. Repo xử lý riêng trường hợp mask rỗng để không tự động cho
`F1 = 1`.

- Xuất hiện: `src/imageguard/evaluation/metrics.py`,
  `docs/evaluation-methodology.md`
- Config: không có key riêng.

## 6. Verdict enum (đóng)

`Verdict` là enum đóng gồm đúng năm giá trị dưới đây, định nghĩa tại
`src/imageguard/core/enums.py` và được mô tả trong
`docs/forensic-report-schema.md`. Mỗi giá trị là một mức *chỉ báo* (indication
level) của công cụ, không phải phán quyết ảnh thật hay ảnh giả mạo.

| Giá trị | Ý nghĩa như một mức chỉ báo |
|---|---|
| `suspicious_indication` | Có dấu hiệu nghi vấn vượt ngưỡng của method (detector conclusive như keypoint/block); fusion có thể đẩy `overall_verdict` lên mức này. Đây là mức chỉ báo mạnh nhất, không phải kết luận giả mạo. |
| `no_indication` | Method không thấy dấu hiệu đủ mạnh theo ngưỡng; cũng không chứng minh ảnh nguyên bản. |
| `inconclusive` | Tín hiệu yếu, mâu thuẫn, thiếu dữ liệu, hoặc chỉ mang tính khám phá (thường gặp ở auxiliary method). |
| `not_applicable` | Method không chạy được cho ảnh/định dạng này (ví dụ ELA trên ảnh không phải JPEG) và bị loại khỏi fusion. |
| `error` | Method lỗi hoặc bị huỷ; fusion ghi cảnh báo và CLI vẫn có thể thoát với mã `0`. |

Các đầu ra của ImageGuard Forensics chỉ là **chỉ báo (indications)** cần điều tra
thêm, không phải bằng chứng hay kết luận rằng một ảnh là giả mạo.
