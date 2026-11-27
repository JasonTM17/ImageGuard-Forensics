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
| Deepfake / AI cues (`deepfake_cues`) | file này, mục dưới | P08+ ✓ |

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

### Deepfake / AI cues (`deepfake_cues`)

Pipeline gồm sáu nhóm phép xử lí ảnh cổ điển: định vị ROI bằng Haar hoặc ứng
viên màu da YCrCb/HSV + SAT; median residual và Otsu/CCL; LBP; FFT; Canny/Hough;
và đối xứng HOG/Hu. Đây là **dấu hiệu khám phá chưa hiệu chuẩn**, không có mô
hình phân loại Deepfake đã huấn luyện và không trả xác suất ảnh giả.

- Ảnh đơn luôn được phân tích như một ảnh đơn, kể cả panorama. Chế độ hai
  camera cần `capture_layout="dual_side_by_side"` và ranh giới
  `capture_split_x` từ metadata PNG của luồng thu nhận hoặc caller rõ ràng.
  Metadata là khai báo bố cục, không phải chữ ký xác thực nguồn camera.
- Hai miền cảm biến được resize, lọc và đo độc lập. Khi lựa chọn ROI, ưu tiên
  mặt đủ kích thước, tương phản và nền tham chiếu; sau đó mới so độ giảm
  residual. Chỉ một ROI được đánh giá, không bao phủ mọi mặt trong ảnh.
- Không có mặt, ROI nhỏ, tương phản thấp hoặc thiếu nền độc lập →
  `cue_unavailable=true`, `score=null`, `INCONCLUSIVE`. Không lấy chính ROI
  làm nền; không mượn nền camera còn lại. Các số đo vùng trung tâm khi chưa
  thấy mặt chỉ là diagnostic, không sinh mask hay kết luận Deepfake.
- FFT dùng cửa sổ Hann, đỉnh phổ/rolloff và độ tập trung **năng lượng
  `|FFT|²` trong 1% bin cao tần lớn nhất**. Chuẩn hóa theo tỷ lệ bin thực tế
  và tham chiếu nhiễu trắng lý tưởng `1 − log(q)` tránh việc ROI nhỏ tự có
  điểm cao. `fft_hf_conc8` vẫn được xuất để đọc báo cáo cũ nhưng không dùng
  quyết định `grid_score`. Cửa sổ Hann làm các bin phụ thuộc; đây không phải
  kiểm định thống kê. Vải, màn hình và các texture thật tuần hoàn vẫn có thể
  tạo cùng dấu hiệu.
- Cờ không gian cần điểm vượt `deepfake.score_thresh` **và ít nhất hai nhóm**
  texture, phổ, biên cùng hỗ trợ. Smoothing/LBP/diện tích ít nhiễu được gom
  vào một nhóm vì chúng có thể cùng mô tả một hiệu ứng. Các nhóm vẫn tương
  quan và chưa được hiệu chuẩn trên dữ liệu thật/giả. `cue_flag=true` chỉ
  dẫn tới `INCONCLUSIVE`, không bao giờ `SUSPICIOUS_INDICATION`.
- Mốc cùng phiên vẫn so residual/LBP/FFT của ROI 96×96 và có thể nâng
  `NO_INDICATION` lên `INCONCLUSIVE` khi khác biệt vượt ngưỡng. Nó xuất
  `reference_cue_flag`, `ref_status` độc lập và **không cộng điểm** vào chỉ
  số không gian. Không dùng ảnh ghép hai camera làm mốc so sánh mặt.
- Burst một camera có `temporal_status` và `temporal_reason` mô tả mức thay
  đổi giữa các khung. Người thật có thể đứng yên; temporal **không đổi điểm
  hay verdict Deepfake**. Manifest kiểm tra hash của ảnh chính/từng khung;
  không bảo đảm timestamp phần cứng hay chứng minh camera đang live.
- `roi_rect` dùng tọa độ ảnh gốc; `roi_rect_working` dùng tọa độ bản làm việc.
  Heatmap/mask không kéo giãn phần block cuối chưa được đo và không lan qua
  ranh giới cảm biến.

Ngưỡng mới nằm trong `configs/thresholds.yaml`, có miền giá trị và ghi rõ
chưa hiệu chuẩn. Giao thức đánh giá coverage, tỷ lệ bật cờ trên ảnh thật/giả
và AUC điểm: [deepfake-evaluation.md](deepfake-evaluation.md).
