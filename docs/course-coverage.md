# Bản đồ chủ đề môn Xử lý ảnh → ImageGuard Forensics

Tài liệu này giúp sinh viên **trình bày đồ án**: mỗi chủ đề cổ điển được ánh xạ tới **mã nguồn**, **kiểm thử**, **tài liệu** và **ví dụ chạy được** trong repo. Số liệu đánh giá forensics chỉ lấy từ [README.md](../README.md) (mục *Giới hạn đã đo*).

## Ranh giới quan trọng

| Vùng | Vai trò |
| --- | --- |
| `src/imageguard/forensics/` | Detector copy-move / tín hiệu phụ — **ảnh hưởng verdict** (trừ aux: không tự đưa lên `SUSPICIOUS`). |
| `src/imageguard/ipkit/` | Toolkit học tập + demo — **không** gọi từ detector, **không** đổi verdict. |
| `src/imageguard/evaluation/` + `enhancement/` | Khử mờ / nhiễu / metric / bootstrap — phục vụ **nghiên cứu & tab Enhance (xem)**, không phải bằng chứng “ảnh đã phục hồi”. |

Chi tiết contract fusion: module `src/imageguard/forensics/fusion.py` và [evaluation-methodology.md](evaluation-methodology.md).

---

## 1. Lọc không gian (Gaussian, median, unsharp)

### Gaussian (làm mờ / kernel)

| Mục | Nội dung |
| --- | --- |
| **Module (học tập)** | `src/imageguard/ipkit/spatial.py` (`gaussian_kernel`, `gaussian_filter`) — NumPy + `convolve2d` |
| **Module (enhance study)** | `src/imageguard/evaluation/enhance_ops.py` (`gaussian_kernel`, `degrade(..., kind="gaussian")`) |
| **Triển khai enhance** | Kernel: `cv2.getGaussianKernel` + tích ngoài NumPy; convolution: `cv2.filter2D` (không tự viết FFT conv ở đây). |
| **Tests ipkit** | `tests/test_ipkit_spatial.py` — so `cv2.GaussianBlur` (sai lệch ±1 mức xám có thể xảy ra) |
| **Tests enhance** | `tests/test_gaussian_kernel_course.py`, `tests/test_gauss_*_course.py`, `tests/test_enhance_ops.py` |
| **Tài liệu** | [enhancement-study.md](enhancement-study.md), [image-processing-toolkit.md](image-processing-toolkit.md) |
| **Ví dụ ipkit** | `gaussian_filter(gray, size=5, sigma=1.0)` |
| **Ví dụ enhance** | `degrade(..., rng=...)` dùng tham số `rng`, không phải `seed`. |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.evaluation.enhance_ops import degrade; im=np.full((32,32,3),120,np.uint8); out,m=degrade(im,'gaussian',{'ksize':5,'sigma':1.2},rng=None); print(m['kind'], out.shape)"
```

### Median (lọc trung vị)

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/forensics/auxiliary/noise_map.py`, `noise_local.py`; `src/imageguard/evaluation/modelbench.py` |
| **Triển khai** | **`cv2.medianBlur`** trên ảnh xám / map σ; thống kê median/MAD bằng **NumPy**. |
| **Tests** | `tests/test_noise_map_small_course.py`, `tests/test_noise_local_course.py`, `tests/test_median_spike_course.py` |
| **Tài liệu** | [algorithms.md](algorithms.md) (Noise inconsistency), [03-block.md](algorithms/03-block.md) (ngữ cảnh block) |
| **Vai trò forensics** | Aux cue (`noise`, `noise_local`) — **không** đủ để verdict `SUSPICIOUS` một mình. |

**ipkit:** `spatial.median_filter` (NumPy thuần), test `tests/test_ipkit_spatial.py` so `cv2.medianBlur`. Median forensics vẫn nằm trong `noise_map` / `noise_local` như bảng trên.

### Unsharp mask

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/evaluation/enhance_ops.py` (`restore`, kind=`unsharp`); UI: `src/imageguard/enhancement/derive.py` |
| **Triển khai** | `cv2.GaussianBlur` + cộng high-frequency bằng **NumPy** (`amount`). |
| **Tests** | `tests/test_enhance_ops.py`, `tests/test_restore_kind_course.py` |
| **Tài liệu** | [enhancement-study.md](enhancement-study.md), [user-guide.md](user-guide.md) (tab Enhance) |
| **Ví dụ** | Đã kiểm tra với `PYTHONPATH=src` và venv dự án. |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.evaluation.enhance_ops import restore; im=np.full((16,16,3),100,np.uint8); out,_=restore(im,'unsharp',{'ksize':3,'sigma':1.0,'amount':1.0}); print(round(float(out.mean()),2))"
```

---

## 2. Lọc tần số / FFT

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/forensics/auxiliary/freq_view.py` (`FreqView`) |
| **Triển khai** | **NumPy** `np.fft.fft2`, `fftshift`, log-magnitude; tiền xử lý gray bằng `cv2.cvtColor`. Cue blockiness JPEG dùng gradient NumPy trên ảnh gray. |
| **Tests** | `tests/test_freq_course.py`, `tests/test_freq_flat_course.py`, `tests/test_freq_cancel_course.py` |
| **Tài liệu** | [algorithms.md](algorithms.md) (Frequency analysis) |
| **Verdict** | Khám phá — **không** trả `SUSPICIOUS`; spectrum là artifact xem. |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; g=np.random.default_rng(0).integers(0,256,(64,64),dtype=np.uint8).astype(np.float32); f=np.fft.fftshift(np.fft.fft2(g)); print('spec', float(np.log1p(np.abs(f)).mean()))"
```

*(Đoạn trên mô phỏng bước FFT trong `freq_view`, không gọi class detector.)*

Wiener trong `enhance_ops` cũng dùng **FFT NumPy** trên từng kênh (xem mục 10).

---

## 3. DCT (Discrete Cosine Transform)

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/forensics/block/dct_blocks.py` |
| **Triển khai** | **DCT-II tự viết**: ma trận cơ sở `dct_basis` (NumPy), nhân `einsum`; zigzag & lượng tử hóa NumPy. `block_std_map` dùng `cv2.boxFilter` để lọc hộp thống kê block. |
| **Tests** | `tests/test_dct_course.py`, `tests/test_dct_basis2_course.py`, `tests/test_block_course.py`, `tests/test_block.py` |
| **Tài liệu** | [algorithms/03-block.md](algorithms/03-block.md) |
| **Verdict** | Block copy-move — một trong hai detector **conclusive** (cùng keypoint). |

```powershell
$env:PYTHONPATH='src'
python -c "from imageguard.forensics.block.dct_blocks import dct_basis; D=dct_basis(8); print(D.shape, round(float(D[0,0]),3))"
```

---

## 4. Mô hình nhiễu (AWGN, salt-and-pepper, speckle, Poisson)

| Loại | Module | Triển khai | Tests | Ghi chú |
| --- | --- | --- | --- | --- |
| **AWGN** | `evaluation/enhance_ops.py` (`degrade`, `awgn`) | NumPy `Generator.normal` + clip uint8 | `tests/test_awgn_seed_course.py`, `tests/test_enhance_ops.py` | Dùng trong enhance study / synth |
| **Salt & pepper** | `ipkit/noise.py` | **NumPy thuần** (seed cố định) | `tests/test_ipkit_noise.py` | Demo ipkit |
| **Speckle** | `ipkit/noise.py` | NumPy | `tests/test_ipkit_noise.py` | Demo ipkit |
| **Poisson** | `ipkit/noise.py` | NumPy `poisson` | `tests/test_ipkit_noise.py` | Demo ipkit |

**Tài liệu:** [image-processing-toolkit.md](image-processing-toolkit.md).

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.ipkit import salt_pepper; g=np.full((32,32),120,np.uint8); print(salt_pepper(g,0.05,1).shape)"
```

AWGN forensics-grade (degrade):

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.evaluation.enhance_ops import degrade; im=np.full((8,8,3),50,np.uint8); out,_=degrade(im,'awgn',{'sigma':3.0},rng=42); print(out.dtype, int(out[0,0,0]))"
```

---

## 5. Hình thái học

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/ipkit/morphology.py` |
| **Triển khai** | **NumPy thuần** (erode/dilate/open/close/gradient/tophat, SE hình học) |
| **Tests** | `tests/test_ipkit_morphology.py`, `tests/test_disk_*_course.py` (kernel hình học liên quan enhance) |
| **Tài liệu** | [image-processing-toolkit.md](image-processing-toolkit.md) |
| **Forensics** | Không dùng trực tiếp cho verdict; mask keypoint có thể dùng blob Gaussian (OpenCV) trong `forensics/keypoint/mask_builder.py`. |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.ipkit import structuring_element, open_; g=np.full((16,16),200,np.uint8); se=structuring_element('rect',5); print(open_(g,se).mean())"
```

---

## 6. Phát hiện cạnh (Sobel, Prewitt, Laplacian, Canny)

| Toán tử | Module | Triển khai | Tests |
| --- | --- | --- | --- |
| **Sobel / Prewitt / Laplacian** | `ipkit/edges.py` | **NumPy** (`convolve2d` tự viết) | `tests/test_ipkit_edges.py`, `tests/test_io_edges.py` |
| **Canny** | `evaluation/blind_kernel.py` | **`cv2.Canny`** (hỗ trợ ước lượng kernel mù — evaluation) | `tests/test_blind_kernel.py` |

**Tài liệu:** [image-processing-toolkit.md](image-processing-toolkit.md); blind kernel: [enhancement-study.md](enhancement-study.md).

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.ipkit import sobel, gradient_magnitude; g=np.full((32,32),128,np.uint8); gx,gy=sobel(g); print(gx.shape, float(gradient_magnitude(g).max()))"
```

---

## 7. Ngưỡng hóa / phân đoạn (Otsu, thích nghi)

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/ipkit/threshold.py` |
| **Triển khai** | **NumPy** (Otsu trên histogram 256-bin; adaptive mean cửa sổ với integral image) |
| **Tests** | `tests/test_ipkit_threshold.py` |
| **Tài liệu** | [image-processing-toolkit.md](image-processing-toolkit.md) |
| **Forensics** | Không nối trực tiếp vào pipeline verdict. |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.ipkit import otsu_threshold; g=np.concatenate([np.zeros(512), np.full(512,200)]).reshape(32,32).astype(np.uint8); print(otsu_threshold(g))"
```

---

## 8. Đặc trưng SIFT / ORB và so khớp

| Bước | Module | Triển khai | Tests | Tài liệu |
| --- | --- | --- | --- | --- |
| Detect | `forensics/keypoint/detectors.py` | **`cv2.SIFT_create` / `cv2.ORB_create`** | `tests/test_keypoint.py`, `tests/test_orb_backend_course.py` | [02-keypoint.md](algorithms/02-keypoint.md) |
| Self-match | `forensics/keypoint/matchers.py` | **`cv2.BFMatcher`**, Lowe ratio, lọc khoảng cách (NumPy) | `tests/test_match_pairs_course.py`, `tests/test_keypoint_course.py` | [02-keypoint.md](algorithms/02-keypoint.md) |

```powershell
$env:PYTHONPATH='src'
python -c "import cv2, numpy as np; from imageguard.forensics.keypoint.detectors import SiftBackend; g=np.random.default_rng(0).integers(0,256,(128,128),dtype=np.uint8); b=SiftBackend(); kps,desc=b.detect(g); print(len(kps), None if desc is None else desc.shape)"
```

**Trình bày:** SIFT/ORB là thư viện OpenCV — sinh viên giải thích **ý tưởng** (scale-space, descriptor, ratio test), không claim “tự implement SIFT”.

---

## 9. RANSAC / hình học

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/forensics/keypoint/geo_verify.py` |
| **Triển khai** | **`cv2.estimateAffine2D(..., method=cv2.RANSAC)`**; clustering displacement bằng NumPy |
| **Tests** | `tests/test_geo_course.py`, `tests/test_geo_empty_course.py`, `tests/test_keypoint.py` |
| **Tài liệu** | [02-keypoint.md](algorithms/02-keypoint.md) |

---

## 10. Phục hồi ảnh (Wiener, Richardson–Lucy, CLAHE, NLM, NAFNet)

| Phương pháp | Module | Triển khai | Tests |
| --- | --- | --- | --- |
| **Wiener** | `evaluation/enhance_ops.py` | **FFT NumPy** theo kênh; kernel từ metadata degrade | `tests/test_wiener_kernel_course.py`, `tests/test_enhance_ops.py` |
| **Richardson–Lucy** | `evaluation/enhance_ops.py` | Lặp RL; mỗi bước `cv2.filter2D` | `tests/test_iters_zero_course.py`, `tests/test_enhance_ops.py` |
| **CLAHE** | `evaluation/enhance_ops.py` | **`cv2.createCLAHE`** trên kênh L (BGR→LAB) | `tests/test_clahe_gray_course.py`, `tests/test_clahe_flat_course.py` |
| **NLM** | `evaluation/enhance_ops.py` | **`cv2.fastNlMeansDenoisingColored`** | `tests/test_nlm_gray_course.py`, `tests/test_nlm_odd_course.py` |
| **NAFNet** | `enhancement/nafnet.py` | **ONNX Runtime** + tiền/hậu xử lý OpenCV (tab Enhance, tùy chọn) | `tests/test_nafnet.py` |

**Facade UI:** `enhancement/derive.py` — không import Qt trong lõi tính toán.

**Tài liệu:** [enhancement-study.md](enhancement-study.md), [modelforge.md](modelforge.md), [user-guide.md](user-guide.md).

Tất cả restoration trên phục vụ **xem / nghiên cứu**, không chứng minh ảnh “gốc”.

---

## 11. Độ đo (PSNR, SSIM, IoU, F1)

| Độ đo | Module | Triển khai | Tests | Tài liệu |
| --- | --- | --- | --- | --- |
| **PSNR** | `evaluation/enhance_metrics.py` | **NumPy** MSE | `tests/test_psnr_shape_course.py`, `tests/test_psnr_delta_course.py` | [enhancement-study.md](enhancement-study.md) |
| **SSIM** | `evaluation/enhance_metrics.py` | Công thức Wang 2004; cửa sổ Gaussian **`cv2.GaussianBlur`** | `tests/test_ssim_flat_course.py`, `tests/test_ssim_small_course.py` | [enhancement-study.md](enhancement-study.md) |
| **IoU / F1 (pixel & image)** | `evaluation/metrics.py` | **NumPy**; quy tắc R1–R5 | `tests/test_metrics.py`, `tests/test_image_f1_course.py`, `tests/test_iou_half_course.py` | [evaluation-methodology.md](evaluation-methodology.md) |

```powershell
$env:PYTHONPATH='src'
python -c "import numpy as np; from imageguard.evaluation.enhance_metrics import psnr; a=np.full((8,8),10,np.uint8); b=a.copy(); b[0,0]=20; print(round(psnr(b,a),3))"
```

Runner đánh giá detector: `evaluation/runner.py` — dùng `metrics.image_metrics` / `pixel_metrics`.

---

## 12. Nén JPEG và ELA

| Chủ đề | Module | Triển khai | Tests | Verdict |
| --- | --- | --- | --- | --- |
| **JPEG encode/degrade** | `evaluation/enhance_ops.py` (`degrade`, `jpeg`) | `cv2.imencode` / `imdecode` | `tests/test_jpeg_quality_course.py`, `tests/test_splicing_jpeg_course.py` | Không phải detector |
| **ELA** | `forensics/auxiliary/ela.py` | Re-encode JPEG (PIL + OpenCV); heatmap chênh | `tests/test_ela.py`, `tests/test_ela_course.py` | Aux — **tối đa `INCONCLUSIVE`**, không `SUSPICIOUS` đơn lẻ |

**Tài liệu:** [algorithms.md](algorithms.md) (mục ELA), [user-guide.md](user-guide.md).

Blockiness JPEG (liên quan chương JPEG 8×8): `freq_view.py` (mục 2).

---

## 13. Thống kê (bootstrap)

| Mục | Nội dung |
| --- | --- |
| **Module** | `src/imageguard/evaluation/enhance_bootstrap.py` (`bootstrap_mean`, `paired_bootstrap_mean`) |
| **Triển khai** | **NumPy** percentile bootstrap (không BCa) |
| **Sử dụng** | `evaluation/enhance_study.py` gom khoảng tin cậy metric enhance |
| **Tests** | `tests/test_enhance_bootstrap.py`, `tests/test_bootstrap_course.py`, `tests/test_boot_*_course.py` |
| **Tài liệu** | [enhancement-study.md](enhancement-study.md), [evaluation-methodology.md](evaluation-methodology.md) |

`modelbench.py` có thêm **Mann–Whitney U** (NumPy) cho tách σ nhiễu — bổ trợ, không thay bootstrap.

---

## Bài giảng Chương 2–7 (slides dunghv) → `imageguard.ipkit`

Nguồn slide: `plans/reports/k2/lectures/C2.txt` … `C7.txt` (bản extract; PDF cùng thư mục trên repo chính). **ipkit không đổi verdict** — cột *OpenCV trong test* chỉ ghi oracle pytest, không phải triển khai `src/`.

### Chương 2 — Image Enhancement in Spatial Domain

| Mục slide | Chủ đề | Module / hàm | Test | OpenCV trong test | Chưa có trong ipkit |
| --- | --- | --- | --- | --- | --- |
| §2 | Biến đổi cường độ (negative, log, gamma, stretch, …) | `intensity.*` | `test_ipkit_intensity.py` | Một số LUT | Biến đổi hình học (scale/rotate/translate) |
| §3 | Histogram, equalize, matching, local | `histogram.*` | `test_ipkit_histogram.py` | `cv2.equalizeHist` cho `equalize` | — |
| §4–§6 | Lọc không gian, làm sắc | `spatial.*` (+ `edges.convolve2d`) | `test_ipkit_spatial.py` | blur, GaussianBlur, medianBlur | — |
| (nhiễu demo) | Salt-pepper, speckle, Poisson | `noise.*` | `test_ipkit_noise.py` | Không | — |
| (biên) | Sobel, Prewitt, Laplacian | `edges.*` | `test_ipkit_edges.py`, `test_ipkit_opencv_reference.py` | Sobel, Laplacian, filter2d | — |
| (ngưỡng) | Otsu, adaptive mean | `threshold.*` | `test_ipkit_threshold.py` | Otsu, adaptiveThreshold | — |

### Chương 3 — Frequency Domain

| Mục slide | Chủ đề | Module / hàm | Test | OpenCV trong test | Chưa có |
| --- | --- | --- | --- | --- | --- |
| §2 | DFT, phổ | `frequency.dft2_centered`, `magnitude_spectrum` | `test_ipkit_frequency.py` | Không (NumPy FFT) | — |
| §3–§4 | Low/high pass ideal, Butterworth, Gaussian; sharpen | `frequency.*_lowpass`, `*_highpass`, `apply_frequency_filter`, `sharpen_frequency` | `test_ipkit_frequency.py` | So lỏng `GaussianBlur` cho một case không gian | Demo “OpenCV in use” trên slide (không port vào ipkit) |

### Chương 4 — Morphological Image Processing

| Mục slide | Chủ đề | Module / hàm | Test | OpenCV trong test | Chưa có |
| --- | --- | --- | --- | --- | --- |
| §2–§3 | Erosion, dilation, open/close, gradient, tophat | `morphology.*` | `test_ipkit_morphology.py`, `test_ipkit_opencv_reference.py` | erode, dilate, morphologyEx | Ellipse SE lớn ≠ xấp xỉ OpenCV |
| §4 | Hit-miss, boundary, fill, CCL, hull, thin, skeleton | `morph_algos.*` | `test_ipkit_morph_algos.py` | Một phần CCL/morph nhị phân | — |

### Chương 5 — Segmentation

| Mục slide | Chủ đề | Module / hàm | Test | OpenCV trong test | Chưa có / khác |
| --- | --- | --- | --- | --- | --- |
| §5.2 | Point/line/edge, Canny, Hough | `segmentation.detect_*`, `canny_edges`, `hough_lines` | `test_ipkit_segmentation.py` | `cv2.Canny` (IoU lỏng) | Canny/Hough không khớp từng pixel OpenCV |
| §5.3 | Threshold lặp | `segmentation.iterative_global_threshold` + `threshold.otsu_threshold` | `test_ipkit_segmentation.py`, `test_ipkit_threshold.py` | Otsu trong threshold | — |
| §5.4 | Region growing | `segmentation.region_growing` | `test_ipkit_segmentation.py` | Không | — |
| §5.5 | k-means, watershed | `kmeans_segmentation`, `simple_watershed` | `test_ipkit_segmentation.py` | Không | Watershed hình thái đầy đủ trên slide |

### Chương 6 — Feature Extraction

| Mục slide | Chủ đề | Module / hàm | Test | OpenCV trong test | Chưa có |
| --- | --- | --- | --- | --- | --- |
| §2 | Moments, Hu | `features_basic.*` | `test_ipkit_features_basic.py` | Không | — |
| §3 | Haar / SAT | `integral_image`, `haar_feature*` | `test_ipkit_features_basic.py` | Không | — |
| §4 | PCA, LDA, LBPH | `features_projection.*`, `features_basic.lbph` | `test_ipkit_features_projection.py`, `test_ipkit_features_basic.py` | Không | — |
| §5 | **SIFT, SURF** | — | — | — | **Không implement** (ý tưởng slide; forensics: `cv2.SIFT` trong keypoint) |
| §6 | HOG | `features_hog.*` | `test_ipkit_features_hog.py` | `cv2.HOGDescriptor` khi cấu hình khớp | — |

### Chương 7 — Pattern Recognition

| Mục slide | Chủ đề | Module / API | Test | OpenCV trong test | Chưa có |
| --- | --- | --- | --- | --- | --- |
| §7.1 | Pattern classes, minimum distance | `minimum_distance_classifier`, `KNNClassifier` | `test_ipkit_classifiers.py` | Không | — |
| §7.2 | Linear SVM | `LinearSVM`, `OneVsRestSVM` | `test_ipkit_classifiers.py` | Không | Kernel SVM phi tuyến đầy đủ |
| §7.3 | MLP, back-prop | `MLPClassifier` | `test_ipkit_classifiers.py` | Không | — |
| §7.4 | CNN | `conv2d_valid`, `relu`, `max_pool2d`, `flatten` | `test_ipkit_classifiers.py`, `test_ipkit_validation.py` | Không | **Huấn luyện CNN sâu / demo OpenCV DNN trên slide** |

---

## Gói `imageguard.ipkit` (tổng hợp)

| File | Chương slide | NumPy thuần trong `src/`? | Verdict |
| --- | --- | --- | --- |
| `ipkit/intensity.py` | C2 §2 | Có | Không |
| `ipkit/histogram.py` | C2 §3 | Có | Không |
| `ipkit/spatial.py` | C2 §4–6 | Có | Không |
| `ipkit/noise.py` | C2 (mô hình nhiễu) | Có | Không |
| `ipkit/edges.py` | C2/C5 biên | Có | Không |
| `ipkit/threshold.py` | C2/C5 ngưỡng | Có | Không |
| `ipkit/frequency.py` | C3 | Có (`np.fft` OK) | Không |
| `ipkit/morphology.py` | C4 §2–3 | Có | Không |
| `ipkit/morph_algos.py` | C4 §4 | Có | Không |
| `ipkit/segmentation.py` | C5 | Có | Không |
| `ipkit/features_basic.py` | C6 §2–4 | Có | Không |
| `ipkit/features_projection.py` | C6 §4 | Có | Không |
| `ipkit/features_hog.py` | C6 §6 | Có | Không |
| `ipkit/classifiers.py` | C7 | Có | Không |

Điểm vào API: `src/imageguard/ipkit/__init__.py`. Mô tả đầy đủ: [image-processing-toolkit.md](image-processing-toolkit.md).

Kiểm thử gói (một lần, đã đo): `python -m pytest tests/ -k ipkit -q -p no:cacheprovider` → **263 passed, 1 skipped** (~7,88 s).

---

## Detector forensics vs chỉ “wrapped OpenCV / thư viện”

| Thành phần | Gọi thư viện | Ảnh hưởng verdict tổng? |
| --- | --- | --- |
| Keypoint SIFT/ORB + BFMatcher + RANSAC | OpenCV | Có — khi trả `SUSPICIOUS_INDICATION` |
| Block DCT + voting | NumPy (+ boxFilter OpenCV) | Có |
| ELA, noise, noise_local, freq, splicing, EXIF | Hỗn hợp OpenCV/NumPy/PIL | **Không** đẩy lên `SUSPICIOUS` đơn lẻ (aux contract) |
| `ipkit`, enhance restore/degrade, NAFNet | NumPy/OpenCV/ONNX | **Không** (xem / đánh giá) |
| `fusion.combine` | Logic enum | Kết hợp kết quả method — **chưa được đánh giá formal** như từng method đơn (README) |

Registry method: `src/imageguard/forensics/registry.py`.

---

## Điểm cần nói thật khi trình bày

1. **SIFT/ORB** lấy từ OpenCV (`detectors.py`); phần “đồ án IP” nằm ở matching, RANSAC, mask, luật verdict và block DCT tự viết — không nhầm với implement SIFT gốc.
2. Công cụ chỉ báo **dấu hiệu nghi vấn**, không khẳng định ảnh giả / nguyên bản (README, phần ranh giới khoa học).
3. Số trên cohort **103 cặp CASIA đã sàng lọc** (README, *Giới hạn đã đo*):
   - **Keypoint:** ảnh chỉnh — 94 suspicious (9 FN); 9 INCONCLUSIVE trên ảnh chỉnh; ảnh gốc — 22 FP suspicious; **59 INCONCLUSIVE** (trong 81 hàng nonsuspicious theo confusion).
   - **Block:** ảnh chỉnh — 37 suspicious (66 không suspicious); 3 INCONCLUSIVE; ảnh gốc — **0 FP suspicious**; 0 INCONCLUSIVE.
   - Macro IoU mask (103 ảnh chỉnh): keypoint ~0,461; block ~0,289 (baseline `.01` trong README).
   - **Verdict fusion trong app/UI chưa được đánh giá** trong sự kiện đo hai method trên.
4. Không dùng số từ CoMoFoD dev để overclaim lên CASIA (README).
5. Tab **Enhance** và **ipkit** minh họa môn học — giải thích rõ trước hội đồng để tránh hiểu nhầm là “khôi phục bằng chứng”.

---

## Lệnh kiểm tra nhanh cho sinh viên

```powershell
# Toàn bộ ipkit (không cần ảnh thật)
$env:PYTHONPATH='src'
python -m pytest tests/ -k ipkit -q -p no:cacheprovider

# Demo slide chương C2–C7 (offline)
python scripts/ipkit_demo.py

# Mẫu nhỏ: ipkit + enhance + DCT forensics
python -m pytest tests/test_ipkit_edges.py tests/test_enhance_ops.py tests/test_dct_course.py -q -p no:cacheprovider
```

CLI phân tích (cần đường dẫn ảnh JPEG/PNG hợp lệ):

```powershell
python -m imageguard analyze <ảnh> --methods keypoint,block,ela
```

Chi tiết: [user-guide.md](user-guide.md), [testing-strategy.md](testing-strategy.md).

---

## Chỉ mục tài liệu liên quan

- [architecture.md](architecture.md) — lớp kiến trúc
- [algorithms/comparison.md](algorithms/comparison.md) — so sánh keypoint vs block
- [limitations.md](limitations.md) — giới hạn đã biết
- [failure-cases.md](failure-cases.md) — mẫu thất bại
- [report/capstone-skeleton.md](report/capstone-skeleton.md) — khung báo cáo
