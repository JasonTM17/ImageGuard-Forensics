# Image Processing Toolkit (`ipkit`)

**Lưu ý quan trọng:** `imageguard.ipkit` chỉ phục vụ **xem/demo** và học tập trong môn Xử lý ảnh (HCMUTE, PGS.TS. Hoàng Văn Dũng — dunghv). Các toán tử **không** được gọi bởi detector pháp y, **không** ảnh hưởng `Verdict` hay pipeline forensics.

Mọi thuật toán lõi dùng **NumPy** (không `cv2`/`scipy`/`skimage` trong `src/imageguard/ipkit/`). OpenCV chỉ dùng trong pytest để đối chiếu khi có toán tử tương đương; tolerance được ghi trong test, không khẳng định độ chính xác forensics.

Điểm vào: `src/imageguard/ipkit/__init__.py` (một số hàm được re-export; module đầy đủ import theo tên, ví dụ `from imageguard.ipkit import intensity`).

Bản đồ slide → mã: [course-coverage.md](course-coverage.md) (mục *Bài giảng Chương 2–7*).

---

## Cường độ điểm — `intensity.py` (Chương 2, mục 2)

Phép toán điểm (LUT 256 mức), áp dụng từng kênh cho ảnh màu.

| Hàm | Tính toán | Tham số chính |
|-----|-----------|---------------|
| `negative` | \(s = L-1-r\) (L=256) | — |
| `log_transform` | \(s = c\log(1+r)\) | `c` (tự chọn nếu `None`) |
| `gamma_transform` | \(s = c\, r^\gamma\) | `gamma`, `c` |
| `contrast_stretch` | kéo giãn theo percentile | `p_low`, `p_high` |
| `piecewise_linear` | biến đổi tuyến tính khúc | `points` |
| `threshold_binary` | ngưỡng cứng 0/255 | `thresh` |
| `bit_plane_slice` | mặt phẳng bit | `bit` 0–7 |

```python
import numpy as np
from imageguard.ipkit.intensity import negative, gamma_transform

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
print(int(negative(gray)[0, 0]))  # 255 (đã chạy: PYTHONPATH=src, venv dự án)
dark = gamma_transform(gray, gamma=2.2)
```

**Tests:** `tests/test_ipkit_intensity.py` — đối chiếu một số LUT với `cv2` khi có (ví dụ log/gamma làm tròn).

---

## Histogram — `histogram.py` (Chương 2, mục 3)

| Hàm | Tính toán | Tham số chính |
|-----|-----------|---------------|
| `histogram` | đếm 256-bin | — |
| `normalized_histogram` | chuẩn hóa về tổng 1 | — |
| `cdf` | CDF tích lũy | — |
| `equalize` | cân bằng toàn cục qua CDF | — |
| `match_histogram` | đặc tả histogram | `reference` (ảnh hoặc hist) |
| `local_equalize` | HE cục bộ + clip | `tile`, `clip_limit` |

```python
import numpy as np
from imageguard.ipkit.histogram import equalize

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
eq = equalize(gray)
assert eq.shape == gray.shape and int(eq.max()) == 255
```

**Tests:** `tests/test_ipkit_histogram.py` — `equalize` so với `cv2.equalizeHist` (sai lệch làm tròn ≤ 1 mức xám trên nhiều mẫu).

---

## Lọc không gian — `spatial.py` (Chương 2, mục 4–6)

Tích chập qua `edges.convolve2d`, lề `replicate`.

| Hàm | Tính toán | Tham số chính |
|-----|-----------|---------------|
| `box_kernel`, `gaussian_kernel` | tạo kernel | `size`, `sigma` |
| `mean_filter` | lọc trung bình | `size` (lẻ) |
| `gaussian_filter` | Gaussian | `size`, `sigma` |
| `median_filter`, `min_filter`, `max_filter`, `midpoint_filter` | thống kê bậc | `size` |
| `laplacian_sharpen` | \(g = f - c\nabla^2 f\) | `c` |
| `unsharp_mask` | làm sắc unsharp | `size`, `sigma`, `amount` |
| `highboost` | high-boost | `size`, `sigma`, `k` |

```python
import numpy as np
from imageguard.ipkit.spatial import gaussian_filter

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
blur = gaussian_filter(gray, size=5, sigma=1.0)
print(round(float(blur.mean()), 2))  # 127.5 (đã chạy)
```

**Tests:** `tests/test_ipkit_spatial.py` — `mean_filter`/`gaussian_filter`/`median_filter` vs `cv2.blur`/`GaussianBlur`/`medianBlur` (sai lệch biên/uint8 có thể ±1).

**Chưa có trong ipkit (slide C2):** biến đổi hình học (scale, rotate, translate).

---

## Nhiễu (`noise.py`)

| Hàm | Mô tả |
|-----|--------|
| `salt_pepper(img, amount, seed)` | Đặt ngẫu nhiên `amount` tỷ lệ pixel thành 0 hoặc 255 (một nửa muối, một nửa tiêu). |
| `speckle(img, sigma, seed)` | \(I' = \mathrm{clip}(\mathrm{round}(I \cdot (1 + \sigma \mathcal{N}(0,1))))\) |
| `poisson(img, seed)` | \(I'_{p} \sim \mathrm{Poisson}(I_p)\), clip về \([0,255]\) |

```python
import numpy as np
from imageguard.ipkit import salt_pepper, speckle, poisson

gray = np.full((64, 64), 120, dtype=np.uint8)
noisy = salt_pepper(gray, amount=0.05, seed=1)
```

**Tests:** `tests/test_ipkit_noise.py` (không dùng `cv2`).

---

## Biên / gradient (`edges.py`)

Convolution 2D tự viết (`convolve2d`) với lề `zero` hoặc `replicate`.

| Hàm | Kernel |
|-----|--------|
| `sobel` | \(G_x, G_y\) Sobel 3×3 chuẩn |
| `prewitt` | Prewitt 3×3 |
| `laplacian` | \([0,1,0; 1,-4,1; 0,1,0]\) |
| `gradient_magnitude` | \(\sqrt{G_x^2 + G_y^2}\) (từ Sobel) |

```python
from imageguard.ipkit import sobel, gradient_magnitude

gx, gy = sobel(gray)
mag = gradient_magnitude(gray)
```

**Tests:** `tests/test_ipkit_edges.py`, `tests/test_ipkit_opencv_reference.py` — khớp `cv2.Sobel`/`Laplacian`/`filter2d` trên kernel tương ứng.

---

## Ngưỡng (`threshold.py`)

| Hàm | Công thức / ý tưởng |
|-----|---------------------|
| `otsu_threshold(gray)` | Tối đa hóa \(\sigma_B^2(t)\) trên histogram 256-bin; trả về `int` ngưỡng. |
| `adaptive_mean_threshold(gray, block, c)` | Pixel \(>\) (trung bình cục bộ \(block×block\) \(- c\)) → 255, ngược lại 0. |

Trên một phần nhỏ pixel, `adaptive_mean_threshold` có thể khác `cv2.adaptiveThreshold` (làm tròn / tie-break ở biên cửa sổ) vì đường mean cục bộ dùng `float64` thay vì đường integer của OpenCV. Test `test_adaptive_mean_vs_cv2_interior` (seed cố định) so khớp với tolerance tỷ lệ mismatch \(\le 0{,}001\).

**Độ phức tạp (sau tối ưu):** `adaptive_mean_threshold` dùng ảnh tích phân (cumulative sum trên ảnh edge-padded) — thời gian \(O(HW)\), bộ nhớ \(O(HW)\).

```python
from imageguard.ipkit import otsu_threshold, adaptive_mean_threshold

t = otsu_threshold(gray)
binary = adaptive_mean_threshold(gray, block=11, c=4.0)
```

**Tests:** `tests/test_ipkit_threshold.py`.

---

## Miền tần số — `frequency.py` (Chương 3)

DFT 2D (`np.fft`), lọc lý tưởng / Butterworth / Gaussian, Laplacian miền tần số.

| Hàm | Tính toán | Tham số chính |
|-----|-----------|---------------|
| `dft2_centered` | DFT + `fftshift` | — |
| `magnitude_spectrum` | biên độ (log tùy chọn) | `log` |
| `distance_grid` | lưới khoảng cách tâm phổ | `shape` |
| `ideal_lowpass`, `butterworth_lowpass`, `gaussian_lowpass` | mask thông thấp | `shape`, `d0` (và `order` cho Butterworth) |
| `ideal_highpass`, `butterworth_highpass`, `gaussian_highpass` | mask thông cao | tương tự |
| `laplacian_frequency` | bộ lọc Laplacian \(-4\pi^2 D^2\) | `shape` |
| `apply_frequency_filter` | nhân mask + IFFT → uint8 | `mask` |
| `sharpen_frequency` | high-boost miền tần số | `d0`, `k` |

```python
import numpy as np
from imageguard.ipkit.frequency import ideal_lowpass, apply_frequency_filter

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
mask = ideal_lowpass((32, 32), d0=8.0)
out = apply_frequency_filter(gray, mask)
assert out.shape == gray.shape and out.dtype == np.uint8
```

**Tests:** `tests/test_ipkit_frequency.py` — oracle chủ yếu NumPy/`np.fft`; một test so sánh lỏng với `cv2.GaussianBlur` cho lọc tương đương không gian.

---

## Hình thái học gray — `morphology.py` (Chương 4, mục 2–3)

Phần tử cấu trúc: `structuring_element("rect" | "cross" | "ellipse", ksize)` (kích thước lẻ).

Gray-scale erosion/dilation = min/max trong footprint của `se`:

- `erode`, `dilate`
- `open_` = dilate ∘ erode
- `close_` = erode ∘ dilate
- `gradient` = dilate − erode (clip uint8)
- `tophat` = ảnh − open

```python
from imageguard.ipkit import structuring_element, open_, tophat

se = structuring_element("ellipse", 5)
clean = open_(gray, se)
```

**Tests:** `tests/test_ipkit_morphology.py`, `tests/test_ipkit_opencv_reference.py` — khớp `cv2.erode`/`dilate`/`morphologyEx` (ellipse lớn có thể khác xấp xỉ OpenCV).

---

## Thuật toán hình thái nhị phân — `morph_algos.py` (Chương 4, mục 4)

Ảnh nhị phân `uint8` {0,255} hoặc `bool`.

| Hàm | Ý tưởng slide | Tham số chính |
|-----|----------------|---------------|
| `require_binary_image` | chuẩn hóa foreground | — |
| `hit_or_miss` | hit-miss transform | `se_fg`, `se_bg` |
| `boundary_extraction` | biên = \(A - (A \ominus B)\) | `se` |
| `region_filling`, `fill_holes` | lấp vùng / lỗ | `se`, seed |
| `connected_components` | nhãn CCL | `connectivity` 4 hoặc 8 |
| `convex_hull` | bọc lồi | — |
| `thinning`, `thickening` | mỏng/dày lặp hit-miss | `max_passes` |
| `skeleton` | Lantuejoul (mỏng + loại điểm cô lập) | `connectivity` |

```python
import numpy as np
from imageguard.ipkit import morph_algos

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
binary = (gray > 128).astype(np.uint8) * 255
labels, count = morph_algos.connected_components(binary, connectivity=8)
print(count)  # 1 (đã chạy trên gradient 16×16)
```

**Tests:** `tests/test_ipkit_morph_algos.py` — một số bước đối chiếu `cv2.connectedComponents` / morphology nhị phân.

---

## Phân đoạn — `segmentation.py` (Chương 5)

| Hàm | Chủ đề slide | Tham số chính |
|-----|--------------|---------------|
| `detect_points` | điểm (Laplacian + ngưỡng) | `T` |
| `detect_lines` | đường (mask hướng) | `direction`, `T` |
| `canny_edges` | Canny giản lược NumPy | `low`, `high`, `sigma` |
| `hough_lines` | Hough đường thẳng | `threshold`, `theta` bước |
| `iterative_global_threshold` | ngưỡng lặp | `T0`, `tol` |
| `region_growing` | mọc vùng | seed, ngưỡng |
| `kmeans_segmentation` | k-means mức xám/màu | `k`, `max_iter` |
| `simple_watershed` | watershed dạy học | `markers` |

Ngưỡng Otsu/adaptive cho phân đoạn cổ điển nằm ở `threshold.py` (C5.3); biên Sobel/Canny dùng `edges.py` / `canny_edges`.

```python
import numpy as np
from imageguard.ipkit.threshold import otsu_threshold

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
print(otsu_threshold(gray))  # 127 (đã chạy)
```

**Tests:** `tests/test_ipkit_segmentation.py` — `canny_edges` so IoU lỏng với `cv2.Canny` (không đòi khớp từng pixel).

**Chưa có / khác slide:** watershed hình thái đầy đủ (C5.5) — chỉ `simple_watershed` minh họa; không có graph cut / active contour.

---

## Đặc trưng cơ bản — `features_basic.py` (Chương 6, mục 2–4 phần moment/Haar/LBP)

| Hàm | Nội dung | Tham số |
|-----|----------|---------|
| `raw_moment`, `central_moment`, `normalized_central_moment` | moment bậc \((p,q)\) | `p`, `q` |
| `centroid` | trọng tâm | — |
| `hu_moments` | 7 bất biến Hu | — |
| `integral_image`, `rect_sum` | SAT | tọa độ rect |
| `haar_feature`, `haar_feature_map` | Haar-like | `kind`, rect |
| `lbp_image`, `uniform_pattern_lookup`, `lbph` | LBP / LBPH | `radius`, `neighbors` |
| `chi_square_distance` | khoảng cách \(\chi^2\) histogram | — |

```python
import numpy as np
from imageguard.ipkit import features_basic as fb

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
print(len(fb.hu_moments(gray)))  # 7 (đã chạy)
```

**Tests:** `tests/test_ipkit_features_basic.py` (oracle tay / công thức, không `cv2`).

---

## HOG — `features_hog.py` (Chương 6, mục 6)

| Hàm | Nội dung | Tham số |
|-----|----------|---------|
| `hog_feature_length` | độ dài vector Dalal–Triggs | `cell`, `block`, `bins`, kích thước ảnh |
| `hog_descriptor` | vector HOG | `cell`, `block`, `bins`, `norm` |
| `hog_visualize` | ảnh minh họa gradient | cùng tham số |

```python
import numpy as np
from imageguard.ipkit import features_hog

gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
vec = features_hog.hog_descriptor(gray, cell=8, block=2, bins=9)
print(len(vec))  # 36 trên ảnh 16×16 (đã chạy)
```

**Tests:** `tests/test_ipkit_features_hog.py` — so với `cv2.HOGDescriptor` khi cấu hình trùng (ảnh đủ lớn trong test).

---

## PCA / LDA — `features_projection.py` (Chương 6, mục 4)

| Hàm | Nội dung | Tham số |
|-----|----------|---------|
| `pca_fit` | SVD, trả mean/components/variance/ratio | `n_components` |
| `pca_transform`, `pca_inverse_transform` | chiếu / khôi phục | mean, components |
| `lda_fit` | Fisher LDA + ridge | `n_components` |
| `lda_transform` | chiếu LDA | tham số từ `lda_fit` |

```python
import numpy as np
from imageguard.ipkit.features_projection import pca_fit, pca_transform

X = np.random.default_rng(0).standard_normal((20, 5))
mean, comps, var, ratio = pca_fit(X, n_components=2)
Z = pca_transform(X, mean, comps)
assert Z.shape == (20, 2)
```

**Tests:** `tests/test_ipkit_features_projection.py` (không `cv2`).

**Chưa có trong ipkit (slide C6 mục 5):** **SIFT**, **SURF** — giải thích ý tưởng trên slide; triển khai pháp y dùng `cv2.SIFT` trong `forensics/keypoint/` (không phải ipkit).

---

## Phân lớp — `classifiers.py` (Chương 7)

| API | Nội dung slide |
|-----|----------------|
| `minimum_distance_classifier` | prototype / khoảng cách Euclid (§7.1) |
| `KNNClassifier` | k-NN |
| `LinearSVM`, `OneVsRestSVM` | SVM tuyến tính, soft margin `C` (§7.2) |
| `MLPClassifier` | back-prop MLP (§7.3) |
| `conv2d_valid`, `relu`, `max_pool2d`, `flatten` | khối CNN (§7.4) — **không** huấn luyện end-to-end |
| `softmax`, `accuracy_score`, `confusion_matrix` | tiện ích đánh giá |

```python
import numpy as np
from imageguard.ipkit.classifiers import minimum_distance_classifier

X = np.array([[0.0], [0.2], [9.0], [9.2]])
y = np.array([0, 0, 1, 1])
pred = minimum_distance_classifier(X, y, np.array([[0.1], [9.1]]))
assert pred.tolist() == [0, 1]
```

**Tests:** `tests/test_ipkit_classifiers.py`, `tests/test_ipkit_validation.py` (không `cv2`).

**Chưa có:** huấn luyện CNN sâu đầy đủ (batch, GPU, dataset ImageNet như slide demo); chỉ building blocks NumPy.

---

## Kiểm thử

Chạy từ thư mục gốc worktree (PowerShell):

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/ -k ipkit -q -p no:cacheprovider
```

Lần chạy ghi nhận khi cập nhật tài liệu này (venv dự án, `PYTHONPATH` trỏ `src` của repo clone): **263 passed, 1 skipped** trong **7,88 s**.

Tài liệu liên quan: [course-coverage.md](course-coverage.md), [testing-strategy.md](testing-strategy.md).
