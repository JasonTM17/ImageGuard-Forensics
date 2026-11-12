# Image Processing Toolkit (`ipkit`)

**Lưu ý quan trọng:** `imageguard.ipkit` chỉ phục vụ **xem/demo** và học tập trong môn Xử lý ảnh. Các toán tử **không** được gọi bởi detector pháp y, **không** ảnh hưởng `Verdict` hay pipeline forensics.

Mọi thuật toán lõi dùng **NumPy** (không `cv2` trong triển khai). OpenCV chỉ dùng trong pytest để đối chiếu.

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

## Ngưỡng (`threshold.py`)

| Hàm | Công thức / ý tưởng |
|-----|---------------------|
| `otsu_threshold(gray)` | Tối đa hóa \(\sigma_B^2(t)\) trên histogram 256-bin; trả về `int` ngưỡng. |
| `adaptive_mean_threshold(gray, block, c)` | Pixel \(>\) (trung bình cục bộ \(block×block\) \(- c\)) → 255, ngược lại 0. |

Trên một phần nhỏ pixel, `adaptive_mean_threshold` có thể khác `cv2.adaptiveThreshold` (làm tròn / tie-break ở biên cửa sổ) vì đường mean cục bộ dùng `float64` thay vì đường integer của OpenCV. Test `test_adaptive_mean_vs_cv2_interior` (seed cố định) so khớp với tolerance tỷ lệ mismatch \(\le 0{,}001\).

**Độ phức tạp (sau tối ưu):** `adaptive_mean_threshold` dùng ảnh tích phân (cumulative sum trên ảnh edge-padded) — thời gian \(O(HW)\), bộ nhớ \(O(HW)\). Erosion/dilation lặp theo từng ô active của phần tử cấu trúc \(k×k\) (tối đa \(k^2\) lần min/max trên view đã pad), không lặp từng pixel Python — \(O(HW \cdot |SE|)\) với \(|SE| \le k^2\).

```python
from imageguard.ipkit import otsu_threshold, adaptive_mean_threshold

t = otsu_threshold(gray)
binary = adaptive_mean_threshold(gray, block=11, c=4.0)
```

## Hình thái học (`morphology.py`)

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

## Kiểm thử

Chạy trong worktree (ví dụ):

```powershell
$env:PYTHONPATH='D:\ig-wt-k1\src'
& 'D:\ImageGuard Forensics\.venv\Scripts\python.exe' -m pytest tests/test_ipkit_*.py -q -p no:cacheprovider
```

Lần chạy gần nhất (worktree `ig-wt-k1`, chỉ `test_ipkit_*.py`): **18 passed** (khoảng 0,5s, pytest `-q -p no:cacheprovider`).
