"""Thư viện toán tử Xử lý ảnh số thuần NumPy (`imageguard.ipkit`)."""

# Cấu trúc mô-đun theo từng nhóm kỹ thuật xử lý ảnh cốt lõi:
#   intensity / histogram / spatial (không gian), frequency (miền tần số),
#   morphology + morph_algos (hình thái học), edges / threshold / segmentation,
#   features_* (đặc trưng & giảm chiều), classifiers (phân lớp mẫu).
# Xuất theo tên module (tránh xung đột tên hàm dùng chung như `laplacian`).
from imageguard.ipkit import (
    classifiers,
    edges,
    features_basic,
    features_hog,
    features_projection,
    frequency,
    histogram,
    intensity,
    morph_algos,
    morphology,
    noise,
    segmentation,
    spatial,
    threshold,
)
from imageguard.ipkit.edges import (
    convolve2d,
    gradient_magnitude,
    laplacian,
    prewitt,
    sobel,
)
from imageguard.ipkit.morphology import (
    close_,
    dilate,
    erode,
    gradient,
    open_,
    structuring_element,
    tophat,
)
from imageguard.ipkit.noise import poisson, salt_pepper, speckle
from imageguard.ipkit.threshold import adaptive_mean_threshold, otsu_threshold

__all__ = [
    "adaptive_mean_threshold",
    "classifiers",
    "close_",
    "convolve2d",
    "dilate",
    "edges",
    "erode",
    "features_basic",
    "features_hog",
    "features_projection",
    "frequency",
    "gradient",
    "gradient_magnitude",
    "histogram",
    "intensity",
    "laplacian",
    "morph_algos",
    "morphology",
    "noise",
    "open_",
    "otsu_threshold",
    "poisson",
    "prewitt",
    "salt_pepper",
    "segmentation",
    "sobel",
    "spatial",
    "speckle",
    "structuring_element",
    "threshold",
    "tophat",
]
