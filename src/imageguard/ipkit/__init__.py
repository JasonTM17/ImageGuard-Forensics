"""Image processing toolkit (demo/view only — not used by forensic detectors)."""

# Các module theo chương của môn học (mỗi module là một chủ đề thầy dạy):
#   C2 intensity/histogram/spatial, C3 frequency, C4 morphology + morph_algos,
#   C5 edges/threshold/segmentation, C6 features_*, C7 classifiers.
# Chỉ xuất tên module (không tách hàm) để tránh trùng tên, ví dụ `laplacian`.
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
