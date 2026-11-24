"""Thư viện toán tử Xử lý ảnh số thuần NumPy (`imageguard.ipkit`)."""

# Tên module (color, geometry, features_harris, ...) được nạp lười qua
# ``__getattr__``. Detector chỉ ``from imageguard.ipkit import features_basic``
# thì không kéo các module giảng dạy mới vào đồ thị import.
import importlib

_SUBMODULES = frozenset(
    {
        "classifiers",
        "color",
        "edges",
        "features_basic",
        "features_harris",
        "features_hog",
        "features_projection",
        "frequency",
        "geometry",
        "histogram",
        "intensity",
        "morph_algos",
        "morphology",
        "noise",
        "segmentation",
        "spatial",
        "threshold",
    }
)


def __getattr__(name: str):
    """Nạp một submodule ipkit đúng lúc người gọi xin tên đó."""
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


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
    "color",
    "convolve2d",
    "dilate",
    "edges",
    "erode",
    "features_basic",
    "features_harris",
    "features_hog",
    "features_projection",
    "frequency",
    "geometry",
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
