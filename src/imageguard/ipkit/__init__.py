"""Image processing toolkit (demo/view only — not used by forensic detectors)."""

from imageguard.ipkit import edges, morphology, noise, threshold
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
    "close_",
    "convolve2d",
    "dilate",
    "edges",
    "erode",
    "gradient",
    "gradient_magnitude",
    "laplacian",
    "morphology",
    "noise",
    "open_",
    "otsu_threshold",
    "poisson",
    "prewitt",
    "salt_pepper",
    "sobel",
    "speckle",
    "structuring_element",
    "threshold",
    "tophat",
]
