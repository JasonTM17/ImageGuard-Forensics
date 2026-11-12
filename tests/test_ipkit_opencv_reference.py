"""Cross-check the NumPy-only ``imageguard.ipkit`` operators against OpenCV.

Reference: ``docs/image-processing-toolkit.md``. That page states one numeric
tolerance — ``adaptive_mean_threshold`` must stay at a mismatch rate <= 0.001
against ``cv2.adaptiveThreshold`` on a fixed seed — and documents that this
operator may legitimately differ at window-mean tie-breaks, so equality is not
asserted there. For everything else the tightest honest tolerance is used:

* integer-valued gradient kernels and morphology: bit-exact, because both
  sides compute exact integer results (float32 / uint8 storage);
* general float kernels: float32 accumulation order differs, ``atol=1e-3``;
* Sobel magnitude: float32 square-root rounding, ``atol=1e-4`` (largest
  observed difference is 6.11e-05 on the seeds below);
* noise operators: OpenCV has no oracle for them, so the formulas documented
  in ``docs/image-processing-toolkit.md`` and seeded determinism are checked.

Border modes are stated in each test. ipkit ``replicate`` ==
``cv2.BORDER_REPLICATE``; ipkit ``zero`` == ``cv2.BORDER_CONSTANT`` with value
0. Morphology pads erosion with 255 and dilation with 0, which equals
OpenCV's default morphology border values for uint8. OpenCV's
``adaptiveThreshold`` has no ``borderType`` parameter and uses replicate
internally. All randomness is seeded; no files are written.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import pytest

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

BorderKind = Literal["replicate", "zero"]

_REPLICATE = cv2.BORDER_REPLICATE
_ZERO = cv2.BORDER_CONSTANT
_CV_BORDER: dict[BorderKind, int] = {"replicate": _REPLICATE, "zero": _ZERO}

_PREWITT_X = np.array(
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64
)
_PREWITT_Y = np.array(
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64
)


def _gray(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Seeded uint8 gray image."""
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _rgb(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Seeded uint8 BGR-free (RGB) image; ipkit is channel-agnostic."""
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def _cv_sobel(gray: np.ndarray, dx: int, dy: int, border: int) -> np.ndarray:
    """OpenCV Sobel with the same 3x3 kernel used by ipkit."""
    return cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=3, borderType=border)


def _box_mean_float64(gray: np.ndarray, block: int) -> np.ndarray:
    """float64 local mean with replicate border, as OpenCV computes it."""
    return cv2.boxFilter(
        gray.astype(np.float64),
        -1,
        (block, block),
        normalize=True,
        borderType=_REPLICATE,
    )


# ---------------------------------------------------------------------------
# edges.py
# ---------------------------------------------------------------------------


def test_sobel_matches_cv2_replicate_border() -> None:
    """Sobel Gx/Gy == cv2.Sobel(ksize=3); border replicate on both sides."""
    gray = _gray((64, 64), 20261112)
    gx, gy = sobel(gray, "replicate")
    ref_gx = _cv_sobel(gray, 1, 0, _CV_BORDER["replicate"])
    ref_gy = _cv_sobel(gray, 0, 1, _CV_BORDER["replicate"])

    assert gx.shape == gray.shape
    assert gx.dtype == np.float32
    np.testing.assert_array_equal(gx, ref_gx)
    np.testing.assert_array_equal(gy, ref_gy)


def test_sobel_matches_cv2_zero_border_odd_shape() -> None:
    """Sobel on 37x53 with border zero == BORDER_CONSTANT (value 0)."""
    gray = _gray((37, 53), 5)
    gx, gy = sobel(gray, "zero")
    ref_gx = _cv_sobel(gray, 1, 0, _CV_BORDER["zero"])
    ref_gy = _cv_sobel(gray, 0, 1, _CV_BORDER["zero"])

    np.testing.assert_array_equal(gx, ref_gx)
    np.testing.assert_array_equal(gy, ref_gy)


def test_sobel_rgb_matches_cv2_per_channel() -> None:
    """RGB Sobel applies the gray path per channel; border replicate."""
    rgb = _rgb((64, 64), 4)
    gx, gy = sobel(rgb, "replicate")
    assert gx.shape == rgb.shape

    for channel in range(3):
        ref_gx = _cv_sobel(rgb[:, :, channel], 1, 0, _CV_BORDER["replicate"])
        ref_gy = _cv_sobel(rgb[:, :, channel], 0, 1, _CV_BORDER["replicate"])
        np.testing.assert_array_equal(gx[:, :, channel], ref_gx)
        np.testing.assert_array_equal(gy[:, :, channel], ref_gy)


@pytest.mark.parametrize("padding", ["replicate", "zero"])
def test_laplacian_matches_cv2_4_neighbour_kernel(padding: BorderKind) -> None:
    """Laplacian [0,1,0;1,-4,1;0,1,0] == cv2.Laplacian(ksize=1)."""
    gray = _gray((65, 49), 11)
    ours = laplacian(gray, padding)
    ref = cv2.Laplacian(gray, cv2.CV_32F, ksize=1, borderType=_CV_BORDER[padding])

    assert ours.dtype == np.float32
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("padding", ["replicate", "zero"])
def test_prewitt_matches_cv2_filter2d(padding: BorderKind) -> None:
    """Prewitt kernels via cv2.filter2D (correlation); border per parameter."""
    gray = _gray((37, 53), 3)
    gx, gy = prewitt(gray, padding)
    ref_gx = cv2.filter2D(
        gray, cv2.CV_32F, _PREWITT_X, borderType=_CV_BORDER[padding]
    )
    ref_gy = cv2.filter2D(
        gray, cv2.CV_32F, _PREWITT_Y, borderType=_CV_BORDER[padding]
    )

    np.testing.assert_array_equal(gx, ref_gx)
    np.testing.assert_array_equal(gy, ref_gy)


@pytest.mark.parametrize("padding", ["replicate", "zero"])
def test_convolve2d_matches_filter2d_for_smoothing_kernel(padding: BorderKind) -> None:
    """5x5 binomial (Gaussian-like) kernel on 64x64; ipkit has no smoothing op.

    The weights sum to 256 and are all dyadic, so both sides are exact even
    though the kernel is float64.
    """
    gray = _gray((64, 64), 0)
    kernel = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]).astype(np.float64)
    kernel /= kernel.sum()

    ours = convolve2d(gray, kernel, padding)
    ref = cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=_CV_BORDER[padding])

    np.testing.assert_array_equal(ours, ref)


def test_convolve2d_matches_filter2d_for_float_kernel_within_tolerance() -> None:
    """7x7 random float kernel; only float32 accumulation order can differ.

    border replicate; measured largest difference for this seed is 3.05e-04.
    """
    rng = np.random.default_rng(1)
    kernel = rng.uniform(-1.0, 1.0, (7, 7)).astype(np.float64)
    gray = _gray((64, 64), 3)

    ours = convolve2d(gray, kernel, "replicate")
    ref = cv2.filter2D(
        gray, cv2.CV_32F, kernel, borderType=_CV_BORDER["replicate"]
    )

    np.testing.assert_allclose(ours, ref, atol=1e-3, rtol=1e-3)


def test_gradient_magnitude_matches_cv2_magnitude() -> None:
    """sqrt(gx^2+gy^2) vs cv2.magnitude; border replicate.

    Difference is float32 square-root rounding only; measured max 6.11e-05.
    """
    gray = _gray((64, 64), 0)
    ours = gradient_magnitude(gray, "replicate")

    gx = _cv_sobel(gray, 1, 0, _CV_BORDER["replicate"])
    gy = _cv_sobel(gray, 0, 1, _CV_BORDER["replicate"])
    ref = cv2.magnitude(gx, gy)

    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, ref, atol=1e-4, rtol=1e-6)


def test_sobel_border_modes_differ_only_on_the_border() -> None:
    """Zero vs replicate padding must agree on the interior (3x3 footprint)."""
    gray = _gray((37, 53), 9)
    gx_rep, _ = sobel(gray, "replicate")
    gx_zero, _ = sobel(gray, "zero")

    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_array_equal(gx_rep[interior], gx_zero[interior])
    assert not np.array_equal(gx_rep, gx_zero)


# ---------------------------------------------------------------------------
# threshold.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "seed"),
    [
        ((64, 64), 0),
        ((64, 64), 7),
        ((64, 64), 20261112),
        ((37, 53), 1),
        ((37, 53), 42),
        ((65, 49), 5),
    ],
)
def test_otsu_matches_cv2_random(shape: tuple[int, int], seed: int) -> None:
    """Otsu threshold (full image, no border) == cv2 THRESH_OTSU."""
    gray = _gray(shape, seed)
    ours = otsu_threshold(gray)
    ref = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    assert isinstance(ours, int)
    assert 0 <= ours <= 255
    assert ours == ref


@pytest.mark.parametrize(
    ("label", "gray"),
    [
        (
            "flat_128",
            np.full((16, 16), 128, dtype=np.uint8),
        ),
        (
            "flat_255",
            np.full((16, 16), 255, dtype=np.uint8),
        ),
        (
            "all_zero",
            np.zeros((16, 16), dtype=np.uint8),
        ),
        (
            "single_pixel",
            np.array([[42]], dtype=np.uint8),
        ),
        (
            "two_level",
            np.repeat(np.array([[0, 200]], dtype=np.uint8), 16, axis=0),
        ),
        (
            "all_256_levels",
            np.arange(256, dtype=np.uint8)
            .repeat(16)
            .reshape(16, 256),
        ),
        (
            "one_white_pixel",
            np.pad(
                np.array([[255]], dtype=np.uint8),
                2,
                mode="constant",
                constant_values=0,
            ),
        ),
    ],
)
def test_otsu_matches_cv2_degenerate_histograms(label: str, gray: np.ndarray) -> None:
    """Degenerate histograms (flat, one pixel, two levels); no border involved."""
    ours = otsu_threshold(gray)
    ref = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    assert ours == ref, label


def test_adaptive_mean_matches_cv2_within_documented_tolerance() -> None:
    """docs/image-processing-toolkit.md: mismatch rate <= 0.001, seed fixed.

    Configuration mirrors the documented reference check (64x64, block 11,
    c=4.0). ipkit uses edge padding; cv2.adaptiveThreshold has no border
    parameter and uses replicate internally, so the borders are comparable.
    Differences stem from the float64 local mean vs OpenCV's integer path and
    are allowed by the doc, so equality is deliberately not asserted.
    """
    gray = _gray((64, 64), 27)
    ours = adaptive_mean_threshold(gray, block=11, c=4.0)
    ref = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        4.0,
    )

    assert ours.shape == gray.shape
    assert ours.dtype == np.uint8
    assert set(np.unique(ours)) <= {0, 255}
    mismatch_rate = float(np.count_nonzero(ours != ref)) / gray.size
    assert mismatch_rate <= 0.001


@pytest.mark.parametrize(("block", "c"), [(11, 4.0), (7, 1.0), (15, 2.5)])
def test_adaptive_mean_differences_are_confined_to_window_ties(
    block: int, c: float
) -> None:
    """Documented tie behavior: only near-threshold pixels may disagree.

    Border replicate on both sides. A pixel whose intensity is more than 0.5
    away from the float64 window mean minus c cannot be flipped by OpenCV's
    integer rounding, so those pixels must match exactly. This replaces an
    equality assertion the doc explicitly warns against.
    """
    gray = _gray((37, 53), 3)
    ours = adaptive_mean_threshold(gray, block=block, c=c)
    ref = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )

    threshold = _box_mean_float64(gray, block) - c
    margin = np.abs(gray.astype(np.float64) - threshold)
    differing = ours != ref
    outside_band = margin > 0.5 + 1e-9

    assert set(np.unique(ours)) <= {0, 255}
    assert np.all(margin[differing] <= 0.5 + 1e-9)
    np.testing.assert_array_equal(ours[outside_band], ref[outside_band])


# ---------------------------------------------------------------------------
# morphology.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rect", "cross"])
@pytest.mark.parametrize("ksize", [3, 5])
def test_structuring_element_matches_cv2_rect_and_cross(
    kind: Literal["rect", "cross"], ksize: int
) -> None:
    """OpenCV documents these two shapes exactly; no border involved."""
    ours = structuring_element(kind, ksize)
    cv_shape = cv2.MORPH_RECT if kind == "rect" else cv2.MORPH_CROSS
    ref = cv2.getStructuringElement(cv_shape, (ksize, ksize))

    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_structuring_element_ellipse_is_a_symmetric_disk(ksize: int) -> None:
    """Ellipse SE is binary, odd-sized and 4-fold symmetric.

    cv2.MORPH_ELLIPSE uses a different polygonal approximation for ksize >= 5,
    so operator tests below pass the *same* explicit SE to both implementations
    instead of relying on OpenCV's shape.
    """
    se = structuring_element("ellipse", ksize)

    assert se.shape == (ksize, ksize)
    assert se.dtype == np.uint8
    assert set(np.unique(se)) <= {0, 1}
    np.testing.assert_array_equal(se, se[::-1, :])
    np.testing.assert_array_equal(se, se[:, ::-1])
    assert se[ksize // 2, :].tolist() == [1] * ksize
    assert se[:, ksize // 2].tolist() == [1] * ksize


@pytest.mark.parametrize("kind", ["rect", "cross"])
def test_erode_dilate_match_cv2_default_border(kind: Literal["rect", "cross"]) -> None:
    """Grayscale min/max filters on 64x64.

    Border handling: ipkit pads erosion with 255 and dilation with 0, equal to
    OpenCV's default morphology border values for uint8 (BORDER_CONSTANT with
    the neutral element), so exact equality is expected.
    """
    gray = _gray((64, 64), 7)
    se = structuring_element(kind, 3)

    np.testing.assert_array_equal(erode(gray, se), cv2.erode(gray, se))
    np.testing.assert_array_equal(dilate(gray, se), cv2.dilate(gray, se))


@pytest.mark.parametrize("kind", ["rect", "cross", "ellipse"])
def test_open_close_match_cv2_on_odd_shape(
    kind: Literal["rect", "cross", "ellipse"]
) -> None:
    """Opening/closing on 37x53 with the same 5x5 SE and default border."""
    gray = _gray((37, 53), 0)
    se = structuring_element(kind, 5)

    np.testing.assert_array_equal(
        open_(gray, se), cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
    )
    np.testing.assert_array_equal(
        close_(gray, se), cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    )


def test_gradient_and_tophat_match_cv2() -> None:
    """Morphological gradient and white top-hat on 65x49, default border."""
    gray = _gray((65, 49), 2)
    se = structuring_element("cross", 5)

    np.testing.assert_array_equal(
        gradient(gray, se), cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se)
    )
    np.testing.assert_array_equal(
        tophat(gray, se), cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    )


def test_morphology_matches_cv2_when_se_is_larger_than_image() -> None:
    """4x7 image with a 5x5 SE; default border must still agree."""
    gray = _gray((4, 7), 1)
    se = structuring_element("rect", 5)

    np.testing.assert_array_equal(erode(gray, se), cv2.erode(gray, se))
    np.testing.assert_array_equal(dilate(gray, se), cv2.dilate(gray, se))
    np.testing.assert_array_equal(
        open_(gray, se), cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
    )


@pytest.mark.parametrize("value", [0, 255])
def test_morphology_matches_cv2_on_constant_image(value: int) -> None:
    """Flat images: min/max under the footprint is a no-op on both sides."""
    gray = np.full((17, 23), value, dtype=np.uint8)
    se = structuring_element("ellipse", 5)

    np.testing.assert_array_equal(erode(gray, se), cv2.erode(gray, se))
    np.testing.assert_array_equal(dilate(gray, se), cv2.dilate(gray, se))
    np.testing.assert_array_equal(
        gradient(gray, se), cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se)
    )


def test_morphology_rgb_matches_cv2_per_channel() -> None:
    """Erosion/dilation on HxWx3; OpenCV also processes channels independently."""
    rgb = _rgb((64, 64), 6)
    se = structuring_element("cross", 3)

    np.testing.assert_array_equal(erode(rgb, se), cv2.erode(rgb, se))
    np.testing.assert_array_equal(dilate(rgb, se), cv2.dilate(rgb, se))


# ---------------------------------------------------------------------------
# noise.py — no OpenCV oracle; documented formulas, ranges, seeded determinism
# ---------------------------------------------------------------------------


def test_salt_pepper_rate_and_polarity() -> None:
    """64x64 = 4096 px at amount 0.05 -> exactly 205 flipped pixels (0 or 255)."""
    gray = np.full((64, 64), 120, dtype=np.uint8)
    noisy = salt_pepper(gray, amount=0.05, seed=7)
    changed = noisy != gray

    assert noisy.shape == gray.shape
    assert noisy.dtype == np.uint8
    assert np.count_nonzero(changed) == round(0.05 * gray.size) == 205
    assert set(np.unique(noisy[changed])) <= {0, 255}
    salts = int(np.count_nonzero(noisy[changed] == 255))
    peppers = int(np.count_nonzero(noisy[changed] == 0))
    assert salts + peppers == 205
    # Half salt / half pepper up to sampling noise (106/99 measured).
    assert abs(salts - peppers) <= 40


def test_speckle_matches_documented_formula() -> None:
    """I' = clip(round(I * (1 + sigma * N(0,1)))) with the same seeded RNG."""
    gray = _gray((64, 64), 0)
    noisy = speckle(gray, sigma=0.1, seed=3)

    rng = np.random.default_rng(3)
    noise = 1.0 + 0.1 * rng.standard_normal(gray.shape)
    expected = np.clip(np.rint(gray.astype(np.float64) * noise), 0, 255)
    expected = expected.astype(np.uint8)

    np.testing.assert_array_equal(noisy, expected)
    np.testing.assert_array_equal(speckle(gray, sigma=0.0, seed=3), gray)


def test_poisson_mean_tracks_intensity() -> None:
    """Poisson(level) over 4096 px: means 99.97 (100) and 200.03 (200) measured."""
    for level in (100, 200):
        gray = np.full((64, 64), level, dtype=np.uint8)
        noisy = poisson(gray, seed=5)

        assert noisy.shape == gray.shape
        assert noisy.dtype == np.uint8
        assert abs(float(noisy.mean()) - level) < 1.0


def test_noise_operators_are_seeded_and_deterministic() -> None:
    """Same seed -> identical output; a different seed changes the sample."""
    gray = _gray((32, 32), 4)

    for op in (
        lambda seed: salt_pepper(gray, 0.1, seed),
        lambda seed: speckle(gray, 0.2, seed),
        lambda seed: poisson(gray, seed),
    ):
        np.testing.assert_array_equal(op(11), op(11))
        assert not np.array_equal(op(11), op(12))
