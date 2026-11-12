"""ipkit.morph_algos — binary morphological algorithms (Ch.4) vs oracles."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.morph_algos import (
    boundary_extraction,
    connected_components,
    convex_hull,
    fill_holes,
    hit_or_miss,
    region_filling,
    require_binary_image,
    skeleton,
    thickening,
    thinning,
)
from imageguard.ipkit.morphology import structuring_element


def _ascii_bool(art: str) -> np.ndarray:
    """Chuyển ASCII ('1'=foreground, '.'=nền) thành mask bool."""
    lines = [ln for ln in art.strip().splitlines() if ln.strip()]
    rows = [[ch == "1" for ch in line.strip()] for line in lines]
    return np.array(rows, dtype=bool)


def _to_cv8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def test_require_binary_accepts_bool_and_uint8() -> None:
    b = _ascii_bool("11\n00")
    assert require_binary_image(b).dtype == bool
    u = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert np.array_equal(require_binary_image(u), _to_cv8(u).astype(bool))


def test_require_binary_rejects_bad() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        require_binary_image(np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="only 0 and 255"):
        require_binary_image(np.array([[1, 2]], dtype=np.uint8))
    with pytest.raises(TypeError):
        require_binary_image(np.array([[1.0, 0.0]]))


def test_boundary_of_square_is_one_pixel_ring() -> None:
    sq = _ascii_bool(
        """
        .....
        .111.
        .111.
        .111.
        .....
        """
    )
    se = structuring_element("rect", 3)
    border = boundary_extraction(sq, se)
    assert np.all(border <= sq)
    assert int(border.sum()) == 8


def test_fill_holes_on_ring() -> None:
    ring = _ascii_bool(
        """
        1111111
        1000001
        1000001
        1000001
        1000001
        1000001
        1111111
        """
    )
    se = structuring_element("rect", 3)
    filled = fill_holes(_to_cv8(ring), se)
    assert filled[3, 3]
    assert np.all(filled[2:5, 2:5])


def test_connected_components_two_blobs_four_vs_eight() -> None:
    diag = _ascii_bool(
        """
        1..
        .1.
        ...
        """
    )
    _lbl4, n4 = connected_components(diag, 4)
    lbl8, n8 = connected_components(diag, 8)
    assert n4 == 2
    assert n8 == 1
    u8 = _to_cv8(diag)
    cv_n, cv_lbl = cv2.connectedComponents(u8, connectivity=8)
    assert n8 == cv_n - 1  # cv2 counts background
    assert np.array_equal(
        (lbl8 > 0).astype(np.uint8) * 255,
        (cv_lbl > 0).astype(np.uint8) * 255,
    )


def test_connected_components_matches_cv2_four() -> None:
    art = _ascii_bool(
        """
        11...
        11...
        ..11.
        ..11.
        """
    )
    lbl, n = connected_components(art, 4)
    cv_n, _cv_lbl = cv2.connectedComponents(_to_cv8(art), connectivity=4)
    assert n == cv_n - 1
    assert set(np.unique(lbl[lbl > 0])) == set(range(1, n + 1))


def test_skeleton_of_thick_line_inside_line() -> None:
    line = _ascii_bool(
        """
        .....
        11111
        11111
        11111
        .....
        """
    )
    se = structuring_element("rect", 3)
    sk = skeleton(line, se)
    assert np.all(sk <= line)
    assert sk.any()


def test_hit_or_miss_finds_isolated_pixel() -> None:
    fg = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    bg = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.uint8)
    img = _ascii_bool(
        """
        0.0.0
        .0.0.
        0.0.0
        """
    ).astype(np.uint8)
    img_u8 = img.astype(np.uint8) * 255
    img_u8[1, 2] = 255
    hits = hit_or_miss(img_u8, fg, bg)
    assert hits[1, 2]
    assert int(hits.sum()) == 1


def test_thinning_idempotent_on_thin_line() -> None:
    thin = _ascii_bool("...\n111\n...")
    once = thinning(thin)
    twice = thinning(once)
    np.testing.assert_array_equal(once, twice)


def test_region_filling_with_explicit_seed() -> None:
    frame = _ascii_bool(
        """
        111
        101
        111
        """
    )
    seed = np.zeros_like(frame)
    seed[1, 1] = True
    filled = region_filling(frame, structuring_element("rect", 3), seed=seed)
    assert filled[1, 1]


def test_convex_hull_stays_in_bbox() -> None:
    mask = _ascii_bool(
        """
        ...........
        ...1111....
        ....11.....
        ...1111....
        ....11.....
        ...........
        """
    )
    hull = convex_hull(mask)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    assert hull[:y0, :].sum() == 0
    assert hull[y1 + 1 :, :].sum() == 0
    assert hull[:, :x0].sum() == 0
    assert hull[:, x1 + 1 :].sum() == 0
    assert hull.sum() >= mask.sum()


def test_thickening_expands_set() -> None:
    line = _ascii_bool(
        """
        .....
        ..1..
        ..1..
        ..1..
        .....
        """
    )
    thick = thickening(line)
    assert thick.sum() >= line.sum()


def test_skeleton_subsets_option() -> None:
    line = _ascii_bool(".111.")
    sk, subs = skeleton(line, structuring_element("rect", 3), return_subsets=True)
    assert isinstance(subs, list)
    assert len(subs) >= 1
    assert np.array_equal(sk, np.logical_or.reduce(subs) if subs else sk)


def test_boundary_hand_oracle_vs_morphology_gradient_differs() -> None:
    """β(A)=A−erosion; cv2 MORPH_GRADIENT là dilate−erode — chỉ so sánh công thức beta."""
    sq = _to_cv8(_ascii_bool(".111.\n.111.\n.111.\n.111.\n.111."))
    se = structuring_element("rect", 3)
    beta = boundary_extraction(sq, se)
    from imageguard.ipkit.morph_algos import _binary_erode

    body = require_binary_image(sq)
    manual = body & ~_binary_erode(body, se)
    np.testing.assert_array_equal(beta, manual)


def test_fill_holes_close_not_equivalent() -> None:
    """Lấp lỗ slide không nhất thiết bằng closing OpenCV trên mọi hình."""
    ring = _ascii_bool(
        """
        11111
        10001
        10001
        10001
        11111
        """
    )
    se = structuring_element("rect", 3)
    filled = fill_holes(ring, se)
    _ = cv2.morphologyEx(_to_cv8(ring), cv2.MORPH_CLOSE, se)
    assert np.all(filled[1:4, 1:4])


def test_connected_components_constant_foreground() -> None:
    const = np.ones((3, 3), dtype=np.uint8) * 255
    _, n = connected_components(const, 4)
    assert n == 1


def test_connected_components_background_only() -> None:
    blank = np.zeros((2, 2), dtype=np.uint8)
    lbl, n = connected_components(blank, 8)
    assert n == 0
    assert np.all(lbl == 0)


def test_hit_or_miss_rejects_mismatched_se_shapes() -> None:
    fg = np.ones((3, 3), dtype=np.uint8)
    bg = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        hit_or_miss(np.zeros((7, 7), dtype=np.uint8), fg, bg)


def test_thinning_max_passes_validation() -> None:
    img = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        thinning(img, max_passes=0)
