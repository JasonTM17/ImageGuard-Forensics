"""ipkit.morphology — structuring elements and ops vs OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.morphology import (
    close_,
    dilate,
    erode,
    gradient,
    open_,
    structuring_element,
    tophat,
)


def test_structuring_element_matches_cv2_rect() -> None:
    k = 5
    ours = structuring_element("rect", k)
    ref = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    np.testing.assert_array_equal(ours, ref)


def test_erode_dilate_vs_cv2_random() -> None:
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    se = structuring_element("cross", 3)
    e = erode(img, se)
    d = dilate(img, se)
    cv_e = cv2.erode(img, se)
    cv_d = cv2.dilate(img, se)
    np.testing.assert_array_equal(e, cv_e)
    np.testing.assert_array_equal(d, cv_d)


def test_open_close_gradient_tophat_vs_cv2() -> None:
    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    se = structuring_element("rect", 3)
    assert np.array_equal(open_(img, se), cv2.morphologyEx(img, cv2.MORPH_OPEN, se))
    assert np.array_equal(close_(img, se), cv2.morphologyEx(img, cv2.MORPH_CLOSE, se))
    assert np.array_equal(
        gradient(img, se), cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se),
    )
    assert np.array_equal(tophat(img, se), cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se))


def test_erode_dilate_large_image_vs_cv2() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    se = structuring_element("rect", 5)
    np.testing.assert_array_equal(erode(img, se), cv2.erode(img, se))
    np.testing.assert_array_equal(dilate(img, se), cv2.dilate(img, se))


def test_morphology_rejects_even_se() -> None:
    with pytest.raises(ValueError, match="odd"):
        structuring_element("rect", 4)
