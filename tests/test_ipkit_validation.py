"""ipkit.morphology — structuring elements must be exactly binary (0/1)."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.ipkit.morphology import close_, dilate, erode, gradient, open_, tophat

_GRAY = np.arange(25, dtype=np.uint8).reshape(5, 5)


@pytest.mark.parametrize(
    "se",
    [
        np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int64),
        np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[np.nan, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[np.inf, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.int8),
    ],
    ids=["negative", "fraction", "nan", "inf", "too-large", "mixed"],
)
def test_structuring_element_must_be_binary(se: np.ndarray) -> None:
    for op in (erode, dilate, open_, close_, gradient, tophat):
        with pytest.raises(ValueError, match="binary"):
            op(_GRAY, se)


def test_structuring_element_rejects_non_finite_message() -> None:
    se = np.ones((3, 3), dtype=np.float64)
    se[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        erode(_GRAY, se)


def test_structuring_element_accepts_bool_and_float_binary() -> None:
    cross = np.zeros((3, 3), dtype=bool)
    cross[1, :] = True
    cross[:, 1] = True
    as_float = cross.astype(np.float64)
    reference = erode(_GRAY, cross)
    np.testing.assert_array_equal(erode(_GRAY, as_float), reference)
    np.testing.assert_array_equal(dilate(_GRAY, cross), dilate(_GRAY, as_float))


def test_erode_rejects_non_binary_zero_one_equivalents() -> None:
    # A tiny positive epsilon is "almost 0" but would activate the cell.
    se = np.zeros((3, 3), dtype=np.float64)
    se[0, 0] = 1e-12
    with pytest.raises(ValueError, match="binary"):
        dilate(_GRAY, se)


def test_structuring_element_rejects_odd_shape_errors() -> None:
    with pytest.raises(ValueError, match="odd sizes"):
        erode(_GRAY, np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="odd sizes"):
        erode(_GRAY, np.ones((3, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="odd sizes"):
        erode(_GRAY, np.ones((0, 0), dtype=np.uint8))


def test_structuring_element_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric or boolean"):
        erode(_GRAY, np.array([["a", "b", "c"]], dtype=object))
