"""Course lock: an even Gaussian kernel size is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_even_gaussian_ksize_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        gaussian_kernel(4, 1.0)
    assert str(exc.value) == "ksize must be odd"
