"""Course lock: a zero Gaussian size is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_zero_gaussian_size_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        gaussian_kernel(0, 1.0)
    assert str(exc.value) == "ksize must be at least 1"
