"""Course lock: a zero Gaussian sigma is below the finite floor."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_zero_gaussian_sigma_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        gaussian_kernel(3, 0.0)
    assert str(exc.value) == "sigma must be at least 2.2250738585072014e-308"
