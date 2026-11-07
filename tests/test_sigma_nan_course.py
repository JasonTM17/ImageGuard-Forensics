"""Course lock: a non-finite Gaussian sigma is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_non_finite_gaussian_sigma_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        gaussian_kernel(3, float("nan"))
    assert str(exc.value) == "sigma must be finite"
