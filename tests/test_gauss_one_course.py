"""Course lock: a 1x1 Gaussian cannot be replaced by a different kernel."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import wrong_kernel


def test_unit_gaussian_has_no_distinct_wrong_kernel() -> None:
    with pytest.raises(ValueError) as exc:
        wrong_kernel("gaussian", {"ksize": 1, "sigma": 1.0})
    assert str(exc.value) == "ksize=1 cannot represent a different Gaussian kernel"
