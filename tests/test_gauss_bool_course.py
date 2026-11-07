"""Course lock: a boolean Gaussian size is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import gaussian_kernel


def test_boolean_gaussian_size_is_rejected() -> None:
    with pytest.raises(TypeError) as exc:
        gaussian_kernel(True, 1.0)
    assert str(exc.value) == "ksize must be an integer"
