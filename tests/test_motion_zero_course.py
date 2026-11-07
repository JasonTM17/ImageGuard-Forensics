"""Course lock: a zero motion length is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import motion_kernel


def test_zero_motion_length_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        motion_kernel(0, 0.0)
    assert str(exc.value) == "length must be at least 1"
