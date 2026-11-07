"""Course lock: a non-finite motion angle is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import motion_kernel


def test_non_finite_motion_angle_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        motion_kernel(3, float("nan"))
    assert str(exc.value) == "angle_deg must be finite"
