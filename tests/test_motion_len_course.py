"""Course lock: a one-pixel motion kernel cannot be replaced."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import wrong_kernel


def test_unit_motion_kernel_cannot_be_replaced() -> None:
    with pytest.raises(ValueError) as exc:
        wrong_kernel("motion", {"length": 1, "angle_deg": 0.0})
    assert str(exc.value) == "length=1 cannot represent a different motion kernel"
