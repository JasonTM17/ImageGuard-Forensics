"""Course lock: a negative disk radius is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import disk_kernel


def test_negative_disk_radius_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        disk_kernel(-1)
    assert str(exc.value) == "radius must be at least 0"
