"""Course lock: a boolean disk radius is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import disk_kernel


def test_boolean_disk_radius_is_rejected() -> None:
    with pytest.raises(TypeError) as exc:
        disk_kernel(True)
    assert str(exc.value) == "radius must be an integer"
