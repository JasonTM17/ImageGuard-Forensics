"""Course lock: a wrong kernel needs a parameter dictionary."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import wrong_kernel


def test_wrong_kernel_rejects_a_non_dict() -> None:
    with pytest.raises(TypeError) as exc:
        wrong_kernel("disk", ["radius"])
    assert str(exc.value) == "params must be a dict so the wrong parameter is recorded"
