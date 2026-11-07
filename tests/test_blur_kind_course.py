"""Course lock: an unknown blur kind is rejected."""

from __future__ import annotations

import pytest

from imageguard.evaluation.enhance_ops import wrong_kernel


def test_unknown_blur_kind_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        wrong_kernel("box", {"ksize": 3})
    assert str(exc.value) == "unsupported blur kind: box"
