"""Course lock: zero Richardson-Lucy iterations are rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_zero_richardson_lucy_iterations_are_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "oracle", "iterations": 0, "kernel": [[1.0]]}
    with pytest.raises(ValueError) as exc:
        restore(image, "richardson_lucy", params)
    assert str(exc.value) == "iterations must be at least 1"
