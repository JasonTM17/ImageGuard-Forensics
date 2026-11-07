"""Course lock: a negative unsharp amount is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_negative_unsharp_amount_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"ksize": 3, "sigma": 1.0, "amount": -0.1}
    with pytest.raises(ValueError) as exc:
        restore(image, "unsharp", params)
    assert str(exc.value) == "amount must be at least 0.0"
