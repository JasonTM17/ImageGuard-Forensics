"""Course lock: Wiener restoration requires a kernel."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_wiener_without_a_kernel_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    params = {"kernel_role": "oracle", "nsr": 0.01}
    with pytest.raises(ValueError) as exc:
        restore(image, "wiener", params)
    assert str(exc.value) == "params must include a kernel"
