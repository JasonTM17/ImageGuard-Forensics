"""Course lock: an even NLM window is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_even_nlm_window_is_rejected() -> None:
    image = np.zeros((8, 8, 3), np.uint8)
    params = {
        "h": 1.0,
        "hColor": 1.0,
        "templateWindowSize": 4,
        "searchWindowSize": 5,
    }
    with pytest.raises(ValueError) as exc:
        restore(image, "nlm", params)
    assert str(exc.value) == "NLM window sizes must be odd"
