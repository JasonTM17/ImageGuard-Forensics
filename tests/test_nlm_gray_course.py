"""Course lock: NLM on a single-channel image is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_nlm_on_gray_image_is_rejected() -> None:
    image = np.zeros((8, 8), np.uint8)
    params = {
        "h": 1.0,
        "hColor": 1.0,
        "templateWindowSize": 3,
        "searchWindowSize": 5,
    }
    with pytest.raises(ValueError) as exc:
        restore(image, "nlm", params)
    assert str(exc.value) == "nlm expects a three-channel BGR image"
