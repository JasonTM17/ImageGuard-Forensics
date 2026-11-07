"""Course lock: a text AWGN generator is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_text_awgn_generator_is_rejected() -> None:
    image = np.zeros((4, 4), np.uint8)
    with pytest.raises(TypeError) as exc:
        degrade(image, "awgn", {"sigma": 1.0}, "nope")
    assert str(exc.value) == "rng must be an integer seed or numpy.random.Generator"
