"""Course lock: an AWGN seed must match the integer generator seed."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_awgn_seed_must_match_the_generator() -> None:
    image = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError) as exc:
        degrade(image, "awgn", {"sigma": 0.0, "seed": 2}, 1)
    assert str(exc.value) == "params seed must match the integer rng seed"
