"""Course lock: an unknown degradation kind is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import degrade


def test_unknown_degradation_kind_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        degrade(np.zeros((4, 4), np.uint8), "magic", {}, None)
    assert str(exc.value) == "unsupported degradation kind: magic"
