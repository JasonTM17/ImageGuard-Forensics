"""Course lock: an unknown restoration kind is rejected."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.enhance_ops import restore


def test_unknown_restoration_kind_is_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        restore(np.zeros((4, 4), np.uint8), "magic", {})
    assert str(exc.value) == "unsupported restoration kind: magic"
