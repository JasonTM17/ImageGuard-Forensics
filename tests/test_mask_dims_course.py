"""Course lock: pixel metrics reject a prediction whose shape differs from the ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.evaluation.metrics import pixel_metrics


def test_pixel_metrics_rejects_a_shape_mismatch() -> None:
    gt = np.zeros((4, 4), np.uint8)
    gt[0, 0] = 255
    with pytest.raises(ValueError) as exc:
        pixel_metrics(np.zeros((4, 5), np.uint8), gt)
    assert str(exc.value) == (
        "mask dims (4, 5) != gt (4, 4) — reject, không silent resize"
    )
