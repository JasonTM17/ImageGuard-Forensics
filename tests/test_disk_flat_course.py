"""Course lock: a zero-radius disk degradation leaves a constant field unchanged."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import degrade


def test_zero_radius_disk_degrade_keeps_the_field() -> None:
    image = np.full((3, 3), 9, np.uint8)
    output, meta = degrade(image, "disk", {"radius": 0}, None)
    np.testing.assert_array_equal(output, image)
    assert meta == {"kind": "disk", "radius": 0, "kernel": [[1.0]]}