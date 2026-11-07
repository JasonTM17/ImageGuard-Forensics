"""Course lock: upscale_mask is identity at scale>=1 and nearest-neighbor otherwise."""

from __future__ import annotations

import numpy as np

from imageguard.forensics.keypoint.mask_builder import upscale_mask


def test_upscale_mask_identity_and_nearest_quarter() -> None:
    mask = np.zeros((2, 2), np.uint8)
    mask[0, 0] = 255
    same = upscale_mask(mask, 1.0, (9, 9))
    assert same is mask
    out = upscale_mask(mask, 0.5, (4, 4))
    assert out.dtype == np.uint8
    assert out.shape == (4, 4)
    assert int(out.sum()) == 1020
    assert out.tolist() == [
        [255, 255, 0, 0],
        [255, 255, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
