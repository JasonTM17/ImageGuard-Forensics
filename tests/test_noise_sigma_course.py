"""Course lock: a flat field has a zero noise-sigma map."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.modelbench import noise_sigma_map


def test_flat_field_noise_sigma_is_zero() -> None:
    sigma = noise_sigma_map(np.full((32, 32), 40, np.uint8), k=5, bs=16)
    np.testing.assert_array_equal(sigma, np.zeros((2, 2), np.float32))
