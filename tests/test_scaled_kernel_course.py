"""Course lock: scaled kernel stays odd and scaled block respects its floor."""

from __future__ import annotations

from imageguard.core.preprocess import scaled_block, scaled_kernel


def test_scaled_kernel_and_block_locked_values() -> None:
    assert scaled_kernel(15, 0.5) == 9
    assert scaled_kernel(5, 0.2) == 3
    assert scaled_kernel(9, 1.0) == 9
    assert scaled_block(16, 0.5) == 8
    assert scaled_block(8, 0.1) == 4
    assert scaled_block(32, 1.0) == 32
