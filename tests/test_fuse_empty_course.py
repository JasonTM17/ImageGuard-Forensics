"""Course lock: fusing no masks returns nothing."""

from __future__ import annotations

from imageguard.forensics.localize.fusion_masks import fuse_masks


def test_empty_mask_list_fuses_to_none() -> None:
    assert fuse_masks([]) is None
