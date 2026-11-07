"""Course lock: unknown keypoint backends are rejected and ORB keeps its feature floor."""

from __future__ import annotations

import cv2
import pytest

from imageguard.core.config import ThresholdSet
from imageguard.forensics.keypoint.detectors import (
    OrbBackend,
    SiftBackend,
    create_backend,
)


def test_unknown_backend_names_sift_and_orb() -> None:
    cfg = ThresholdSet(
        raw={
            "keypoint": {
                "backend": {"value": "brisk", "source": "course"},
                "nfeatures": {"value": 0, "source": "course"},
            }
        }
    )
    with pytest.raises(ValueError, match=r"brisk.*sift\|orb"):
        create_backend(cfg)


def test_backend_norms_and_orb_feature_floor() -> None:
    assert SiftBackend.norm == cv2.NORM_L2
    assert OrbBackend.norm == cv2.NORM_HAMMING
    orb = OrbBackend(nfeatures=10)
    assert orb._det.getMaxFeatures() == 2000
