"""Course lock: NoiseMap returns inconclusive when cancel is already set."""

from __future__ import annotations

import threading

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.noise_map import NoiseMap


def test_noise_map_pre_cancelled_is_inconclusive(ts) -> None:
    array = np.full((64, 64, 3), 128, dtype=np.uint8)
    doc = ImageDocument(
        array=array,
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="c" * 64,
        file_bytes=0,
        file_name="cancel.png",
    )
    cancel = threading.Event()
    cancel.set()

    result = NoiseMap().analyze(doc, ts, cancel)

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert any("cancel" in item.lower() for item in result.limitations)
