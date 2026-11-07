"""Course lock: SplicingCues returns inconclusive when cancel is already set."""

from __future__ import annotations

import threading

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_splicing_pre_cancelled_is_inconclusive(ts) -> None:
    rng = np.random.default_rng(23)
    array = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    doc = ImageDocument(
        array=array,
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="s" * 64,
        file_bytes=0,
        file_name="sp_cancel.png",
    )
    cancel = threading.Event()
    cancel.set()

    result = SplicingCues().analyze(doc, ts, cancel)

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert any("cancel" in item.lower() for item in result.limitations)
