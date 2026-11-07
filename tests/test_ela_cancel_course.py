"""Course lock: ELA returns inconclusive when cancel is already set."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA


def test_ela_pre_cancelled_on_jpeg_is_inconclusive(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    array = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    path = tmp_path / "cancel.jpg"
    assert cv2.imwrite(str(path), array, [cv2.IMWRITE_JPEG_QUALITY, 85])

    doc = safe_open(path, ts)
    assert isinstance(doc, ImageDocument)

    cancel = threading.Event()
    cancel.set()
    result = ELA().analyze(doc, ts, cancel)

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert any("cancel" in item.lower() for item in result.limitations)
