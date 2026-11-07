"""Course lock: non-editor Software EXIF is inconclusive processing cue."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.meta_report import audit_exif


def test_non_editor_software_exif_is_inconclusive(ts) -> None:
    doc = ImageDocument(
        array=np.zeros((16, 16, 3), dtype=np.uint8),
        width=16,
        height=16,
        channels=3,
        image_format="png",
        sha256="f" * 64,
        file_bytes=0,
        file_name="firmware.png",
        exif={
            "present": {"Software": "SomeCameraFirmware 1.0"},
            "missing": [],
            "tag_count": 1,
        },
    )
    result = audit_exif(doc, ts)
    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    findings = result.evidence["findings"]
    assert len(findings) == 1
    assert "phần mềm xử lý" in findings[0]
