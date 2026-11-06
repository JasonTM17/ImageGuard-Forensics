"""Course lock: EXIF DateTime ≠ DateTimeOriginal yields one mismatch finding."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.meta_report import audit_exif


def _doc_with_exif(present: dict, missing: list[str], tag_count: int) -> ImageDocument:
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    return ImageDocument(
        array=array,
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="d" * 64,
        file_bytes=0,
        file_name="datetime.png",
        exif={"present": present, "missing": missing, "tag_count": tag_count},
    )


def test_exif_datetime_mismatch_locks_single_finding(ts) -> None:
    doc = _doc_with_exif(
        present={
            "DateTime": "2020:01:01 00:00:00",
            "DateTimeOriginal": "2019:01:01 00:00:00",
        },
        missing=[],
        tag_count=2,
    )
    result = audit_exif(doc, ts)

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    findings = result.evidence["findings"]
    assert len(findings) == 1
    text = findings[0]
    assert "DateTime" in text and "DateTimeOriginal" in text
