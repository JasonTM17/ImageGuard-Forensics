"""Course lock: clean, self-consistent EXIF yields NO_INDICATION with no findings.

Metadata that is present, complete, and internally consistent — Make/Model only,
or a DateTime/DateTimeOriginal pair that agrees — is not evidence of tampering.
`audit_exif` must return zero findings and NO_INDICATION, never
SUSPICIOUS_INDICATION. This complements the inconclusive locks in
test_exif_audit_course.py and test_exif_datetime_course.py.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.meta_report import audit_exif

_SIZE = 32


def _doc(present: dict, missing: list[str], tag_count: int) -> ImageDocument:
    """In-memory 32×32 PNG document — pixels are irrelevant to EXIF audit."""
    return ImageDocument(
        array=np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8),
        width=_SIZE,
        height=_SIZE,
        channels=3,
        image_format="png",
        sha256="c" * 64,
        file_bytes=0,
        file_name="exif_clean_course.png",
        exif={"present": present, "missing": missing, "tag_count": tag_count},
    )


def test_clean_make_model_only_is_no_indication(ts) -> None:
    doc = _doc(
        present={"Make": "Canon", "Model": "Canon EOS 5D Mark IV"},
        missing=[],
        tag_count=2,
    )

    r = audit_exif(doc, ts)

    assert r.method == MethodName.EXIF_AUDIT
    assert r.verdict == Verdict.NO_INDICATION
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r.score is None
    assert r.evidence["findings"] == []


def test_matching_datetime_pair_is_no_indication(ts) -> None:
    doc = _doc(
        present={
            "Make": "Canon",
            "DateTime": "2020:01:01 00:00:00",
            "DateTimeOriginal": "2020:01:01 00:00:00",
        },
        missing=[],
        tag_count=3,
    )

    r = audit_exif(doc, ts)

    assert r.method == MethodName.EXIF_AUDIT
    assert r.verdict == Verdict.NO_INDICATION
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r.score is None
    assert r.evidence["findings"] == []
