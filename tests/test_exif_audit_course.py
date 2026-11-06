"""Course lock: EXIF audit reports findings but never convicts.

Metadata is self-declared and easy to edit or strip. An image with no EXIF
(common after a re-save or a social-network round trip) and an image whose only
trace is an editor signature (Photoshop) must both stay INCONCLUSIVE — the
first with an explicit "no EXIF" finding, the second with a single editor
finding. `audit_exif` never emits a score and never SUSPICIOUS_INDICATION.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.meta_report import audit_exif

_SIZE = 64


def _doc(exif: dict) -> ImageDocument:
    """In-memory 64×64 PNG document — pixels are irrelevant to EXIF audit."""
    img = np.random.default_rng(0).integers(
        0, 256, (_SIZE, _SIZE, 3), dtype=np.uint8
    )
    return ImageDocument(
        array=img,
        width=_SIZE,
        height=_SIZE,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=0,
        file_name="exif_course.png",
        exif=exif,
    )


def test_absent_exif_is_inconclusive_with_no_exif_finding() -> None:
    doc = _doc({"tag_count": 0, "present": {}, "missing": ["Make", "Model"]})

    r = audit_exif(doc)

    assert r.method == MethodName.EXIF_AUDIT
    assert r.verdict == Verdict.INCONCLUSIVE
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r.score is None
    findings = r.evidence["findings"]
    assert len(findings) >= 2
    assert any("không có EXIF" in f for f in findings)


def test_photoshop_signature_is_one_inconclusive_finding() -> None:
    doc = _doc({
        "tag_count": 1,
        "present": {"Software": "Adobe Photoshop 24.0"},
        "missing": [],
    })

    r = audit_exif(doc)

    assert r.verdict == Verdict.INCONCLUSIVE
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r.score is None
    findings = r.evidence["findings"]
    assert len(findings) == 1
    assert "Photoshop" in findings[0] or "chỉnh sửa" in findings[0]
