import hashlib

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.evaluation.splicing_study import run_splicing_study
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_synthetic_probe_has_zero_suspicious_verdicts_and_complete_accounting():
    summary = run_splicing_study()

    assert summary["sample_count"] == 3
    assert summary["scored_count"] + summary["excluded_count"] == 3
    assert summary["suspicious_indication_count"] == 0
    assert summary["masks_missing"] <= 3
    assert len(summary["samples"]) == 3


def test_png_document_is_inconclusive_without_a_cue_mask():
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(256, dtype=np.uint8)[None, :]
    png_identity = b"synthetic-png-document"
    doc = ImageDocument(
        array=array,
        width=256,
        height=256,
        channels=3,
        image_format="png",
        sha256=hashlib.sha256(png_identity).hexdigest(),
        file_bytes=len(png_identity),
        file_name="cue-unavailable.png",
    )

    result = SplicingCues().analyze(doc, load_thresholds())

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.mask is None
    assert result.evidence["cue_unavailable"] is True
