import hashlib

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.evaluation.splicing_study import run_splicing_study
from imageguard.forensics.auxiliary.splicing import SplicingCues


def test_synthetic_probe_has_zero_suspicious_verdicts_and_complete_accounting():
    summary = run_splicing_study()

    assert summary["sample_count"] == 3
    assert (
        summary["scored_count"] + summary["excluded_count"]
        == summary["sample_count"]
    )
    assert summary["suspicious_indication_count"] == 0
    assert len(summary["samples"]) == 3
    assert summary["masks_missing"] == sum(
        sample["mask_missing"] for sample in summary["samples"]
    )
    assert summary["cue_absent_count"] == sum(
        sample["cue_absent"] for sample in summary["samples"]
    )
    assert summary["iou_sample_count"] == sum(
        sample["iou"] is not None for sample in summary["samples"]
    )


def test_png_document_is_inconclusive_without_a_cue_mask():
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(256, dtype=np.uint8)[None, :]
    ok, encoded_png = cv2.imencode(".png", array)
    assert ok
    png_bytes = encoded_png.tobytes()
    doc = ImageDocument(
        array=array,
        width=256,
        height=256,
        channels=3,
        image_format="png",
        sha256=hashlib.sha256(png_bytes).hexdigest(),
        file_bytes=len(png_bytes),
        file_name="cue-unavailable.png",
    )

    result = SplicingCues().analyze(doc, load_thresholds())

    assert result.verdict == Verdict.INCONCLUSIVE
    assert result.mask is None
    assert result.evidence["cue_unavailable"] is True
