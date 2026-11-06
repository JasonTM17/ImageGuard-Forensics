"""Small synthetic probe for the splicing cue analyzer.

This is a reproducible course probe, not a camera-photo benchmark. Its scores
and localization overlap describe this generated sample set only.
"""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.splicing import SplicingCues


def run_splicing_study() -> dict[str, object]:
    """Analyze three seeded JPEG composites and summarize cue-mask overlap."""
    seed = 20261106
    sample_count = 3
    rng = np.random.default_rng(seed)
    analyzer = SplicingCues()
    cfg = load_thresholds()
    samples: list[dict[str, object]] = []
    scored_count = 0
    excluded_count = 0
    suspicious_count = 0
    masks_missing = 0
    cue_absent_count = 0
    ious: list[float] = []

    height = width = 256
    yy, xx = np.mgrid[0:height, 0:width]
    for index in range(sample_count):
        base_luma = 72 + xx * 0.28 + yy * 0.16
        base_noise = rng.normal(0, 5.0, (height, width, 1))
        base_color = np.array([0.90, 1.0, 1.08], dtype=np.float32)
        base = np.clip(
            base_luma[..., None] * base_color + base_noise,
            0,
            255,
        ).astype(np.uint8)

        donor_luma = 150 - xx * 0.18 + yy * 0.12
        donor_noise = rng.normal(0, 10.0, (height, width, 1))
        donor_color = np.array([1.08, 0.82, 0.72], dtype=np.float32)
        donor = np.clip(
            donor_luma[..., None] * donor_color + donor_noise,
            0,
            255,
        ).astype(np.uint8)

        paste_height = int(rng.integers(54, 78))
        paste_width = int(rng.integers(54, 78))
        top = int(rng.integers(36, height - paste_height - 35))
        left = int(rng.integers(36, width - paste_width - 35))
        bottom = top + paste_height
        right = left + paste_width
        composite = base.copy()
        composite[top:bottom, left:right] = donor[top:bottom, left:right]

        ground_truth = np.zeros((height, width), dtype=np.uint8)
        ground_truth[top:bottom, left:right] = 255

        ok, encoded = cv2.imencode(
            ".jpg",
            composite,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode the synthetic JPEG")
        jpeg_bytes = encoded.tobytes()
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV could not decode the synthetic JPEG")

        doc = ImageDocument(
            array=decoded,
            width=width,
            height=height,
            channels=3,
            image_format="jpeg",
            sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
            file_bytes=len(jpeg_bytes),
            file_name=f"splicing-probe-{index + 1}.jpg",
        )
        result = analyzer.analyze(doc, cfg)

        if result.score is None:
            excluded_count += 1
        else:
            scored_count += 1
        if result.verdict == Verdict.SUSPICIOUS_INDICATION:
            suspicious_count += 1

        cue_absent = bool(result.evidence.get("cue_unavailable", False))
        if cue_absent:
            cue_absent_count += 1
        mask_missing = result.mask is None
        if mask_missing:
            masks_missing += 1
            iou = None
        else:
            predicted = result.mask > 0
            truth = ground_truth > 0
            union = int(np.logical_or(predicted, truth).sum())
            intersection = int(np.logical_and(predicted, truth).sum())
            iou = intersection / union if union else 0.0
            ious.append(iou)

        samples.append(
            {
                "sample": index + 1,
                "verdict": result.verdict.value,
                "score": result.score,
                "cue_absent": cue_absent,
                "cue_absence_reason": (
                    next(
                        (item for item in result.limitations if item.startswith("cue suy giảm:")),
                        None,
                    )
                    if cue_absent
                    else None
                ),
                "mask_missing": mask_missing,
                "iou": iou,
            }
        )

    return {
        "seed": seed,
        "sample_count": sample_count,
        "scored_count": scored_count,
        "excluded_count": excluded_count,
        "suspicious_indication_count": suspicious_count,
        "masks_missing": masks_missing,
        "cue_absent_count": cue_absent_count,
        "mean_iou": math.fsum(ious) / len(ious) if ious else None,
        "iou_sample_count": len(ious),
        "samples": samples,
    }
