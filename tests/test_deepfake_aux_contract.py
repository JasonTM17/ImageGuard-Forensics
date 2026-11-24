"""Wukong gate: deepfake stays exploratory when no reference and no burst are loaded."""

from __future__ import annotations

from pathlib import Path

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.fusion import AUX_METHODS, combine
from tests.test_deepfake_cues import _doc, _synthetic_face_frame


def test_deepfake_module_does_not_assign_suspicious() -> None:
    source = Path("src/imageguard/forensics/auxiliary/deepfake_cues.py").read_text(
        encoding="utf-8"
    )
    assert "Verdict.SUSPICIOUS_INDICATION" not in source
    assert MethodName.DEEPFAKE_CUES in AUX_METHODS


def test_analyze_and_fusion_stay_exploratory_for_score_thresholds() -> None:
    frame = _synthetic_face_frame(smooth_face=True)
    doc = _doc(frame, "webcam-smooth.png")
    for thresh in (0.0, 0.35, 1.0):
        cfg = load_thresholds()
        cfg.raw["deepfake"] = {"score_thresh": thresh}
        result = DeepfakeCues().analyze(doc, cfg, None)
        assert result.verdict in {
            Verdict.INCONCLUSIVE,
            Verdict.NO_INDICATION,
            Verdict.NOT_APPLICABLE,
        }
        overall, _signals, _warnings = combine([result])
        assert overall != Verdict.SUSPICIOUS_INDICATION


def test_fusion_would_promote_a_forged_suspicious_deepfake_result() -> None:
    forged = AlgorithmResult(
        method=MethodName.DEEPFAKE_CUES,
        verdict=Verdict.SUSPICIOUS_INDICATION,
    )
    overall, _signals, _warnings = combine([forged])
    assert overall == Verdict.SUSPICIOUS_INDICATION
