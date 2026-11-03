"""Tests cho L2 signals: noise, freq, splicing — contract AlgorithmResult."""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.forensics.auxiliary.freq_view import FreqView
from imageguard.forensics.auxiliary.noise_map import NoiseMap
from imageguard.forensics.auxiliary.splicing import SplicingCues
from imageguard.forensics.registry import load_method


def _doc(p, ts):
    out = safe_open(p, ts)
    assert not hasattr(out, "code"), getattr(out, "message", "")
    return out


def test_noise_detects_smoother_patch(valid_png, ts):
    """Dán vùng blur (noise bị xoá) → σ thấp → outlier → INCONCLUSIVE."""
    doc = _doc(valid_png, ts)
    img = doc.array.copy()
    img[60:160, 60:160] = cv2.GaussianBlur(img[60:160, 60:160], (15, 15), 5)
    doc.array = img
    r = NoiseMap().analyze(doc, ts)
    assert r.verdict == Verdict.INCONCLUSIVE
    assert r.score is not None and r.score_definition
    assert r.mask is not None and (r.mask > 0).sum() > 0
    assert any("outlier" in l or "corroborate" in l for l in r.limitations)


def test_noise_clean_uniform(valid_jpg, ts):
    """Ảnh noise đồng đều → ít outlier → NO_INDICATION."""
    doc = _doc(valid_jpg, ts)
    r = NoiseMap().analyze(doc, ts)
    assert r.verdict in (Verdict.NO_INDICATION, Verdict.INCONCLUSIVE)
    if r.score is not None:
        assert r.score < 0.5  # ảnh random-uniform không có vùng bất thường lớn


def test_noise_never_suspicious(valid_jpg, ts):
    """Contract cứng: aux signal không bao giờ SUSPICIOUS."""
    doc = _doc(valid_jpg, ts)
    r = NoiseMap().analyze(doc, ts)
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION


def test_freq_jpeg_blockiness(valid_jpg, ts):
    doc = _doc(valid_jpg, ts)
    r = FreqView().analyze(doc, ts)
    assert r.verdict in (Verdict.NO_INDICATION, Verdict.INCONCLUSIVE)
    assert "blockiness" in r.evidence
    assert r.heatmap is not None  # spectrum artifact
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION


def test_freq_nonjpeg(valid_png, ts):
    doc = _doc(valid_png, ts)
    r = FreqView().analyze(doc, ts)
    assert r.verdict in (Verdict.NO_INDICATION, Verdict.INCONCLUSIVE)
    assert r.evidence["format"] == "png"


def test_splicing_jpeg_cues(valid_jpg, ts):
    doc = _doc(valid_jpg, ts)
    img = doc.array.copy()
    img[50:150, 50:150] = cv2.GaussianBlur(img[50:150, 50:150], (15, 15), 5)
    doc.array = img
    r = SplicingCues().analyze(doc, ts)
    assert r.verdict in (Verdict.NO_INDICATION, Verdict.INCONCLUSIVE)
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION
    assert "noise_verdict" in r.evidence


def test_splicing_nonjpeg_degrades(valid_png, ts):
    """Non-JPEG → ELA N/A → cue suy giảm thành NO_INDICATION + ghi lý do."""
    doc = _doc(valid_png, ts)
    r = SplicingCues().analyze(doc, ts)
    assert r.verdict == Verdict.NO_INDICATION
    assert any("suy giảm" in l or "không áp dụng" in l for l in r.limitations)


def test_registry_l2_specs():
    for spec in ("noise", "freq", "splicing", "noise_local"):
        assert load_method(spec) is not None


def test_noise_local_detects_smooth_patch(valid_png, ts):
    """Vùng blur mịn (giả lập model-fill) → σ thấp → component lớn →
    mask phải phủ được vùng patch — khác NoiseMap toàn ảnh."""
    from imageguard.forensics.auxiliary.noise_local import NoiseLocal

    doc = _doc(valid_png, ts)
    img = doc.array.copy()
    img[60:160, 60:160] = cv2.GaussianBlur(img[60:160, 60:160], (15, 15), 5)
    doc.array = img
    r = NoiseLocal().analyze(doc, ts)
    assert r.verdict == Verdict.INCONCLUSIVE
    assert r.mask is not None
    # mask phải phủ một phần thật của patch mịn (60:160 vùng 100px)
    gt = np.zeros_like(r.mask)
    gt[60:160, 60:160] = 255
    inter = ((r.mask > 0) & (gt > 0)).sum()
    assert inter / max((r.mask > 0).sum(), 1) > 0.15, \
        "mask noise_local không chạm vùng mịn"


def test_noise_local_never_suspicious(valid_jpg, ts):
    from imageguard.forensics.auxiliary.noise_local import NoiseLocal

    doc = _doc(valid_jpg, ts)
    r = NoiseLocal().analyze(doc, ts)
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION


def test_cancel_cooperative(valid_png, ts):
    doc = _doc(valid_png, ts)
    ev = threading.Event()
    ev.set()
    r = NoiseMap().analyze(doc, ts, ev)
    assert r.verdict == Verdict.INCONCLUSIVE
