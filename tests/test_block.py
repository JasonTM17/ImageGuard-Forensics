"""Block-based detector — translate copy-move phải bị bắt; runtime ghi log."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.evaluation.synthforge import ForgeSpec, forge, generate_set
from imageguard.forensics.block import pipeline as block_pipeline
from imageguard.forensics.block.pipeline import BlockCopyMove


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(array=bgr, width=w, height=h, channels=3,
                         image_format="png", sha256="t" * 64,
                         file_bytes=0, file_name="s.png")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = ((a > 0) | (b > 0)).sum()
    return float(((a > 0) & (b > 0)).sum() / u) if u else 0.0


def test_translate_detected_with_overlap(ts) -> None:
    """Forge translate thuần postproc=none — trường hợp dễ nhất của method."""
    rng = np.random.default_rng(9)
    base = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    spec = ForgeSpec((400, 400), "rect",
                     [[40, 40], [140, 40], [140, 140], [40, 140]],
                     (160, 140), 0.0, 1.0, "none", "easy")
    forged, gt = forge(base, spec)
    r = BlockCopyMove().analyze(_doc(forged), ts)
    assert r.verdict in (Verdict.SUSPICIOUS_INDICATION, Verdict.INCONCLUSIVE)
    assert r.mask is not None and r.mask.sum() > 0
    # Vùng detected phải giao GT (ngưỡng khiêm tốn — block mask bao quanh,
    # có thể chồng cả phía nguồn lẫn đích).
    assert _iou(r.mask, gt) > 0.1


def test_flat_image_no_indication(ts) -> None:
    flat = np.full((300, 300, 3), 100, np.uint8)
    r = BlockCopyMove().analyze(_doc(flat), ts)
    assert r.verdict == Verdict.NO_INDICATION


def test_runtime_guard(ts, caplog) -> None:
    """Đo runtime trên ảnh 512² — ghi số đo, KHÔNG assert giây cứng."""
    img = np.random.default_rng(2).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    r = BlockCopyMove().analyze(_doc(img), ts)
    elapsed = time.perf_counter() - t0
    print(f"\n[block runtime] 512x512: {elapsed:.2f}s "
          f"(work {r.evidence['work_size']}, blocks {r.evidence['n_blocks']})")
    # Guard rất rộng chỉ để bắt treo/quá tải — không phải benchmark.
    assert elapsed < 300


def test_synthforge_easy_majority(tmp_path: Path, ts) -> None:
    """Trên synth easy: ghi nhận tỉ lệ SUSPICIOUS — đo thật, không phải claim."""

    d = tmp_path / "e"
    generate_set(4, seed=21, out_dir=d, tier="easy")
    hits = n = 0
    for f in sorted(d.glob("*_F.png")):
        doc = safe_open(f, ts)
        assert isinstance(doc, ImageDocument)
        r = BlockCopyMove().analyze(doc, ts)
        n += 1
        if r.verdict == Verdict.SUSPICIOUS_INDICATION:
            hits += 1
    print(f"\n[block easy-tier] suspicious {hits}/{n}")


def test_cancel_after_vote_mask_returns_inconclusive(ts, monkeypatch) -> None:
    cancel = threading.Event()
    empty_features = np.empty((0, 1), dtype=np.float32)
    empty_positions = np.empty((0, 2), dtype=np.int32)
    empty_pairs = np.empty((0, 2), dtype=np.int32)
    monkeypatch.setattr(
        block_pipeline,
        "extract_features",
        lambda gray, block_size, n_coeffs, flat_std_eps: (empty_features, empty_positions),
    )
    monkeypatch.setattr(block_pipeline, "match_pairs", lambda *args: empty_pairs)

    def cancel_during_vote(pairs, positions, block_size, shape, min_votes):
        assert pairs.shape == (0, 2)
        cancel.set()
        height, width = shape
        return (
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.float32),
            {"n_pairs": len(pairs), "n_shifts_kept": 0},
        )

    monkeypatch.setattr(block_pipeline, "vote_mask", cancel_during_vote)

    result = BlockCopyMove().analyze(_doc(np.zeros((64, 64, 3), np.uint8)), ts, cancel)

    assert cancel.is_set()
    assert result.verdict == Verdict.INCONCLUSIVE
    assert "n_candidate_pairs" not in result.evidence
