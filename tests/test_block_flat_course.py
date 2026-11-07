"""Course lock: no usable DCT blocks means the method must abstain."""

from __future__ import annotations

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.block import pipeline as block_pipeline
from imageguard.forensics.block.pipeline import BlockCopyMove


def test_uniform_field_block_has_no_features() -> None:
    doc = ImageDocument(
        array=np.full((64, 64, 3), 80, np.uint8),
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="e" * 64,
        file_bytes=3,
        file_name="n.png",
    )
    cfg = load_thresholds()
    result = BlockCopyMove().analyze(doc, cfg)
    assert result.method is MethodName.COPY_MOVE_BLOCK
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.score == 0.0
    assert result.params_hash == cfg.snapshot_hash()
    assert result.mask.shape == (64, 64)
    assert result.heatmap.shape == (64, 64)
    assert int(np.count_nonzero(result.mask)) == 0
    assert any("không đủ bằng chứng" in text for text in result.limitations)
    assert result.evidence == {
        "n_blocks": 0,
        "n_candidate_pairs": 0,
        "n_shifts_kept": 0,
        "top_shift_votes": 0,
        "work_size": [64, 64],
    }


def test_informative_blocks_without_shift_stay_no_indication(ts, monkeypatch):
    doc = ImageDocument(
        array=np.zeros((64, 64, 3), dtype=np.uint8),
        width=64,
        height=64,
        channels=3,
        image_format="png",
        sha256="d" * 64,
        file_bytes=64 * 64 * 3,
        file_name="no-match.png",
    )
    features = np.ones((2, 24), dtype=np.int32)
    positions = np.array([[0, 0], [40, 40]], dtype=np.int32)
    monkeypatch.setattr(
        block_pipeline,
        "extract_features",
        lambda *args, **kwargs: (features, positions),
    )
    monkeypatch.setattr(
        block_pipeline, "match_pairs",
        lambda *args, **kwargs: np.empty((0, 2), dtype=np.int32),
    )

    result = BlockCopyMove().analyze(doc, ts)

    assert result.verdict is Verdict.NO_INDICATION
    assert result.evidence["n_blocks"] == 2
    assert result.evidence["n_candidate_pairs"] == 0
    assert result.score == 0.0
