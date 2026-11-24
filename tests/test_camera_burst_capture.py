"""Burst lossless không mở webcam thật. Manifest chỉ chứa basename."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imageguard.ingest import camera as camera_mod
from imageguard.ingest.camera_burst import capture_device_burst, load_burst_frames


def test_capture_burst_writes_frames_and_basename_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    frame = np.full((32, 40, 3), 90, dtype=np.uint8)
    monkeypatch.setattr(
        camera_mod, "_read_frame_from_device", lambda _idx, _warm: frame.copy()
    )
    primary, manifest = capture_device_burst(0, tmp_path, frame_count=4, warmup_frames=1)
    loaded = load_burst_frames(manifest)
    assert len(loaded) == 4
    assert manifest.name == f"{primary.stem}.burst.json"
    text = manifest.read_text(encoding="utf-8")
    assert ".." not in text
    assert ":\\" not in text
    assert all(item.shape == frame.shape for item in loaded)


def test_load_burst_rejects_parent_names(tmp_path: Path) -> None:
    manifest = tmp_path / "shot.burst.json"
    manifest.write_text(
        '{"version":1,"frames":["ok.png","../escape.png","ok.png","ok.png"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_burst_frames(manifest)


def test_frame_count_outside_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        capture_device_burst(0, tmp_path, frame_count=3)


def test_analyze_reads_burst_manifest_beside_the_png(
    tmp_path: Path, monkeypatch
) -> None:
    frame = np.full((96, 128, 3), 40, dtype=np.uint8)
    monkeypatch.setattr(
        camera_mod, "_read_frame_from_device", lambda _idx, _warm: frame.copy()
    )
    primary, _manifest = capture_device_burst(0, tmp_path, frame_count=4, warmup_frames=1)
    seen: list[int] = []

    def _fake_apply(result, frames):
        seen.append(len(frames))
        return result

    monkeypatch.setattr(
        "imageguard.forensics.auxiliary.temporal_face_roi.apply_temporal_cue",
        _fake_apply,
    )
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

    AnalysisService().analyze(
        AnalysisRequest(path=primary, methods=["deepfake"], include_exif=False)
    )
    assert seen == [4]
