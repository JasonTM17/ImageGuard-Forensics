"""Burst lossless không mở webcam thật. Manifest chỉ chứa basename."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.ingest import camera as camera_mod
from imageguard.ingest.camera_burst import capture_device_burst, load_burst_frames


class _FakeCapture:
    def __init__(self) -> None:
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.reads += 1
        frame = np.full((64, 80, 3), 90, dtype=np.uint8)
        frame[0, 0] = (self.reads, self.reads + 1, self.reads + 2)
        return True, frame

    def release(self) -> None:
        self.released = True


def _capture_burst_once(tmp_path: Path, monkeypatch) -> tuple[Path, Path, _FakeCapture]:
    fake = _FakeCapture()
    monkeypatch.setattr(camera_mod, "_open_capture", lambda _idx: fake)
    primary, manifest = capture_device_burst(
        0, tmp_path, frame_count=4, warmup_frames=1, frame_interval_s=0.001
    )
    return primary, manifest, fake


def test_capture_burst_uses_one_open_and_writes_safe_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    primary, manifest, fake = _capture_burst_once(tmp_path, monkeypatch)
    loaded = load_burst_frames(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert fake.released is True
    assert fake.reads == 5  # warmup=1 + frame_count=4 trong một phiên mở
    assert len(loaded) == 4
    assert manifest.name == f"{primary.stem}.burst.json"
    text = manifest.read_text(encoding="utf-8")
    assert ".." not in text
    assert ":\\" not in text
    assert all(item.shape == (64, 80, 3) for item in loaded)
    assert [tuple(int(v) for v in item[0, 0]) for item in loaded] == [
        (reads, reads + 1, reads + 2) for reads in (2, 3, 4, 5)
    ]
    assert data["frame_interval_s"] == 0.001
    assert data["primary_sha256"] == hashlib.sha256(primary.read_bytes()).hexdigest()
    assert set(data["sha256"]) == set(data["frames"])


def test_load_burst_rejects_parent_names(tmp_path: Path) -> None:
    manifest = tmp_path / "shot.burst.json"
    names = ["ok0.png", "../escape.png", "ok2.png", "ok3.png"]
    sha_map = {name: "0" * 64 for name in names}
    manifest.write_text(
        json.dumps({
            "version": 2,
            "primary": names[2],
            "frames": names,
            "frame_interval_s": 0.75,
            "frame_read_offsets_s": [0.0, 0.75, 1.5, 2.25],
            "primary_sha256": sha_map[names[2]],
            "sha256": sha_map,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="không an toàn"):
        load_burst_frames(manifest)


def test_frame_count_outside_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        capture_device_burst(0, tmp_path, frame_count=3)


def test_load_burst_rejects_declared_pixel_bomb_before_decode(tmp_path: Path) -> None:
    def png_header(width: int, height: int) -> bytes:
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + ihdr
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
            + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    names = [f"f{i}.png" for i in range(4)]
    for name in names:
        (tmp_path / name).write_bytes(png_header(10000, 10000))
    manifest = tmp_path / "shot.burst.json"
    payloads = {name: (tmp_path / name).read_bytes() for name in names}
    sha_map = {name: hashlib.sha256(payloads[name]).hexdigest() for name in names}
    manifest.write_text(
        json.dumps({
            "version": 2,
            "primary": names[2],
            "frames": names,
            "frame_interval_s": 0.75,
            "frame_read_offsets_s": [0.0, 0.75, 1.5, 2.25],
            "primary_sha256": sha_map[names[2]],
            "sha256": sha_map,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="TOO_MANY_PIXELS"):
        load_burst_frames(manifest)


def test_load_burst_rejects_manifest_primary_hash_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    primary, manifest, _fake = _capture_burst_once(tmp_path, monkeypatch)
    frame = cv2.imread(str(primary), cv2.IMREAD_COLOR)
    assert frame is not None
    frame[0, 0, 0] = int(frame[0, 0, 0]) ^ 1
    assert cv2.imwrite(str(primary), frame)
    changed_hash = hashlib.sha256(primary.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="primary PNG"):
        load_burst_frames(manifest, expected_primary_sha256=changed_hash)


def test_analyze_reads_burst_manifest_beside_the_png(
    tmp_path: Path, monkeypatch
) -> None:
    primary, _manifest, _fake = _capture_burst_once(tmp_path, monkeypatch)
    seen: list[int] = []

    def _fake_apply(result, frames):
        seen.append(len(frames))
        result.evidence["temporal_frame_count"] = len(frames)
        return result

    monkeypatch.setattr(
        "imageguard.forensics.auxiliary.temporal_face_roi.apply_temporal_cue",
        _fake_apply,
    )
    from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

    report = AnalysisService().analyze(
        AnalysisRequest(path=primary, methods=["deepfake"], include_exif=False)
    )
    assert seen == [4]
    deepfake = next(m for m in report.methods if m.method.value == "deepfake_cues")
    assert deepfake.evidence["temporal_manifest_sha256"] == hashlib.sha256(
        _manifest.read_bytes()
    ).hexdigest()


def test_load_burst_rejects_legacy_v1_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "shot.burst.json"
    manifest.write_text(
        json.dumps({"version": 1, "primary": "f0.png", "frames": [f"f{i}.png" for i in range(4)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        load_burst_frames(manifest)


def test_load_burst_rejects_frame_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    _primary, manifest, _fake = _capture_burst_once(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    first = data["frames"][0]
    frame = cv2.imread(str(tmp_path / first), cv2.IMREAD_COLOR)
    assert frame is not None
    frame[1, 0, 0] = int(frame[1, 0, 0]) ^ 1
    assert cv2.imwrite(str(tmp_path / first), frame)

    with pytest.raises(ValueError, match="sha256 manifest"):
        load_burst_frames(manifest)


def test_zero_frame_interval_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_interval_s"):
        capture_device_burst(0, tmp_path, frame_count=4, frame_interval_s=0.0)


def test_load_burst_rejects_duplicate_names_and_wrong_midpoint(tmp_path: Path, monkeypatch) -> None:
    _primary, manifest, _fake = _capture_burst_once(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    duplicated = dict(data)
    duplicated["frames"] = list(data["frames"])
    duplicated["frames"][0] = duplicated["frames"][1]
    duplicated["primary"] = duplicated["frames"][2]
    manifest.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(ValueError, match="trùng lặp"):
        load_burst_frames(manifest)

    wrong_midpoint = dict(data)
    wrong_midpoint["primary"] = data["frames"][0]
    manifest.write_text(json.dumps(wrong_midpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="frame giữa"):
        load_burst_frames(manifest)


def test_capture_manifest_records_measured_read_offsets(tmp_path: Path, monkeypatch) -> None:
    _primary, manifest, _fake = _capture_burst_once(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    offsets = data["frame_read_offsets_s"]
    assert len(offsets) == 4
    assert offsets[0] == 0.0
    assert all(float(offsets[i]) <= float(offsets[i + 1]) for i in range(3))


def test_load_burst_downscales_working_frames(tmp_path: Path, monkeypatch) -> None:
    class LargeFakeCapture(_FakeCapture):
        def read(self):
            self.reads += 1
            frame = np.full((800, 1000, 3), 90, dtype=np.uint8)
            frame[0, 0] = (self.reads, self.reads + 1, self.reads + 2)
            return True, frame

    fake = LargeFakeCapture()
    monkeypatch.setattr(camera_mod, "_open_capture", lambda _idx: fake)
    _primary, manifest = capture_device_burst(
        0, tmp_path, frame_count=4, warmup_frames=1, frame_interval_s=0.001
    )
    loaded = load_burst_frames(manifest)
    assert all(item.shape == (512, 640, 3) for item in loaded)
