"""Numeric-only real-camera validation for the exploratory burst cue.

Captures into a temporary directory that is deleted automatically. The script
prints JSON only: no image paths, no persistent frames, no database writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from imageguard.core.enums import MethodName
from imageguard.ingest.camera_burst import capture_device_burst
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService


def _analyze(primary: Path) -> dict[str, object]:
    report = AnalysisService().analyze(
        AnalysisRequest(path=primary, methods=["deepfake"], include_exif=False)
    )
    method = next(m for m in report.methods if m.method is MethodName.DEEPFAKE_CUES)
    return {
        "verdict": method.verdict.value,
        "overall": report.overall_verdict.value,
        "score": method.score,
        "temporal_static_score": method.evidence.get("temporal_static_score"),
        "temporal_face_detected": method.evidence.get("temporal_face_detected"),
        "temporal_frames": method.evidence.get("temporal_frame_count"),
        "temporal_status": method.evidence.get("temporal_status"),
        "temporal_manifest_sha256": method.evidence.get("temporal_manifest_sha256"),
    }


def _write_frozen_control(primary: Path, control_dir: Path) -> Path:
    """Create distinct files with identical pixels to simulate a frozen feed."""
    control_dir.mkdir(parents=True, exist_ok=True)
    raw = primary.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    names = [f"frozen_{i}.png" for i in range(5)]
    for name in names:
        (control_dir / name).write_bytes(raw)
    frozen_primary = control_dir / names[2]
    manifest = {
        "version": 2,
        "device_index": 0,
        "captured_at": "validation-control",
        "primary": names[2],
        "primary_sha256": digest,
        "frame_interval_s": 0.75,
        "frame_read_offsets_s": [0.0, 0.75, 1.5, 2.25, 3.0],
        "frames": names,
        "sha256": {name: digest for name in names},
    }
    (control_dir / f"{frozen_primary.stem}.burst.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return frozen_primary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ig-camera-validation-") as tmp:
        root = Path(tmp)
        capture_started = time.perf_counter()
        primary, _manifest = capture_device_burst(args.device, root / "live", frame_count=5)
        capture_s = round(time.perf_counter() - capture_started, 3)
        live = _analyze(primary)
        frozen = _analyze(_write_frozen_control(primary, root / "frozen"))
    live["capture_s"] = capture_s
    print(json.dumps({"live_burst": live, "frozen_acquisition_control": frozen}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
