"""Coverage gaps: evaluation.robustness.save_battery."""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.evaluation.robustness import save_battery


def test_save_battery_writes_json_roundtrip(tmp_path: Path) -> None:
    payload = {
        "copy_move_keypoint": {
            "none": {"n": 2, "n_suspicious": 1, "n_inconclusive": 0, "detection_rate": 0.5},
        },
    }
    out = tmp_path / "nested" / "battery.json"
    path = save_battery(payload, out)

    assert path == out
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload
