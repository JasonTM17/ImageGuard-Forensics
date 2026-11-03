"""Runner end-to-end — deterministic + provenance đầy đủ."""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.evaluation.datasets import SynthAdapter
from imageguard.evaluation.runner import run_eval
from imageguard.evaluation.synthforge import generate_set
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def test_eval_deterministic_and_provenance(tmp_path: Path, ts) -> None:
    d = tmp_path / "s"
    generate_set(4, seed=7, out_dir=d, tier="easy")
    adapter = SynthAdapter(d)
    m = [KeypointCopyMove()]
    r1 = run_eval(adapter, m, ts, "dev", seed=7)
    r2 = run_eval(adapter, m, ts, "dev", seed=7)
    for name in r1.per_method:
        a, b = r1.per_method[name], r2.per_method[name]
        # cùng seed + cùng config → số đo giống hệt (trừ experiment_id timestamp)
        assert a["tp"] == b["tp"] and a["f1"] == b["f1"]
        assert a["n"] == b["n"]
    prov = r1.provenance
    assert prov["seed"] == 7 and prov["config_hash"] and prov["dataset_version"] != "absent"
    assert len(prov["git_sha"]) >= 7


def test_report_files_written(tmp_path: Path, ts) -> None:
    d = tmp_path / "s"
    generate_set(2, seed=3, out_dir=d, tier="easy")
    r = run_eval(SynthAdapter(d), [KeypointCopyMove()], ts, "dev", seed=3)
    jp, cp = r.save(tmp_path / "rep")
    data = json.loads(jp.read_text())
    assert data["provenance"]["git_sha"]
    assert cp.exists() and "copy_move_keypoint" in cp.read_text()
