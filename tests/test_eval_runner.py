"""Runner end-to-end — deterministic + provenance đầy đủ."""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.evaluation import runner
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
    assert isinstance(prov["working_tree_clean"], bool | type(None))


def test_synth_version_tracks_forged_image_bytes(tmp_path: Path) -> None:
    d = tmp_path / "s"
    generate_set(1, seed=11, out_dir=d, tier="easy")
    adapter = SynthAdapter(d)
    entry = adapter.entries[0]
    ext = ".jpg" if entry.get("format") == "jpeg" else ".png"
    forged_path = d / f"{entry['id']}_F{ext}"
    meta_path = d / "meta.json"
    original_meta = meta_path.read_bytes()
    original_version = adapter.version()

    image_bytes = bytearray(forged_path.read_bytes())
    image_bytes[len(image_bytes) // 2] ^= 1
    forged_path.write_bytes(image_bytes)

    assert meta_path.read_bytes() == original_meta
    assert adapter.version() != original_version


def test_working_tree_clean_provenance_uses_git_porcelain(
    tmp_path: Path, ts, monkeypatch
) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / "meta.json").write_text("[]", encoding="utf-8")
    adapter = SynthAdapter(d)
    porcelain = " M src/imageguard/example.py\n"
    expected_cwd = Path(runner.__file__).resolve().parents[3]

    def fake_check_output(args, **kwargs):
        assert kwargs["cwd"] == expected_cwd
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["timeout"] == 10
        if args == ["git", "rev-parse", "--short", "HEAD"]:
            return "deadbee"
        if args == ["git", "status", "--porcelain"]:
            return porcelain
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(runner.subprocess, "check_output", fake_check_output)
    dirty = run_eval(adapter, [], ts, seed=1)
    assert dirty.provenance["working_tree_clean"] is False

    porcelain = ""
    clean = run_eval(adapter, [], ts, seed=1)
    assert clean.provenance["working_tree_clean"] is True


def test_report_files_written(tmp_path: Path, ts) -> None:
    d = tmp_path / "s"
    generate_set(2, seed=3, out_dir=d, tier="easy")
    r = run_eval(SynthAdapter(d), [KeypointCopyMove()], ts, "dev", seed=3)
    jp, cp = r.save(tmp_path / "rep")
    data = json.loads(jp.read_text())
    assert data["provenance"]["git_sha"]
    assert cp.exists() and "copy_move_keypoint" in cp.read_text()


def test_load_mask_reads_non_ascii_path(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from imageguard.core.models import ImageDocument
    from imageguard.evaluation.runner import _load_mask

    mask = np.zeros((24, 40), dtype=np.uint8)
    mask[4:12, 7:19] = 255
    mask_path = tmp_path / "mặt_nạ.png"
    mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())
    doc = ImageDocument(
        array=np.zeros((24, 40, 3), dtype=np.uint8), width=40, height=24,
        channels=3, image_format="png", sha256="x", file_bytes=1,
        file_name="image.png")

    loaded = _load_mask(mask_path, doc)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == mask.shape
