"""Run the actual local CLI pipeline on generated face-shaped controls.

The real/fake labels here are software-fixture labels, not real Deepfake ground
truth. This test verifies artifact wiring and source-group isolation, never
detection accuracy on photographs or generated human faces.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

from imageguard.forensics.auxiliary.deepfake_model import FEATURE_NAMES, load_model
from tests.test_face_roi_selection import _frontal_face_control

REPO = Path(__file__).resolve().parents[1]


def _cli(*arguments: str | Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    # Redirected Windows processes can inherit a non-UTF-8 code page. Both
    # study scripts must emit UTF-8 JSON and support Vietnamese artifact names.
    environment["PYTHONIOENCODING"] = "ascii"
    source_path = str(REPO / "src")
    environment["PYTHONPATH"] = source_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, *map(str, arguments)], cwd=REPO, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )


def test_training_evaluation_and_analyze_clis_share_one_validated_model(tmp_path):
    samples = []
    for group, seed in enumerate((301, 302, 303, 401)):
        split = "dev" if group < 3 else "test"
        for label, noise in (("real", 12.0), ("fake", 2.0)):
            sid = f"{split}_{group}_{label}"
            # Each pair shares a generated source control; all versions stay
            # in one declared group. Full pixel/face processing is exercised.
            image = _frontal_face_control(
                (135 - group, 168 - group, 215 - group), angle=group - 1,
                noise=noise, seed=seed,
            )
            image_path = tmp_path / f"{sid}.png"
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            image_path.write_bytes(encoded.tobytes())
            samples.append({"id": sid, "path": image_path.name, "label": label,
                            "split": split, "source_group": f"generated_source_{group}"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "samples": samples}), encoding="utf-8")
    model_path, audit_path = tmp_path / "mô_hình_đã_học.json", tmp_path / "dev_audit.json"

    trained = _cli("scripts/train_deepfake_model.py", "--manifest", manifest, "--root", tmp_path,
                   "--folds", "3", "--output", model_path, "--report", audit_path)
    assert trained.returncode == 0, trained.stderr
    assert model_path.name in trained.stdout
    model = load_model(model_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert model["training"]["n_fitted"] == 6
    assert model["training"]["n_excluded"] == 0
    assert model["training"]["sample_ids"] == sorted(row["id"] for row in samples if row["split"] == "dev")
    assert model["training"]["source_groups"] == [f"generated_source_{group}" for group in range(3)]
    assert all(set(row["features"]) == set(FEATURE_NAMES) for row in audit["study"]["samples"])
    for fold in audit["training"]["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])

    evaluated = _cli("scripts/evaluate_deepfake_cues.py", "--manifest", manifest, "--root", tmp_path,
                     "--split", "test", "--model", model_path)
    assert evaluated.returncode == 0, evaluated.stderr
    evaluation = json.loads(evaluated.stdout)
    assert evaluation["model_sha256"] == model["model_sha256"]
    assert evaluation["summary"]["n_total"] == 2
    assert evaluation["summary"]["n_eligible"] == 2
    assert all(row["split"] == "test" for row in evaluation["samples"])

    heldout = next(row for row in samples if row["split"] == "test" and row["label"] == "real")
    analyzed = _cli("-m", "imageguard", "analyze", tmp_path / heldout["path"],
                    "--methods", "deepfake", "--deepfake-model", model_path, "--no-persist")
    assert analyzed.returncode == 0, analyzed.stderr
    report = json.loads(analyzed.stdout)
    result = next(method for method in report["methods"] if method["method"] == "deepfake_cues")
    assert result["evidence"]["model_status"] == "applied"
    assert result["evidence"]["model_sha256"] == model["model_sha256"]
    assert result["evidence"]["model_in_training_data"] is False
    assert result["evidence"]["score_is_probability"] is False
    assert result["score"] is not None

    # A separate manifest cannot bypass the model's source ledger: change only
    # the declared group of a byte-distinct held-out image to a dev group.
    leaked = {**heldout, "id": "new_test_id", "source_group": "generated_source_0"}
    leak_manifest = tmp_path / "leak_manifest.json"
    leak_manifest.write_text(json.dumps({"version": 1, "samples": [leaked]}), encoding="utf-8")
    rejected = _cli("scripts/evaluate_deepfake_cues.py", "--manifest", leak_manifest, "--root", tmp_path,
                    "--model", model_path)
    assert rejected.returncode == 2
    assert "overlaps model development data" in rejected.stderr
    assert load_model(model_path)["model_sha256"] == model["model_sha256"]
