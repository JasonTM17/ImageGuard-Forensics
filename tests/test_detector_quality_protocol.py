"""Protocol invariants use isolated image fixtures, never real locked inference."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PIL import Image, ImageOps

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from scripts import evaluate_detector_quality as protocol


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data" / "assets"
    root.mkdir(parents=True)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 255
    for sid in ["001", "002", "003", "004", "005", "006", "007", "008", "099", "001_variant"]:
        image = np.random.default_rng(int(sid[:3])).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        for kind, data in (("F", image), ("O", image), ("B", mask)):
            Image.fromarray(data).save(root / f"{sid}_{kind}.png")
    _write_provenance(root)
    out = tmp_path / "manifest.json"
    manifest = protocol.freeze(root, out, repo=tmp_path)
    return root, out, manifest


def _write_provenance(root):
    records = [{"file": p.name, "bytes": p.stat().st_size, "sha256": protocol.file_hash(p)}
               for p in sorted(root.glob("*.png"))]
    (root / "provenance.json").write_text(json.dumps({"files": records}), encoding="utf-8")


def _legacy_manifest(path, manifest):
    payload = {key: manifest[key] for key in (
        "seed", "dataset_root", "adapter_version", "provenance_sha256", "files",
        "exposed_bases", "exclusions")}
    rows = []
    for source in manifest["rows"]:
        row = {key: value for key, value in source.items() if key != "scene_id"}
        base = row["base_image_id"]
        row["split"] = ("diagnostic" if base in payload["exposed_bases"] else
                        "dev" if protocol.split_of(base) == "dev" else "locked")
        rows.append(row)
    legacy = protocol._sealed({"protocol": 1, **payload, "rows": rows})
    protocol._write(path, legacy)
    return legacy


def _reencode_original(root, source, target):
    with Image.open(root / f"{source}_O.png") as image:
        image.save(root / f"{target}_O.png", compress_level=0)
    assert protocol.file_hash(root / f"{source}_O.png") != protocol.file_hash(root / f"{target}_O.png")


def test_same_pixels_across_ids_are_isolated_transitively_including_excluded_base(dataset, tmp_path):
    root, _, _ = dataset
    # Base 001 connects 002 to exposed but excluded 099 through its variant.
    _reencode_original(root, "001", "002")
    _reencode_original(root, "099", "001_variant")
    _write_provenance(root)
    manifest = protocol.freeze(root, tmp_path / "scenes.json", repo=tmp_path)
    linked = [row for row in manifest["rows"] if row["base_image_id"] in {"001", "002"}]
    assert {row["split"] for row in linked} == {"diagnostic"}
    assert len({row["scene_id"] for row in linked}) == 1
    assert not any(row["base_image_id"] == "099" for row in manifest["rows"])
    bad = copy.deepcopy(manifest)
    for row in bad["rows"]:
        if row["base_image_id"] == "002":
            row["split"] = "locked"
    bad.pop("sha256")
    with pytest.raises(ValueError, match="split leakage"):
        protocol.validate_manifest(protocol._sealed(bad), root)


def test_exif_rotated_original_alias_cannot_enter_locked(dataset, tmp_path):
    root, _, _ = dataset
    with Image.open(root / "006_O.png") as original:
        stored = original.transpose(Image.Transpose.ROTATE_90)
        exif = stored.getexif()
        exif[274] = 6  # Display rotated 90 degrees clockwise.
        stored.save(root / "002_O.png", exif=exif)
        with Image.open(root / "002_O.png") as alias:
            assert alias.tobytes() != original.tobytes()
            assert ImageOps.exif_transpose(alias).tobytes() == original.tobytes()
    _write_provenance(root)
    manifest = protocol.freeze(root, tmp_path / "oriented-scenes.json", repo=tmp_path)
    linked = [row for row in manifest["rows"] if row["base_image_id"] in {"002", "006"}]
    assert {row["split"] for row in linked} == {"diagnostic"}
    assert len({row["scene_id"] for row in linked}) == 1


def test_migration_retains_dev_and_only_keeps_cold_groups_locked(dataset, tmp_path):
    root, _, _ = dataset
    _reencode_original(root, "001", "002")
    _write_provenance(root)
    current = protocol.freeze(root, tmp_path / "current.json", repo=tmp_path)
    prior_path = tmp_path / "legacy.json"
    prior = _legacy_manifest(prior_path, current)
    manifest = protocol.freeze(root, tmp_path / "migrated.json", repo=tmp_path,
                               prior_manifest=prior_path)
    def dev_files(m):
        return {row["image_path"] for row in m["rows"] if row["split"] == "dev"}

    assert dev_files(manifest) == dev_files(prior)
    assert {row["base_image_id"] for row in manifest["rows"] if row["split"] == "locked"} == {"003", "007"}
    assert {row["split"] for row in manifest["rows"] if row["base_image_id"] == "002"} == {"diagnostic"}
    assert manifest["prior_manifest"]["sha256"] == prior["sha256"]
    protocol.validate_manifest(manifest, root)


def test_freeze_groups_derivatives_and_quarantines_exposure(dataset):
    root, _, manifest = dataset
    rows = manifest["rows"]
    assert {r["split"] for r in rows if r["base_image_id"] == "001"} == {
        "dev" if protocol.split_of("001") == "dev" else "locked"}
    assert all(r["split"] == "diagnostic" for r in rows if r["base_image_id"] == "006")
    assert not any(r["base_image_id"] == "099" for r in rows)
    assert manifest["exclusions"] == [{"sample_id": "099", "base_image_id": "099",
                                       "reason": "099_mask_convention_unresolved", "n_images": 2}]
    assert manifest["adapter_version"] == "present-unverified"
    protocol.validate_manifest(manifest, root)


@pytest.mark.parametrize("change", ["exposed", "split", "duplicate", "gt", "path", "exclude"])
def test_manifest_rejects_semantically_invalid_resealed_content(dataset, change):
    root, _, manifest = dataset
    bad = copy.deepcopy(manifest)
    if change == "exposed":
        bad["exposed_bases"].remove("006")
    elif change == "split":
        row = next(r for r in bad["rows"] if r["split"] == "dev")
        row["split"] = "locked"
    elif change == "duplicate":
        bad["rows"].append(copy.deepcopy(bad["rows"][0]))
    elif change == "gt":
        next(r for r in bad["rows"] if r["label"] == "forged")["gt_mask_path"] = None
    elif change == "path":
        bad["files"]["../escape_F.png"] = "f" * 64
    else:
        sid = bad["rows"][0]["sample_id"]
        base = bad["rows"][0]["base_image_id"]
        bad["rows"] = [r for r in bad["rows"] if r["sample_id"] != sid]
        bad["exclusions"].append({"sample_id": sid, "base_image_id": base,
                                  "reason": "unreadable_image_or_mask", "n_images": 2})
    bad.pop("sha256")
    with pytest.raises((ValueError, TypeError)):
        protocol.validate_manifest(protocol._sealed(bad), root)


def test_hash_drift_and_unrecorded_assets_are_rejected(dataset):
    root, _, manifest = dataset
    path = root / "001_F.png"
    raw = path.read_bytes()
    path.write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="changed after freeze"):
        protocol.validate_manifest(manifest, root)
    path.write_bytes(raw)
    (root / "009_F.png").write_bytes(raw)
    with pytest.raises(ValueError, match="inventory changed"):
        protocol.validate_manifest(manifest, root)


def test_invalid_gt_excluded_without_mutating_asset(dataset, tmp_path):
    root, _, _ = dataset
    mask = root / "002_B.png"
    Image.fromarray(np.full((64, 64), 127, np.uint8)).save(mask)
    provenance = json.loads((root / "provenance.json").read_text())
    for record in provenance["files"]:
        if record["file"] == mask.name:
            record.update(sha256=protocol.file_hash(mask), bytes=mask.stat().st_size)
    (root / "provenance.json").write_text(json.dumps(provenance))
    original_hash = protocol.file_hash(mask)
    manifest = protocol.freeze(root, tmp_path / "second.json", repo=tmp_path)
    assert {e["sample_id"]: e["reason"] for e in manifest["exclusions"]}["002"] == "non_binary_mask"
    assert protocol.file_hash(mask) == original_hash
    assert not any(r["base_image_id"] == "002" for r in manifest["rows"])


@pytest.mark.parametrize("relative", ["../file", "/tmp/file", "x/../file", "C:/file", "x\\file"])
def test_relative_paths_cannot_escape_dataset(tmp_path, relative):
    with pytest.raises(ValueError):
        protocol._inside(tmp_path, relative)


def test_symlink_dataset_rejected(dataset, tmp_path):
    root, _, manifest = dataset
    link = tmp_path / "link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("OS symlink capability unavailable")
    with pytest.raises(ValueError, match="symlink"):
        protocol.validate_manifest(manifest, link)


class FixtureMethod:
    name = "copy_move_keypoint"

    def __init__(self, *, crash=False):
        self.calls = []
        self.crash = crash

    def analyze(self, doc, cfg):
        self.calls.append(doc.file_name)
        if self.crash and doc.file_name.endswith("_F.png"):
            raise RuntimeError("fixture-only failure")
        return AlgorithmResult(method=MethodName.COPY_MOVE_KEYPOINT,
                               verdict=Verdict.INCONCLUSIVE, mask=None,
                               evidence={"example": np.int64(2)},
                               params_hash=cfg.snapshot_hash())


@pytest.fixture
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(protocol, "source_identity", lambda repo: {
        "git_sha": "a" * 40, "source_sha256": "b" * 64,
        "working_tree_clean": True, "files": {}})
    monkeypatch.setattr(protocol, "protocol_hash", lambda repo: "c" * 64)


@pytest.mark.parametrize("crash", [False, True])
def test_same_invocation_traces_and_denominators(dataset, tmp_path, isolated_runtime, crash):
    root, manifest_path, manifest = dataset
    method = FixtureMethod(crash=crash)
    result = protocol.evaluate(manifest_path, "dev", protocol.REPO / "configs/thresholds.yaml",
                               [method], tmp_path / "out", root=root, repo=tmp_path)
    selected = [r for r in manifest["rows"] if r["split"] == "dev"]
    assert sorted(method.calls) == sorted(r["image_path"] for r in selected)
    assert len(result["rows"]) == len(selected)
    assert result["manifest_sha256"] == manifest["sha256"]
    metric = result["eval"]["per_method"][method.name]
    forged = sum(r["label"] == "forged" for r in selected)
    assert metric["n_scored"] + metric["n_excluded"] == len(selected)
    if crash:
        assert metric["n_excluded"] == forged
        assert result["counts"][method.name]["method_exception"] == forged
    else:
        assert metric["pixel_n_images"] == forged
        assert metric["pixel_macro"]["iou"] == 0.0
        assert result["counts"][method.name]["inconclusive"] == len(selected)
        assert all(r["pixel"]["iou"] == 0.0 for r in result["rows"] if r["label"] == "forged")


def test_locked_inference_requires_bound_contract_before_method_call(dataset, tmp_path, isolated_runtime):
    root, manifest, _ = dataset
    method = FixtureMethod()
    with pytest.raises(ValueError, match="requires --pair-contract"):
        protocol.evaluate(manifest, "locked", protocol.REPO / "configs/thresholds.yaml",
                          [method], tmp_path / "rejected", root=root, repo=tmp_path)
    assert method.calls == []


def _complete_contract(path, manifest, config):
    common_config = protocol.file_hash(config)
    contract = {"approved": True, "event_id": "locked-scene-test",
                "manifest_sha256": manifest["sha256"], "protocol_sha256": "c" * 64,
                "criteria_sha256": "d" * 64, "seed": manifest["seed"],
                "methods": ["copy_move_keypoint"],
                "baseline": {"source_sha256": "b" * 64, "config_sha256": common_config},
                "candidate": {"source_sha256": "f" * 64, "config_sha256": common_config}}
    protocol._write(path, contract)


def test_legacy_locked_manifest_is_historical_before_inference_or_receipt(
        dataset, tmp_path, isolated_runtime):
    root, _, manifest = dataset
    legacy_path = tmp_path / "legacy.json"
    legacy = _legacy_manifest(legacy_path, manifest)
    config = protocol.REPO / "configs/thresholds.yaml"
    pair = tmp_path / "pair.json"
    _complete_contract(pair, legacy, config)
    method = FixtureMethod()
    out = tmp_path / "locked-out"
    with pytest.raises(ValueError, match="historical"):
        protocol.evaluate(legacy_path, "locked", config, [method], out,
                          root=root, repo=tmp_path, pair_contract=pair, role="baseline")
    assert method.calls == []
    assert not out.exists()
    assert not (tmp_path / "event-locked-scene-test").exists()


def test_clean_scene_locked_pair_runs_only_selected_rows(dataset, tmp_path, isolated_runtime):
    root, manifest_path, manifest = dataset
    config = protocol.REPO / "configs/thresholds.yaml"
    pair = tmp_path / "pair.json"
    _complete_contract(pair, manifest, config)
    method = FixtureMethod()
    report = protocol.evaluate(manifest_path, "locked", config, [method], tmp_path / "locked-out",
                               root=root, repo=tmp_path, pair_contract=pair, role="baseline")
    assert report["protocol"] == 2
    expected = {row["image_path"]: row["scene_id"] for row in manifest["rows"]
                if row["split"] == "locked"}
    assert expected
    assert set(method.calls) == set(expected)
    assert {row["image_path"]: row["scene_id"] for row in report["rows"]} == expected
    assert (tmp_path / "event-locked-scene-test" / "baseline.json").is_file()


def _lock_pair_fixture(tmp_path, *, event_id="fixture"):
    identity = {"working_tree_clean": True, "source_sha256": "a" * 64}
    manifest = {"sha256": "b" * 64, "seed": 42}
    contract = {"approved": True, "event_id": event_id, "manifest_sha256": "b" * 64,
                "protocol_sha256": "c" * 64, "criteria_sha256": "d" * 64, "seed": 42,
                "methods": ["copy_move_keypoint"],
                "baseline": {"source_sha256": "a" * 64, "config_sha256": "e" * 64},
                "candidate": {"source_sha256": "f" * 64, "config_sha256": "e" * 64}}
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    receipts_root = tmp_path / "receipts"
    receipts_root.mkdir()
    path = receipts_root / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    args = (path, "baseline", identity, manifest, "e" * 64, ["copy_move_keypoint"], "c" * 64,
            cohort_root)
    return path, identity, manifest, contract, args, cohort_root, receipts_root


def test_event_role_receipt_is_exclusive(tmp_path):
    _, _, _, _, args, _, _ = _lock_pair_fixture(tmp_path)
    protocol._lock_pair(*args)
    with pytest.raises(FileExistsError):
        protocol._lock_pair(*args)


def test_locked_cohort_rejects_consumption_under_different_event_id(tmp_path):
    path, identity, manifest, contract, args, cohort_root, _ = _lock_pair_fixture(
        tmp_path, event_id="event-a")
    protocol._lock_pair(*args)
    contract["event_id"] = "event-b"
    path.write_text(json.dumps(contract), encoding="utf-8")
    candidate_identity = {**identity, "source_sha256": "f" * 64}
    with pytest.raises(ValueError, match="different event_id"):
        protocol._lock_pair(path, "candidate", candidate_identity, manifest, "e" * 64,
                            ["copy_move_keypoint"], "c" * 64, cohort_root)
    assert not (path.parent / "event-event-b").exists()


def test_locked_cohort_rejects_contract_copied_to_new_directory_with_new_event_id(tmp_path):
    path, identity, manifest, contract, args, cohort_root, receipts_root = _lock_pair_fixture(
        tmp_path, event_id="event-a")
    protocol._lock_pair(*args)
    other_receipts = tmp_path / "other-receipts"
    other_receipts.mkdir()
    contract["event_id"] = "event-b"
    copied = other_receipts / "pair.json"
    copied.write_text(json.dumps(contract), encoding="utf-8")
    candidate_identity = {**identity, "source_sha256": "f" * 64}
    with pytest.raises(ValueError, match="different event_id"):
        protocol._lock_pair(copied, "candidate", candidate_identity, manifest, "e" * 64,
                            ["copy_move_keypoint"], "c" * 64, cohort_root)
    assert not (other_receipts / "event-event-b").exists()
    assert not (other_receipts / protocol._COHORT_LEDGER).exists()
    assert (cohort_root / protocol._COHORT_LEDGER).is_file()


def test_locked_cohort_same_event_id_allows_both_roles(tmp_path):
    path, identity, manifest, _, _, cohort_root, receipts_root = _lock_pair_fixture(
        tmp_path, event_id="shared-event")
    protocol._lock_pair(path, "baseline", identity, manifest, "e" * 64,
                        ["copy_move_keypoint"], "c" * 64, cohort_root)
    candidate_identity = {**identity, "source_sha256": "f" * 64}
    protocol._lock_pair(path, "candidate", candidate_identity, manifest, "e" * 64,
                        ["copy_move_keypoint"], "c" * 64, cohort_root)
    event_dir = receipts_root / "event-shared-event"
    assert (event_dir / "baseline.json").is_file()
    assert (event_dir / "candidate.json").is_file()
    assert (cohort_root / protocol._COHORT_LEDGER).is_file()
    assert not (receipts_root / protocol._COHORT_LEDGER).exists()


def test_locked_cohort_ledger_lives_in_cohort_root_not_contract_directory(tmp_path):
    _, _, _, _, args, cohort_root, receipts_root = _lock_pair_fixture(tmp_path)
    protocol._lock_pair(*args)
    assert (cohort_root / protocol._COHORT_LEDGER).is_file()
    assert not (receipts_root / protocol._COHORT_LEDGER).exists()


def test_locked_cohort_ledger_creation_is_exclusive(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    identity = {"manifest_sha256": "b" * 64, "protocol_sha256": "c" * 64,
                "event_id": "fixture", "sha256": "deadbeef"}
    protocol._write(cohort_root / protocol._COHORT_LEDGER, identity)
    with pytest.raises(FileExistsError):
        protocol._write(cohort_root / protocol._COHORT_LEDGER,
                        {"manifest_sha256": "b" * 64, "protocol_sha256": "c" * 64,
                         "event_id": "fixture"})


def test_locked_cohort_corrupt_ledger_fails_closed(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    ledger = cohort_root / protocol._COHORT_LEDGER
    ledger.write_text('{"manifest_sha256":"b",' + '"sha256":"not-a-digest"}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or corrupt"):
        protocol._claim_locked_cohort(cohort_root, manifest_sha256="b" * 64,
                                      protocol_sha256="c" * 64, event_id="fixture")


def test_locked_cohort_claim_fails_when_cohort_root_missing(tmp_path):
    missing = tmp_path / "no-such-cohort"
    with pytest.raises(ValueError, match="missing or inaccessible"):
        protocol._claim_locked_cohort(missing, manifest_sha256="b" * 64,
                                      protocol_sha256="c" * 64, event_id="fixture")


def test_locked_cohort_claim_fails_when_cohort_root_not_writable(tmp_path, monkeypatch):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()

    def deny_write(path, payload):
        raise PermissionError("read-only cohort root")

    monkeypatch.setattr(protocol, "_write", deny_write)
    with pytest.raises(ValueError, match="cannot be written"):
        protocol._claim_locked_cohort(cohort_root, manifest_sha256="b" * 64,
                                      protocol_sha256="c" * 64, event_id="fixture")


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("opposite", [None, {}, [],
                                     {"source_sha256": "short", "config_sha256": "e" * 64},
                                     {"source_sha256": "f" * 64, "config_sha256": 0},
                                     {"source_sha256": "f" * 64, "config_sha256": "e" * 64,
                                      "extra": True}])
def test_locked_pair_rejects_missing_or_malformed_opposite_before_inference(
        dataset, tmp_path, isolated_runtime, role, opposite):
    root, manifest_path, manifest = dataset
    config = protocol.REPO / "configs/thresholds.yaml"
    contract = {"approved": True, "event_id": "complete-pair-required",
                "manifest_sha256": manifest["sha256"], "protocol_sha256": "c" * 64,
                "criteria_sha256": "d" * 64, "seed": 42,
                "methods": ["copy_move_keypoint"],
                role: {"source_sha256": "b" * 64,
                       "config_sha256": protocol.file_hash(config)}}
    if opposite is not None:
        contract["candidate" if role == "baseline" else "baseline"] = opposite
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    method = FixtureMethod()
    with pytest.raises(ValueError, match="complete frozen pair"):
        protocol.evaluate(manifest_path, "locked", config, [method], tmp_path / "locked-out",
                          root=root, repo=tmp_path, pair_contract=path, role=role)
    assert method.calls == []
    assert not (tmp_path / "event-complete-pair-required").exists()


def _coverage_reports(tmp_path, *, migrate_failure):
    """Paired reports: recovery is useful, migration hides newly failed controls."""
    method = "copy_move_keypoint"
    criteria = {"approved": True, "approved_by": "fixture-user", "tolerance": 1e-12,
                "primary": {"method": method, "objective": "tp_increase", "minimum": 5},
                "methods": {method: {"fp_delta_max": 1, "tp_delta_min": 5,
                                     "authentic_inconclusive_delta_max": 0,
                                     "forged_inconclusive_delta_max": 0,
                                     "macro_iou_delta_min": -.01,
                                     "micro_iou_delta_min": -.01}}}
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
    reports = {}
    for role in ("baseline", "candidate"):
        rows = []
        for label in ("authentic", "forged"):
            for index in range(6):
                status = "no_indication"
                if index < 5:
                    if role == "baseline" and label == "forged":
                        status = "method_exception"
                    elif role == "candidate" and label == "forged":
                        status = "suspicious_indication"
                    elif role == "candidate" and migrate_failure:
                        status = "method_exception"
                rows.append({"image_path": f"{label}-{index}.png", "base_image_id": str(index),
                             "scene_id": str(index) * 64,
                             "label": label, "method": method, "status": status})
        scored = [r for r in rows if r["status"] != "method_exception"]
        metric = protocol.image_metrics([r["status"] == "suspicious_indication" for r in scored],
                                        [r["label"] == "forged" for r in scored])
        metric.update(n_scored=len(scored), n_excluded=len(rows) - len(scored), n_error=0,
                      n_not_applicable=0, n_inconclusive=0, pixel_macro={"iou": 0.0},
                      pixel_micro={"iou": 0.0})
        counts = {status: sum(r["status"] == status for r in rows)
                  for status in {r["status"] for r in rows}}
        reports[role] = {"protocol": protocol.PROTOCOL, "protocol_sha256": "a" * 64,
                         "manifest_sha256": "b" * 64, "seed": 42, "methods": [method],
                         "runtime": {}, "n_selected": 12, "pair_sha256": "c" * 64,
                         "criteria_sha256": protocol.file_hash(criteria_path), "split": "locked",
                         "role": role, "rows": rows, "counts": {method: counts},
                         "eval": {"per_method": {method: metric}}}
    return reports, criteria_path


@pytest.mark.parametrize("migrate_failure", [True, False])
def test_compare_rejects_new_row_failures_but_allows_recovery(tmp_path, migrate_failure):
    reports, criteria_path = _coverage_reports(tmp_path, migrate_failure=migrate_failure)
    for role, report in reports.items():
        protocol._write(tmp_path / role / "report.json", protocol._sealed(report))
    result = protocol.compare(tmp_path / "baseline", tmp_path / "candidate", criteria_path,
                              tmp_path / "comparison.json")
    assert result["status"] == ("FAIL" if migrate_failure else "PASS")
    assert ("coverage:new_failed_rows" in result["methods"]["copy_move_keypoint"]["failed"]) is migrate_failure


def test_compare_rejects_legacy_locked_reports(tmp_path):
    reports, criteria_path = _coverage_reports(tmp_path, migrate_failure=False)
    for role, report in reports.items():
        report["protocol"] = 1
        protocol._write(tmp_path / role / "report.json", protocol._sealed(report))
    out = tmp_path / "comparison.json"
    with pytest.raises(ValueError, match="historical"):
        protocol.compare(tmp_path / "baseline", tmp_path / "candidate", criteria_path, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["histogram", "unknown_status", "aggregate_excluded"])
def test_report_rejects_status_or_aggregate_accounting_drift(tmp_path, change):
    reports, _ = _coverage_reports(tmp_path, migrate_failure=False)
    report = reports["baseline"]
    if change == "histogram":
        report["counts"]["copy_move_keypoint"] = {"no_indication": 12}
    elif change == "unknown_status":
        report["rows"][0]["status"] = "unrecognised_failure"
    else:
        report["eval"]["per_method"]["copy_move_keypoint"]["n_excluded"] -= 1
    with pytest.raises(ValueError):
        protocol._check_report(report)


@pytest.mark.parametrize("verdict", [Verdict.ERROR, Verdict.NOT_APPLICABLE])
def test_error_and_not_applicable_keep_trace_and_runner_accounting(
        dataset, tmp_path, isolated_runtime, verdict):
    class StatusMethod(FixtureMethod):
        def analyze(self, doc, cfg):
            self.calls.append(doc.file_name)
            return AlgorithmResult(method=MethodName.COPY_MOVE_KEYPOINT, verdict=verdict)

    root, manifest_path, manifest = dataset
    method = StatusMethod()
    result = protocol.evaluate(manifest_path, "dev", protocol.REPO / "configs/thresholds.yaml",
                               [method], tmp_path / "status", root=root, repo=tmp_path)
    expected = sum(r["split"] == "dev" for r in manifest["rows"])
    metric = result["eval"]["per_method"][method.name]
    assert len(method.calls) == expected
    assert result["counts"][method.name] == {verdict.value: expected}
    assert metric["n_scored"] == metric["n_excluded"] == 0
    assert metric["n_error"] == (expected if verdict == Verdict.ERROR else 0)
    assert metric["n_not_applicable"] == (expected if verdict == Verdict.NOT_APPLICABLE else 0)


def test_malformed_predicted_mask_geometry_rejects_evaluation(
        dataset, tmp_path, isolated_runtime):
    class BadGeometryMethod(FixtureMethod):
        def analyze(self, doc, cfg):
            self.calls.append(doc.file_name)
            return AlgorithmResult(method=MethodName.COPY_MOVE_KEYPOINT,
                                   verdict=Verdict.SUSPICIOUS_INDICATION,
                                   mask=np.ones((doc.height - 1, doc.width), np.uint8))

    root, manifest_path, _ = dataset
    method = BadGeometryMethod()
    out = tmp_path / "bad-geometry"
    with pytest.raises(ValueError, match="invalid predicted mask geometry"):
        protocol.evaluate(manifest_path, "dev", protocol.REPO / "configs/thresholds.yaml",
                          [method], out, root=root, repo=tmp_path)
    assert method.calls
    assert not (out / "report.json").exists()
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "invalid_mask_geometry"
    assert all(r["status"] == "invalid_mask_geometry" for r in failure["rows"])


def test_compare_rejects_partial_pair_and_unapproved_criteria(dataset, tmp_path, isolated_runtime):
    root, manifest, _ = dataset
    result = protocol.evaluate(manifest, "dev", protocol.REPO / "configs/thresholds.yaml",
                               [FixtureMethod()], tmp_path / "base", root=root, repo=tmp_path)
    candidate = copy.deepcopy(result)
    candidate["rows"].pop()
    candidate.pop("sha256")
    protocol._write(tmp_path / "candidate/report.json", protocol._sealed(candidate))
    criteria = tmp_path / "criteria.json"
    criteria.write_text(json.dumps({"approved": False}))
    with pytest.raises(ValueError, match="incomplete/duplicate report"):
        protocol.compare(tmp_path / "base", tmp_path / "candidate", criteria, tmp_path / "comparison.json")
    with pytest.raises(ValueError, match="incomplete locked pair"):
        protocol.compare(tmp_path / "base", tmp_path / "base", criteria, tmp_path / "comparison.json")


def test_freeze_does_not_overwrite_manifest(dataset, tmp_path):
    root, manifest, _ = dataset
    with pytest.raises(FileExistsError):
        protocol.freeze(root, manifest, repo=tmp_path)


def test_wrong_editable_checkout_is_rejected_before_inference(dataset, tmp_path, monkeypatch, isolated_runtime):
    from imageguard.forensics import registry

    root, manifest, _ = dataset
    method = FixtureMethod()
    monkeypatch.setattr(registry, "__file__", str(tmp_path / "foreign/imageguard/registry.py"))
    with pytest.raises(ValueError, match="another checkout"):
        protocol.evaluate(manifest, "dev", protocol.REPO / "configs/thresholds.yaml",
                          [method], tmp_path / "foreign-run", root=root, repo=tmp_path)
    assert method.calls == []


def test_freeze_resolves_root_before_containment_check(dataset, tmp_path):
    root, _, _ = dataset
    sibling = tmp_path.parent / "outside"
    with pytest.raises(ValueError, match="inside the repository"):
        protocol.freeze(root / ".." / ".." / ".." / sibling.name,
                        tmp_path / "rejected.json", repo=tmp_path)


@pytest.mark.parametrize("improves", [True, False])
def test_compare_applies_explicit_correction_and_no_regression_contract(
        dataset, tmp_path, isolated_runtime, improves):
    root, manifest, _ = dataset
    raw = protocol.evaluate(manifest, "dev", protocol.REPO / "configs/thresholds.yaml",
                            [FixtureMethod()], tmp_path / "raw", root=root, repo=tmp_path)
    method = "copy_move_keypoint"
    criteria = {"approved": True, "approved_by": "fixture-user", "tolerance": 1e-12,
                "primary": {"method": method, "objective": "fp_reduction", "minimum": 1},
                "methods": {method: {key: 0 for key in protocol._CRITERIA}}}
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(json.dumps(criteria))
    for role in ("baseline", "candidate"):
        report = copy.deepcopy(raw)
        report.update(split="locked", role=role, pair_sha256="f" * 64,
                      criteria_sha256=protocol.file_hash(criteria_path))
        row = next(r for r in report["rows"] if r["label"] == "authentic")
        row["status"] = ("no_indication" if improves and role == "candidate"
                         else "suspicious_indication")
        metric = report["eval"]["per_method"][method]
        metric.update(protocol.image_metrics(
            [r["status"] == "suspicious_indication" for r in report["rows"]],
            [r["label"] == "forged" for r in report["rows"]]))
        metric["n_inconclusive"] -= 1
        report["counts"][method] = {status: sum(r["status"] == status for r in report["rows"])
                                    for status in {r["status"] for r in report["rows"]}}
        report.pop("sha256")
        protocol._write(tmp_path / role / "report.json", protocol._sealed(report))
    result = protocol.compare(tmp_path / "baseline", tmp_path / "candidate",
                              criteria_path, tmp_path / "comparison.json")
    assert result["status"] == ("PASS" if improves else "FAIL")
    assert ("primary_correction_count" in result["methods"][method]["failed"]) is not improves
