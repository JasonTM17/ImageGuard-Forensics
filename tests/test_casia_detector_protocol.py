"""Synthetic native-layout fixtures; never open the retained CASIA population."""

import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts import casia_detector_protocol as protocol


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, pixels, mode=None, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels) if mode is None else Image.fromarray(pixels, mode)
    image.save(path, **kwargs)


def _publish(case, monkeypatch):
    """Pin a complete synthetic authority chain solely inside this test process."""
    repo, cohort, docs = case["repo"], case["cohort"], case["docs"]
    for pair in cohort["selected_pairs"] + cohort["held_aside_pairs"]:
        for key in ("original", "edited", "raw_mask"):
            item = pair[key]
            item["sha256"] = _sha(repo / item["relative_path"])
        pair["derived_mask_sha256"] = _sha(repo / pair["derived_mask"])
        projection = next(r for r in docs[protocol._PROJECTION]["rows"]
                          if r["original_id"] == pair["original_id"])
        projection["source_mask_sha256"] = pair["raw_mask"]["sha256"]
        projection["derived_mask_sha256"] = pair["derived_mask_sha256"]
    cohort["asset_sha256"] = {}
    for pair in cohort["selected_pairs"] + cohort["held_aside_pairs"]:
        cohort["asset_sha256"].update({pair[k]["relative_path"]: pair[k]["sha256"]
                                     for k in ("original", "edited", "raw_mask")})
        cohort["asset_sha256"][pair["derived_mask"]] = pair["derived_mask_sha256"]
    reports = case["cohort_path"].parent
    reports.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in (protocol._FREEZE, protocol._AUDIT, protocol._PROJECTION):
        doc = docs[name]
        if name == protocol._AUDIT:
            doc["freeze_sha256"] = hashes[protocol._FREEZE]
        if name == protocol._PROJECTION:
            doc["cohort_freeze_sha256"] = hashes[protocol._FREEZE]
            doc["cohort_audit_sha256"] = hashes[protocol._AUDIT]
        path = reports / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        hashes[name] = _sha(path)
    cohort["input_sha256"] = hashes
    case["cohort_path"].write_text(json.dumps(cohort), encoding="utf-8")
    monkeypatch.setattr(protocol, "COHORT_SHA256", _sha(case["cohort_path"]))


@pytest.fixture
def case(tmp_path, monkeypatch):
    repo = tmp_path / "research repo"
    root = repo / protocol.DATASET_ROOT
    pairs, audit, projection = [], [], []
    rng = np.random.default_rng(42)
    for number in range(3):
        oid, eid = f"Au_arc_{number:05d}", f"Tp_S_NNN_S_N_arc{number:05d}_copy"
        row = {"original_id": oid, "edited_id": eid}
        for key, sid in (("original", oid), ("edited", eid)):
            path = root / key / "nested" / f"{sid}.png"
            _save(path, rng.integers(0, 256, (8, 10, 3), dtype=np.uint8))
            with Image.open(path) as image:
                fp = protocol._fingerprint(image)
            row[key] = {"relative_path": path.relative_to(repo).as_posix(),
                        "sha256": _sha(path), "stored_size": [10, 8], "rendered_size": [10, 8],
                        "fingerprints": [fp, fp]}
        rgb = np.full((8, 10, 3), [0, 0, 255], dtype=np.uint8)
        rgb[1:3, 1:3] = [255, 0, 0]
        rgb[4:6, 5:7] = [0, 255, 0]
        raw_path = root / "mask" / f"{eid}.png"
        _save(raw_path, rgb)
        derived_path = root / protocol.ORACLE_POLICY / f"{eid}.png"
        _save(derived_path, np.where(rgb[..., 2] == 0, 255, 0).astype(np.uint8))
        row["raw_mask"] = {"relative_path": raw_path.relative_to(repo).as_posix(),
                           "sha256": _sha(raw_path), "orientation": 1, "size": [10, 8]}
        row["derived_mask"] = derived_path.relative_to(repo).as_posix()
        row["derived_mask_sha256"] = _sha(derived_path)
        row["rank_sha256"] = hashlib.sha256(
            f"{protocol.RANK_DOMAIN}\0{42}\0{oid}".encode("ascii")).hexdigest()
        pairs.append(row)
        audit.append({"original_id": oid, "edited_id": eid, "issues": [], "scene_flags": [],
                      "original": row["original"], "edited": row["edited"], "mask": row["raw_mask"]})
        projection.append({"original_id": oid, "edited_id": eid, "state": "projected",
                           "size": [10, 8], "source_mask": row["raw_mask"]["relative_path"],
                           "source_mask_sha256": row["raw_mask"]["sha256"],
                           "derived_mask": row["derived_mask"],
                           "derived_mask_sha256": row["derived_mask_sha256"], "foreground_pixels": 8})
    pairs.sort(key=lambda r: (r["rank_sha256"], r["original_id"]))
    for rank, row in enumerate(pairs, 1):
        row["rank"] = rank
    quarantine = {"original_id": "Au_bad", "edited_id": "Tp_bad",
                  "issues": ["geometry"], "scene_flags": []}
    audit.append(copy.deepcopy(quarantine))
    projection.append({"original_id": "Au_bad", "edited_id": "Tp_bad", "state": "quarantined"})
    cohort = {"protocol": "imageguard-casia-formal-cohort-proposal-v1", "seed": 42,
              "rank_domain": protocol.RANK_DOMAIN, "oracle_policy": protocol.ORACLE_POLICY,
              "detector_inference": False, "selected_pairs": pairs[:2], "held_aside_pairs": pairs[2:],
              "quarantined_pairs": [quarantine]}
    docs = {protocol._FREEZE: {"rows": [[r["original_id"], r["edited_id"]]
                                     for r in pairs + [quarantine]]},
            protocol._AUDIT: {"rows": audit, "detector_inference": False},
            protocol._PROJECTION: {"rows": projection, "policy": protocol.ORACLE_POLICY,
                                   "detector_inference": False}}
    result = {"repo": repo, "root": root, "cohort": cohort, "docs": docs,
              "cohort_path": repo / "plans" / "receipts" / "cohort.json"}
    _publish(result, monkeypatch)
    return result


def _build(case):
    return protocol.build_manifest(case["cohort_path"], case["root"], repo=case["repo"])


def _reseal(manifest):
    manifest["sha256"] = protocol._digest({k: v for k, v in manifest.items() if k != "sha256"})
    return manifest


def test_freezes_exact_selected_nested_rows_and_preserves_assets(case):
    before = {p.relative_to(case["root"]): _sha(p) for p in case["root"].rglob("*") if p.is_file()}
    manifest = _build(case)
    assert manifest["protocol"] == 2
    assert manifest["dataset_kind"] == "casia-cmfd-native-v1"
    assert len(manifest["files"]) == 8
    assert len(manifest["rows"]) == 4
    assert {r["split"] for r in manifest["rows"]} == {"locked"}
    assert {r["label"] for r in manifest["rows"]} == {"authentic", "forged"}
    assert len({r["sample_id"] for r in manifest["rows"]}) == 4
    assert all("/nested/" in r["image_path"] for r in manifest["rows"])
    assert all(r["base_image_id"] == r["scene_id"] for r in manifest["rows"])
    protocol.validate_manifest(manifest, case["root"])
    after = {p.relative_to(case["root"]): _sha(p) for p in case["root"].rglob("*") if p.is_file()}
    assert before == after


def test_held_aside_and_quarantine_media_never_opened(case, monkeypatch):
    held = case["cohort"]["held_aside_pairs"][0]
    for key in ("original", "edited", "raw_mask"):
        (case["repo"] / held[key]["relative_path"]).unlink()
    (case["repo"] / held["derived_mask"]).unlink()
    _build(case)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "id", "class", "group", "scene",
                                    "split", "path", "traversal", "gt", "held", "extra_file"])
def test_resealing_cannot_bypass_frozen_identity(case, mutation):
    manifest = _build(case)
    if mutation == "duplicate":
        manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
    elif mutation == "missing":
        manifest["rows"].pop()
    elif mutation == "extra_file":
        manifest["files"]["original/extra.png"] = "0" * 64
    elif mutation == "held":
        manifest["rows"][0]["sample_id"] = case["cohort"]["held_aside_pairs"][0]["original_id"]
    else:
        key, value = {"id": ("sample_id", "Au_wrong"), "class": ("label", "forged"),
                      "group": ("base_image_id", "wrong"), "scene": ("scene_id", "wrong"),
                      "split": ("split", "dev"), "path": ("image_path", "original/wrong.png"),
                      "traversal": ("image_path", "../outside.png"),
                      "gt": ("gt_mask_path", "mask/raw.png")}[mutation]
        manifest["rows"][0][key] = value
    with pytest.raises(ValueError, match="identity drift"):
        protocol.validate_manifest(_reseal(manifest), case["root"])


def test_raw_authority_pin_survives_manifest_reseal(case):
    manifest = _build(case)
    receipt = manifest["cohort_receipt"]
    value = json.loads(receipt["raw_json"])
    value["selected_pairs"], value["held_aside_pairs"] = value["held_aside_pairs"], value["selected_pairs"]
    receipt["raw_json"] = json.dumps(value)
    receipt["raw_sha256"] = hashlib.sha256(receipt["raw_json"].encode()).hexdigest()
    with pytest.raises(ValueError, match="receipt identity drift"):
        protocol.validate_manifest(_reseal(manifest), case["root"])


def test_lineage_receipt_hash_drift(case):
    manifest = _build(case)
    manifest["lineage_receipts"][protocol._PROJECTION]["raw_json"] += " "
    with pytest.raises(ValueError, match="receipt hash drift"):
        protocol.validate_manifest(_reseal(manifest), case["root"])


def test_selected_file_hash_drift(case):
    manifest = _build(case)
    image = case["root"] / manifest["rows"][0]["image_path"]
    image.write_bytes(image.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="asset hash drift"):
        protocol.validate_manifest(manifest, case["root"])


@pytest.mark.parametrize("defect", ["palette", "projection", "nonbinary", "empty", "geometry",
                                  "orientation", "fingerprint", "original_alias"])
def test_rejects_pixel_and_geometry_defects_even_with_coherent_hashes(case, monkeypatch, defect):
    pair = case["cohort"]["selected_pairs"][0]
    raw = case["repo"] / pair["raw_mask"]["relative_path"]
    derived = case["repo"] / pair["derived_mask"]
    if defect == "palette":
        pixels = np.asarray(Image.open(raw)).copy()
        pixels[0, 0] = [255, 255, 0]
        _save(raw, pixels)
    elif defect in {"projection", "nonbinary"}:
        pixels = np.asarray(Image.open(derived)).copy()
        pixels[0, 0] = 255 if defect == "projection" else 1
        _save(derived, pixels)
    elif defect == "empty":
        _save(raw, np.full((8, 10, 3), [0, 0, 255], dtype=np.uint8))
        _save(derived, np.zeros((8, 10), dtype=np.uint8))
    elif defect == "geometry":
        _save(derived, np.zeros((4, 5), dtype=np.uint8))
    elif defect == "orientation":
        image_path = case["repo"] / pair["edited"]["relative_path"]
        with Image.open(image_path) as image:
            exif = image.getexif()
            exif[274] = 6
            image.save(image_path, exif=exif)
    elif defect == "fingerprint":
        pair["original"]["fingerprints"] = [["0" * 64, 0], ["0" * 64, 0]]
    elif defect == "original_alias":
        other = case["cohort"]["selected_pairs"][1]
        path = case["repo"] / pair["original"]["relative_path"]
        path.write_bytes((case["repo"] / other["original"]["relative_path"]).read_bytes())
        pair["original"]["fingerprints"] = copy.deepcopy(other["original"]["fingerprints"])
    _publish(case, monkeypatch)
    with pytest.raises(ValueError):
        _build(case)


@pytest.mark.parametrize("relative", ["../escape.png", "/abs.png", "original/./x.png",
                                      "original//x.png", "C:/x.png", "original\\x.png"])
def test_path_containment_rejects_unsafe_paths(case, relative):
    with pytest.raises(ValueError, match="unsafe"):
        protocol._inside(case["root"], relative)


def test_rejects_symlink_asset(case):
    manifest = _build(case)
    asset = case["root"] / manifest["rows"][0]["image_path"]
    target = asset.with_name("target.png")
    target.write_bytes(asset.read_bytes())
    asset.unlink()
    try:
        asset.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ValueError, match="symlink"):
        protocol.validate_manifest(manifest, case["root"])


@pytest.mark.parametrize("probe", ["is_symlink", "is_junction"])
def test_reparse_point_guard_without_os_privilege(case, monkeypatch, probe):
    path_type = type(case["root"])
    original_probe = getattr(path_type, probe)
    monkeypatch.setattr(path_type, probe,
                        lambda path: path == case["root"] or original_probe(path))
    with pytest.raises(ValueError, match="symlink/junction"):
        _build(case)


@pytest.mark.parametrize("defect", ["duplicate_audit", "missing_projection", "reordered_ranks",
                                   "duplicate_id", "missing_selected", "wrong_size"])
def test_invalid_metadata_authority_fails_closed(case, monkeypatch, defect):
    if defect == "duplicate_audit":
        case["docs"][protocol._AUDIT]["rows"].append(
            copy.deepcopy(case["docs"][protocol._AUDIT]["rows"][0]))
    elif defect == "missing_projection":
        case["docs"][protocol._PROJECTION]["rows"].pop()
    elif defect == "reordered_ranks":
        case["cohort"]["selected_pairs"].reverse()
    elif defect == "duplicate_id":
        case["cohort"]["selected_pairs"][0]["original_id"] = (
            case["cohort"]["selected_pairs"][1]["original_id"])
    elif defect == "missing_selected":
        case["cohort"]["selected_pairs"].pop()
    else:
        case["cohort"]["selected_pairs"][0]["original"]["rendered_size"] = [5, 4]
    _publish(case, monkeypatch)
    with pytest.raises(ValueError):
        _build(case)


def test_physical_root_override_is_portable(case, tmp_path):
    import shutil

    other = tmp_path / "shared native assets"
    shutil.copytree(case["root"], other)
    manifest = protocol.build_manifest(case["cohort_path"], other, repo=case["repo"])
    assert manifest["dataset_root"] == protocol.DATASET_ROOT
    protocol.validate_manifest(manifest, other)
