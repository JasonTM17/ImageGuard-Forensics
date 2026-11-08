"""Read-only CASIA native manifest freezer; no detector imports or inference.

The proposal and intake receipts are embedded as exact UTF-8 bytes (represented
as JSON strings). Their pinned raw hashes, rather than a resealable manifest
checksum, establish population authority. Only selected assets are opened.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps

DATASET_KIND = "casia-cmfd-native-v1"
DATASET_ROOT = "data/external/casia-cmfd-busternet/native-cohort-v1"
COHORT_SHA256 = "f8bbe9d196c91dab35d8b3e55ebbb540ce677ce0fc4228976efc5050e85057bc"
ORACLE_POLICY = "casia-busternet-blue-complement-v1"
RANK_DOMAIN = "imageguard-casia-formal-103-v1"
_FREEZE = "casia-native-cohort-v1-freeze.json"
_AUDIT = "casia-native-cohort-v1-audit.json"
_PROJECTION = "casia-author-mask-projection-v1.json"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str), "invalid relative path")
    p = PurePosixPath(relative)
    _require(relative and not p.is_absolute() and "\\" not in relative and ":" not in relative
             and all(part not in {"", ".", ".."} for part in relative.split("/")),
             "unsafe relative path")
    result = root.joinpath(*p.parts)
    _require(not any(x.is_symlink() or x.is_junction() for x in (result, *result.parents)),
             "symlink/junction path")
    _require(result.resolve().is_relative_to(root.resolve()), "path escapes root")
    return result


def _native(relative):
    prefix = DATASET_ROOT + "/"
    _require(isinstance(relative, str) and relative.startswith(prefix), "wrong native root")
    result = relative[len(prefix):]
    # Lexical validation is also performed on held-aside metadata, without I/O.
    _require(result and "\\" not in result and ":" not in result
             and all(x not in {"", ".", ".."} for x in result.split("/")), "unsafe native path")
    return result


def _receipt(raw: bytes, expected: str) -> dict:
    _require(hashlib.sha256(raw).hexdigest() == expected, "receipt hash drift")
    return {"raw_sha256": expected, "raw_json": raw.decode("utf-8")}


def _document(receipt, expected):
    _require(set(receipt) == {"raw_sha256", "raw_json"}, "invalid receipt envelope")
    _require(receipt["raw_sha256"] == expected, "receipt identity drift")
    _receipt(receipt["raw_json"].encode("utf-8"), expected)
    value = json.loads(receipt["raw_json"])
    _require(isinstance(value, dict), "invalid receipt document")
    return value


def _indexed(rows):
    result = {row["original_id"]: row for row in rows}
    _require(len(result) == len(rows), "duplicate original IDs")
    return result


def _metadata(cohort, receipts):
    _require(cohort["protocol"] == "imageguard-casia-formal-cohort-proposal-v1"
             and cohort["seed"] == 42 and cohort["rank_domain"] == RANK_DOMAIN
             and cohort["oracle_policy"] == ORACLE_POLICY
             and cohort["detector_inference"] is False, "cohort policy drift")
    _require(set(receipts) == set(cohort["input_sha256"]) == {_FREEZE, _AUDIT, _PROJECTION},
             "missing lineage receipt")
    docs = {name: _document(receipts[name], sha)
            for name, sha in cohort["input_sha256"].items()}
    freeze, audit, projection = (docs[n] for n in (_FREEZE, _AUDIT, _PROJECTION))
    _require(audit["freeze_sha256"] == cohort["input_sha256"][_FREEZE]
             and projection["cohort_freeze_sha256"] == cohort["input_sha256"][_FREEZE]
             and projection["cohort_audit_sha256"] == cohort["input_sha256"][_AUDIT]
             and projection["policy"] == ORACLE_POLICY
             and audit["detector_inference"] is False
             and projection["detector_inference"] is False, "intake lineage drift")
    selected, held, quarantined = (cohort[k] for k in (
        "selected_pairs", "held_aside_pairs", "quarantined_pairs"))
    _require(selected, "empty selected population")
    all_rows = selected + held + quarantined
    population = _indexed(all_rows)
    _require(len({r["edited_id"] for r in all_rows}) == len(all_rows), "duplicate edited IDs")
    audited, projected = _indexed(audit["rows"]), _indexed(projection["rows"])
    frozen = {r[0]: r[1] for r in freeze["rows"]}
    _require(len(frozen) == len(freeze["rows"])
             and set(frozen) == set(population) == set(audited) == set(projected),
             "intake membership drift")
    survivors = selected + held
    ordered = sorted(survivors, key=lambda r: (
        hashlib.sha256(f"{RANK_DOMAIN}\0{42}\0{r['original_id']}".encode("ascii")).hexdigest(),
        r["original_id"]))
    _require(survivors == ordered, "selected/held rank drift")
    assets, original_pixels = {}, set()
    for rank, row in enumerate(survivors, 1):
        oid, eid = row["original_id"], row["edited_id"]
        _require(oid != eid and row["rank"] == rank
                 and row["rank_sha256"] == hashlib.sha256(
                     f"{RANK_DOMAIN}\0{42}\0{oid}".encode("ascii")).hexdigest(), "rank/ID drift")
        a, p = audited[oid], projected[oid]
        _require(frozen[oid] == a["edited_id"] == p["edited_id"] == eid
                 and not a["issues"] and not a["scene_flags"] and p["state"] == "projected",
                 "eligible lineage drift")
        _require(row["original"] == a["original"] and row["edited"] == a["edited"]
                 and row["raw_mask"] == a["mask"]
                 and row["derived_mask"] == p["derived_mask"]
                 and row["derived_mask_sha256"] == p["derived_mask_sha256"]
                 and p["source_mask"] == row["raw_mask"]["relative_path"]
                 and p["source_mask_sha256"] == row["raw_mask"]["sha256"], "asset lineage drift")
        size = row["original"]["stored_size"]
        _require(size == row["original"]["rendered_size"] == row["edited"]["stored_size"]
                 == row["edited"]["rendered_size"] == row["raw_mask"]["size"] == p["size"]
                 and row["raw_mask"]["orientation"] == 1, "metadata geometry/orientation drift")
        for key in ("original", "edited"):
            item = row[key]
            _require(item["fingerprints"][0] == item["fingerprints"][1],
                     "metadata orientation drift")
            native = _native(item["relative_path"])
            expected_id = oid if key == "original" else eid
            _require(PurePosixPath(native).parts[0] == key
                     and PurePosixPath(native).stem == expected_id, "media path/ID drift")
        pixels = row["original"]["fingerprints"][1][0]
        _require(pixels not in original_pixels, "decoded original alias")
        original_pixels.add(pixels)
        _require(PurePosixPath(_native(row["raw_mask"]["relative_path"])).parts[0] == "mask"
                 and PurePosixPath(row["raw_mask"]["relative_path"]).stem == eid
                 and PurePosixPath(_native(row["derived_mask"])).parts[0] == ORACLE_POLICY
                 and PurePosixPath(row["derived_mask"]).stem == eid, "mask path/ID drift")
        entries = {row[k]["relative_path"]: row[k]["sha256"]
                   for k in ("original", "edited", "raw_mask")}
        entries[row["derived_mask"]] = row["derived_mask_sha256"]
        _require(len(entries) == 4 and not set(entries).intersection(assets), "reused asset")
        assets.update(entries)
    _require(assets == cohort["asset_sha256"], "cohort asset inventory drift")
    for row in quarantined:
        oid, eid = row["original_id"], row["edited_id"]
        a, p = audited[oid], projected[oid]
        _require(frozen[oid] == a["edited_id"] == p["edited_id"] == eid
                 and p["state"] == "quarantined" and (a["issues"] or a["scene_flags"])
                 and row["issues"] == a["issues"] and row["scene_flags"] == a["scene_flags"],
                 "quarantine lineage drift")
    return selected, projected


def _expected(cohort, receipts):
    selected, projected = _metadata(cohort, receipts)
    files, rows = {}, []
    for pair in selected:
        original = _native(pair["original"]["relative_path"])
        for key in ("original", "edited", "raw_mask"):
            files[_native(pair[key]["relative_path"])] = pair[key]["sha256"]
        files[_native(pair["derived_mask"])] = pair["derived_mask_sha256"]
        for label, key in (("authentic", "original"), ("forged", "edited")):
            rows.append({"sample_id": pair[f"{key}_id"],
                         "base_image_id": pair["original_id"], "scene_id": pair["original_id"],
                         "split": "locked", "label": label, "category": "copy_move",
                         "image_path": _native(pair[key]["relative_path"]),
                         "gt_mask_path": _native(pair["derived_mask"]) if key == "edited" else None,
                         "original_path": original})
    return files, rows, selected, projected


def _fingerprint(image):
    rgb = image.convert("RGB")
    pixels = hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    return [pixels, dhash]


def _media(root, files, selected, projected):
    for name, expected in files.items():
        path = _inside(root, name)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(actual == expected, "selected asset hash drift")
    original_pixels = set()
    for pair in selected:
        size = tuple(pair["original"]["stored_size"])
        pair_pixels = {}
        for key in ("original", "edited"):
            with Image.open(_inside(root, _native(pair[key]["relative_path"]))) as image:
                image.load()
                _require(image.getexif().get(274, 1) == 1, "media orientation drift")
                rendered = ImageOps.exif_transpose(image)
                _require(image.size == rendered.size == size, "media geometry drift")
                fp = [_fingerprint(image), _fingerprint(rendered)]
                _require(fp == pair[key]["fingerprints"], "decoded metadata drift")
                pair_pixels[key] = fp[1][0]
        _require(pair_pixels["original"] not in original_pixels, "decoded original alias")
        original_pixels.add(pair_pixels["original"])
        with Image.open(_inside(root, _native(pair["raw_mask"]["relative_path"]))) as raw:
            raw.load()
            _require(raw.size == size and raw.getexif().get(274, 1) == 1,
                     "raw mask geometry/orientation drift")
            _require(raw.mode == "RGB", "raw mask must be exact RGB")
            rgb = np.asarray(raw)
            _require(np.all(np.any(np.all(rgb[..., None, :] == np.array(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]]), axis=-1), axis=-1)),
                "invalid author mask palette")
            expected = np.where(rgb[..., 2] == 0, 255, 0).astype(np.uint8)
        with Image.open(_inside(root, _native(pair["derived_mask"]))) as derived:
            derived.load()
            _require(derived.size == size and derived.getexif().get(274, 1) == 1
                     and derived.mode == "L", "projected mask geometry/orientation/mode drift")
            pixels = np.asarray(derived)
            _require(np.isin(pixels, [0, 255]).all() and np.array_equal(pixels, expected),
                     "projected mask does not equal blue complement")
        foreground = int(np.count_nonzero(expected))
        _require(foreground > 0 and foreground == projected[pair["original_id"]]["foreground_pixels"],
                 "empty mask or foreground metadata drift")


def build_manifest(cohort_path: Path, root: Path, *, repo: Path) -> dict:
    """Return a sealed, fully validated manifest, without writing any files.

    ``root`` overrides the physical native asset directory. ``repo`` supplies
    the checkout boundary for metadata receipts; the manifest keeps portable
    repo-relative paths and never records the physical override.
    """
    cohort_path, root, repo = Path(cohort_path), Path(root), Path(repo)
    # Inspect the supplied spelling before resolving so a metadata symlink is
    # never hidden by normalization.
    relative = cohort_path.absolute().relative_to(repo.absolute()).as_posix()
    cohort_path = _inside(repo, relative)
    receipt = _receipt(cohort_path.read_bytes(), COHORT_SHA256)
    cohort = _document(receipt, COHORT_SHA256)
    receipts = {name: _receipt(_inside(cohort_path.parent, name).read_bytes(), sha)
                for name, sha in cohort["input_sha256"].items()}
    files, rows, _, _ = _expected(cohort, receipts)
    payload = {"protocol": 2, "dataset_kind": DATASET_KIND, "dataset_root": DATASET_ROOT,
               "adapter_version": DATASET_KIND, "seed": 42, "oracle_policy": ORACLE_POLICY,
               "cohort_receipt": receipt, "lineage_receipts": receipts,
               "files": files, "rows": rows, "exclusions": []}
    manifest = {**payload, "sha256": _digest(payload)}
    validate_manifest(manifest, root)
    return manifest


def validate_manifest(manifest: dict, root: Path) -> None:
    """Fail closed on resealed population/lineage/row mutations and asset drift."""
    try:
        _require(set(manifest) == {"protocol", "dataset_kind", "dataset_root", "adapter_version",
                                  "seed", "oracle_policy", "cohort_receipt", "lineage_receipts",
                                  "files", "rows", "exclusions", "sha256"}, "invalid manifest keys")
        _require(manifest["sha256"] == _digest({k: v for k, v in manifest.items() if k != "sha256"})
                 and manifest["protocol"] == 2 and manifest["dataset_kind"] == DATASET_KIND
                 and manifest["adapter_version"] == DATASET_KIND
                 and manifest["dataset_root"] == DATASET_ROOT and manifest["seed"] == 42
                 and manifest["oracle_policy"] == ORACLE_POLICY
                 and manifest["exclusions"] == [], "manifest identity/policy drift")
        cohort = _document(manifest["cohort_receipt"], COHORT_SHA256)
        files, rows, selected, projected = _expected(cohort, manifest["lineage_receipts"])
        _require(manifest["files"] == files and manifest["rows"] == rows,
                 "frozen membership, row or file identity drift")
        _media(Path(root), files, selected, projected)
    except (KeyError, TypeError, IndexError, AttributeError, OSError, UnicodeError) as exc:
        raise ValueError("invalid CASIA manifest or unreadable selected asset") from exc
