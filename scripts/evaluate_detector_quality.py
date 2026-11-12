"""Content-hashed detector studies; locked inference requires a frozen pair contract.

This research CLI wraps the production runner without changing its metrics or
detectors. All artifact paths are relative; official assets are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imageguard.core import limits as core_limits
from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.evaluation.comofod import CoMoFoDAdapter
from imageguard.evaluation.datasets import Sample, split_of
from imageguard.evaluation.metrics import aggregate_pixel, image_metrics, pixel_metrics
from imageguard.evaluation.runner import run_eval
from imageguard.forensics.registry import load_method
from scripts import casia_detector_protocol

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = 2
EXPOSED = frozenset(["006", "028", "029", "048", "058", "098", "099", "121", "127", "132", "138"])
_BASE = re.compile(r"^(\d{3})(?:[_-].*)?$")
_SUFFIX = re.compile(r"^(.+)_(F|O|B|C|M)\.(png|jpg|jpeg|tif|bmp)$", re.IGNORECASE)
_CRITERIA = {"fp_delta_max", "tp_delta_min", "authentic_inconclusive_delta_max",
             "forged_inconclusive_delta_max", "macro_iou_delta_min", "micro_iou_delta_min"}
_PIXEL_REPORT_TOLERANCE = 1e-12
_SCORED_STATUS = frozenset(v.value for v in (
    Verdict.SUSPICIOUS_INDICATION, Verdict.NO_INDICATION, Verdict.INCONCLUSIVE))
_UNSCORED_STATUS = frozenset((Verdict.ERROR.value, Verdict.NOT_APPLICABLE.value,
                             "method_exception", "input_excluded"))
_COHORT_LEDGER = "locked-cohort-consumption.json"


class HistoricalProtocolError(ValueError):
    """V1 data may establish exposure history but cannot support inference."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_path(path: Path) -> None:
    if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
        raise ValueError("symlink/junction paths are not accepted")


def _inside(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if (not relative or p.is_absolute() or "\\" in relative or ":" in relative
            or any(part in {".", ".."} for part in relative.split("/"))):
        raise ValueError("unsafe relative artifact path")
    result = root.joinpath(*p.parts)
    _plain_path(result)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes dataset root")
    return result


def _base(sid: str) -> str:
    match = _BASE.fullmatch(sid)
    if not match:
        raise ValueError("expected a three-digit CoMoFoD base ID")
    return match[1]


def _sealed(payload: dict) -> dict:
    return {**payload, "sha256": digest(payload)}


def _load_sealed(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    checksum = value.pop("sha256", None)
    if checksum != digest(value):
        raise ValueError("artifact checksum mismatch")
    value["sha256"] = checksum
    return value


def _write(path: Path, payload: dict) -> None:
    _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _enforce_decode_limits(path: Path) -> None:
    """Reject paths that exceed shared file/header decode limits before pixel decode."""
    _plain_path(path)
    try:
        size = path.stat().st_size
    except OSError as err:
        raise ValueError(f"unreadable image: {path.name}") from err
    if size > core_limits.MAX_FILE_BYTES:
        raise ValueError(
            f"image file exceeds limit ({size} > {core_limits.MAX_FILE_BYTES}): {path.name}"
        )
    try:
        with Image.open(path) as probe:
            width, height = probe.size
    except Image.DecompressionBombError as err:
        raise ValueError(
            f"image dimensions exceed decode limits: {path.name}"
        ) from err
    except (OSError, ValueError) as err:
        raise ValueError(f"image header unreadable: {path.name}") from err
    if (width * height > core_limits.MAX_PIXELS
            or max(width, height) > core_limits.MAX_DECODE_SIDE):
        raise ValueError(
            f"image dimensions exceed decode limits ({width}x{height}): {path.name}"
        )


def _strict_pair(image: Path, original: Path, mask: Path) -> str | None:
    for path in (image, original, mask):
        try:
            _enforce_decode_limits(path)
        except ValueError:
            return "decode_limit_exceeded"
    try:
        with Image.open(image) as f, Image.open(original) as o, Image.open(mask) as m:
            f.load()
            o.load()
            m.load()
            if f.size != o.size or m.size != f.size:
                return "geometry_mismatch"
            values = np.asarray(m.convert("L"))
            if not np.isin(values, [0, 255]).all():
                return "non_binary_mask"
            if not np.any(values == 255):
                return "empty_mask"
    except (OSError, ValueError, Image.DecompressionBombError):
        return "unreadable_image_or_mask"
    return None


def _inventory(files: dict) -> dict:
    groups = {}
    for name in files:
        match = _SUFFIX.fullmatch(name)
        if not match:
            raise ValueError("unsupported manifest file")
        sid, kind = match[1], match[2].upper()
        _base(sid)
        if kind in groups.setdefault(sid, {}):
            raise ValueError("duplicate image/mask kind")
        groups[sid][kind] = name
    if any(not {"F", "O", "B"}.issubset(group) for group in groups.values()):
        raise ValueError("incomplete original/forged/binary-mask inventory")
    return groups


def _rgb_sha(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()


def _scene_ids(root: Path, inventory: dict) -> dict[str, str]:
    """Join bases with identical stored or detector-oriented RGB originals."""
    bases = {_base(sid) for sid in inventory}
    parent = {base: base for base in bases}

    def representative(base: str) -> str:
        while parent[base] != base:
            parent[base] = parent[parent[base]]
            base = parent[base]
        return base

    original_owner = {}
    originals_by_base = {}
    for sid, group in sorted(inventory.items()):
        base = _base(sid)
        original_path = _inside(root, group["O"])
        _enforce_decode_limits(original_path)
        with Image.open(original_path) as image:
            pixel_hashes = {_rgb_sha(image)}
            if image.getexif().get(ExifTags.Base.Orientation, 1) != 1:
                # safe_open normalizes orientation before exposing pixels to a detector.
                with ImageOps.exif_transpose(image) as oriented:
                    pixel_hashes.add(_rgb_sha(oriented))
        for pixel_sha in sorted(pixel_hashes):
            other = original_owner.setdefault(pixel_sha, base)
            parent[representative(base)] = representative(other)
        originals_by_base.setdefault(base, set()).update(pixel_hashes)

    components = {}
    for base, hashes in originals_by_base.items():
        components.setdefault(representative(base), set()).update(hashes)
    ids = {base: digest(sorted(hashes)) for base, hashes in components.items()}
    return {base: ids[representative(base)] for base in originals_by_base}


def _base_partitions(scenes, exposed, prior):
    prior_dev = {row["base_image_id"] for row in prior["rows"] if row["split"] == "dev"} if prior else set()
    exposed_scenes = {scenes[base] for base in exposed | prior_dev if base in scenes}
    first_base = {}
    for base, scene in sorted(scenes.items()):
        first_base.setdefault(scene, base)
    return {base: ("dev" if base in prior_dev else
                   "diagnostic" if scene in exposed_scenes else
                   "dev" if not prior and split_of(first_base[scene]) == "dev" else "locked")
            for base, scene in scenes.items()}


def _validate_prior_manifest(prior, manifest):
    """Tie observed v1 rows to the exact same inventory and eligibility."""
    if not isinstance(prior, dict) or prior.get("protocol") != 1:
        raise ValueError("prior manifest must be a sealed protocol v1 manifest")
    payload = {k: v for k, v in prior.items() if k != "sha256"}
    if prior.get("sha256") != digest(payload):
        raise ValueError("prior manifest checksum mismatch")
    if any(prior[field] != manifest[field] for field in ("provenance_sha256", "files", "exclusions")):
        raise ValueError("prior manifest inventory, provenance or eligibility differs")
    exposed = set(prior["exposed_bases"])
    if not EXPOSED.issubset(exposed) or not exposed.issubset(manifest["exposed_bases"]):
        raise ValueError("prior manifest exposure inventory was narrowed")
    expected = []
    for row in manifest["rows"]:
        base = row["base_image_id"]
        legacy = {key: value for key, value in row.items() if key != "scene_id"}
        legacy["split"] = ("diagnostic" if base in exposed else
                           "dev" if split_of(base) == "dev" else "locked")
        expected.append(legacy)
    if sorted(prior["rows"], key=lambda row: row["image_path"]) != sorted(
            expected, key=lambda row: row["image_path"]):
        raise ValueError("prior manifest row accounting or legacy split differs")


def freeze(root: Path, out: Path, *, seed: int = 42,
           exposed=(), prior_manifest: Path | None = None, repo: Path = REPO) -> dict:
    """Audit provenance and construct a split isolated by original scene."""
    root, repo = root.absolute(), repo.absolute()
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be a nonnegative signed 32-bit integer")
    _plain_path(root)
    if not root.resolve().is_relative_to(repo.resolve()):
        raise ValueError("freeze root must be inside the repository")
    root, repo = root.resolve(), repo.resolve()
    provenance_path = _inside(root, "provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    records = provenance.get("files", [])
    files = {}
    for record in records:
        name = record["file"]
        if name in files or not _SUFFIX.fullmatch(name):
            raise ValueError("duplicate or unsupported provenance record")
        path = _inside(root, name)
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"provenance hash drift: {name}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"provenance size drift: {name}")
        files[name] = record["sha256"]
    actual = {p.name for p in root.iterdir() if _SUFFIX.fullmatch(p.name)}
    if not files or actual != set(files):
        raise ValueError("asset inventory differs from provenance")
    scenes = _scene_ids(root, _inventory(files))
    _plain_path(provenance_path)
    adapter = CoMoFoDAdapter(root)
    samples = list(adapter.samples("dev")) + list(adapter.samples("test"))
    pairs = {}
    for sample in samples:
        sid = sample.base_image_id
        label_rows = pairs.setdefault(sid, {})
        if sample.label in label_rows:
            raise ValueError("duplicate sample label")
        label_rows[sample.label] = sample
    forged_ids = {_SUFFIX.fullmatch(name)[1] for name in files
                  if _SUFFIX.fullmatch(name)[2].upper() == "F"}
    if forged_ids != set(pairs):
        raise ValueError("forged image missing discoverable ground truth")
    prior = _load_sealed(prior_manifest) if prior_manifest is not None else None
    if prior is not None and prior.get("protocol") != 1:
        raise ValueError("prior manifest must be a sealed protocol v1 manifest")
    exposed_ids = EXPOSED | {_base(s) for s in exposed}
    if prior is not None:
        exposed_ids |= set(prior["exposed_bases"])
    partitions = _base_partitions(scenes, exposed_ids, prior)
    exclusions, rows = [], []
    rejected = {}
    for sid, pair in sorted(pairs.items()):
        base = _base(sid)
        if set(pair) != {"forged", "authentic"}:
            raise ValueError("every forged row needs a paired original")
        f, o = pair["forged"], pair["authentic"]
        if f.gt_mask_path.name != f"{sid}_B{f.gt_mask_path.suffix}":
            raise ValueError("strict protocol requires the binary B mask")
        reason = _strict_pair(f.image_path, o.image_path, f.gt_mask_path)
        if base == "099":
            reason = "099_mask_convention_unresolved"
        if reason:
            rejected[base] = reason
    for sid, pair in sorted(pairs.items()):
        base = _base(sid)
        if base in rejected:
            exclusions.append({"sample_id": sid, "base_image_id": base,
                               "reason": rejected[base], "n_images": 2})
            continue
        for label in ("authentic", "forged"):
            sample = pair[label]
            rows.append({"image_path": sample.image_path.name,
                         "gt_mask_path": sample.gt_mask_path.name
                         if sample.gt_mask_path else None,
                         "label": label, "category": sample.category,
                         "sample_id": sid, "base_image_id": base,
                         "scene_id": scenes[base], "split": partitions[base]})
    manifest = _sealed({"protocol": PROTOCOL, "seed": seed,
                        "dataset_root": root.relative_to(repo).as_posix(),
                        "adapter_version": adapter.version(),
                        "provenance_sha256": file_hash(provenance_path),
                        "files": files, "exposed_bases": sorted(exposed_ids),
                        "prior_manifest": prior, "exclusions": exclusions, "rows": rows})
    validate_manifest(manifest, root)
    _write(out, manifest)
    return manifest


def validate_manifest(manifest: dict, root: Path) -> None:
    if manifest.get("protocol") == 1:
        raise HistoricalProtocolError("Protocol v1 manifest is historical. Re-freeze with --prior-manifest.")
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("protocol") != PROTOCOL or manifest.get("sha256") != digest(payload):
        raise ValueError("invalid manifest identity")
    kind = manifest.get("dataset_kind")
    if kind == casia_detector_protocol.DATASET_KIND:
        casia_detector_protocol.validate_manifest(manifest, root)
        return
    if kind is not None:
        raise ValueError("unknown dataset kind")
    if not EXPOSED.issubset(manifest["exposed_bases"]):
        raise ValueError("known exposure inventory was narrowed")
    _plain_path(root)
    if file_hash(_inside(root, "provenance.json")) != manifest["provenance_sha256"]:
        raise ValueError("provenance changed after freeze")
    files = manifest["files"]
    inventory = _inventory(files)
    if {p.name for p in root.iterdir() if _SUFFIX.fullmatch(p.name)} != set(files):
        raise ValueError("asset inventory changed after freeze")
    for name, checksum in files.items():
        if file_hash(_inside(root, name)) != checksum:
            raise ValueError(f"asset changed after freeze: {name}")
    scenes = _scene_ids(root, inventory)
    prior = manifest["prior_manifest"]
    if prior is not None:
        _validate_prior_manifest(prior, manifest)
    partitions = _base_partitions(scenes, set(manifest["exposed_bases"]), prior)
    seen, groups = set(), {}
    for row in manifest["rows"]:
        name, label, base = row["image_path"], row["label"], row["base_image_id"]
        if name in seen or name not in files or label not in {"authentic", "forged"}:
            raise ValueError("duplicate, unknown or invalid manifest row")
        seen.add(name)
        if base != _base(row["sample_id"]):
            raise ValueError("derivative base ID mismatch")
        if (base == "099" or row["split"] != partitions.get(base)
                or row["scene_id"] != scenes.get(base)):
            raise ValueError("exposed/excluded base or scene split leakage")
        pair = groups.setdefault(row["sample_id"], {})
        if label in pair:
            raise ValueError("duplicate paired label")
        pair[label] = row
        suffix = _SUFFIX.fullmatch(name)
        if not suffix or suffix[1] != row["sample_id"] or suffix[2].upper() != (
                "F" if label == "forged" else "O"):
            raise ValueError("image label/path mismatch")
        gt = row["gt_mask_path"]
        if (label == "authentic" and gt is not None) or (label == "forged" and (
                gt not in files or gt != f"{row['sample_id']}_B{Path(gt).suffix}")):
            raise ValueError("ground truth label/path mismatch")
    for pair in groups.values():
        if set(pair) != {"authentic", "forged"}:
            raise ValueError("incomplete paired labels")
        f, o = pair["forged"], pair["authentic"]
        if _strict_pair(_inside(root, f["image_path"]), _inside(root, o["image_path"]),
                        _inside(root, f["gt_mask_path"])):
            raise ValueError("invalid strict ground truth")
    excluded = {e["sample_id"] for e in manifest["exclusions"]}
    all_forged = {_SUFFIX.fullmatch(n)[1] for n in files
                  if _SUFFIX.fullmatch(n)[2].upper() == "F"}
    if set(groups) & excluded or set(groups) | excluded != all_forged:
        raise ValueError("missing or contradictory sample accounting")
    justified = {}
    for sid, group in inventory.items():
        reason = _strict_pair(_inside(root, group["F"]), _inside(root, group["O"]),
                              _inside(root, group["B"]))
        if _base(sid) == "099":
            reason = "099_mask_convention_unresolved"
        if reason:
            justified[_base(sid)] = reason
    expected_excluded = {sid for sid in inventory if _base(sid) in justified}
    if excluded != expected_excluded or len(excluded) != len(manifest["exclusions"]):
        raise ValueError("unjustified or duplicate exclusion")
    for excluded_row in manifest["exclusions"]:
        base = _base(excluded_row["sample_id"])
        if (excluded_row["base_image_id"] != base or excluded_row["n_images"] != 2
                or excluded_row["reason"] != justified[base]):
            raise ValueError("incorrect exclusion accounting")
    for partition in ("dev", "locked"):
        if {r["label"] for r in manifest["rows"] if r["split"] == partition} != {
                "authentic", "forged"}:
            raise ValueError(f"{partition} needs both classes")


def source_identity(repo: Path = REPO) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       timeout=10).decode("utf-8").strip()
    names = set(git("ls-files", "-z", "--", "src", "configs", "scripts", "tests",
                    "pyproject.toml").split("\0"))
    names.update(git("ls-files", "-z", "--others", "--exclude-standard", "--", "src",
                     "configs", "scripts", "tests").split("\0"))
    names.discard("")
    files = {name: file_hash(repo / name) if (repo / name).is_file() else "missing"
             for name in sorted(names)}
    return {"git_sha": git("rev-parse", "HEAD"),
            "working_tree_clean": not git("status", "--porcelain"),
            "files": files, "source_sha256": digest(files)}


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "platform": platform.system(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "opencv-python", "Pillow", "PyYAML")}}


def protocol_hash(repo: Path = REPO) -> str:
    paths = ["scripts/evaluate_detector_quality.py",
             "scripts/casia_detector_protocol.py",
             *[f"src/imageguard/evaluation/{name}.py"
               for name in ("runner", "metrics", "comofod", "datasets")],
             "src/imageguard/core/io.py", "src/imageguard/core/config.py"]
    return digest({name: file_hash(repo / name) for name in paths})


def own_checkout_imports() -> dict:
    """A shared editable venv must not substitute another checkout's detectors."""
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "imageguard" or name.startswith("imageguard."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to((REPO / "src").resolve()):
                raise ValueError("imageguard module imported from another checkout")
            paths[name] = Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    return paths


def _json_evidence(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _json_evidence(value.item())
    if isinstance(value, dict):
        return {str(k): _json_evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_evidence(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"type": type(value).__name__}


class ManifestAdapter:
    name = "comofod_protocol"

    def __init__(self, manifest: dict, root: Path, split: str):
        self.manifest, self.root, self.split = manifest, root, split
        self.name = manifest.get("dataset_kind", "comofod_protocol")
        self.current = None

    def version(self):
        return self.manifest["sha256"]

    def samples(self, split="dev"):
        if split != self.split:
            raise ValueError("adapter partition mismatch")
        for row in self.manifest["rows"]:
            if row["split"] == split:
                self.current = row
                yield Sample(self.root / row["image_path"],
                             self.root / row["gt_mask_path"] if row["gt_mask_path"] else None,
                             row["label"], row["category"], row["base_image_id"])
        self.current = None


class CaptureMethod:
    def __init__(self, method, adapter, records, out):
        self.method, self.adapter, self.records, self.out = method, adapter, records, out
        self.name = method.name

    def analyze(self, doc, cfg):
        row = self.adapter.current
        if (doc.file_name != Path(row["image_path"]).name
                or doc.sha256 != self.adapter.manifest["files"][row["image_path"]]):
            raise ValueError("runner document does not match the frozen row")
        record = {**row, "method": self.name, "status": "method_exception"}
        self.records.append(record)
        try:
            result = self.method.analyze(doc, cfg)
        except Exception as error:
            record["exception_type"] = type(error).__name__
            raise
        record.update({"status": result.verdict.value, "score": result.score,
                       "score_definition": result.score_definition,
                       "params_hash": result.params_hash, "duration_ms": result.duration_ms,
                       "limitations": result.limitations,
                       "evidence": _json_evidence(result.evidence),
                       "mask_present": result.mask is not None,
                       "mask_area_ratio": float(np.count_nonzero(result.mask) / result.mask.size)
                       if result.mask is not None and result.mask.size else 0.0})
        if result.mask is not None:
            if result.mask.shape != (doc.height, doc.width):
                record["status"] = "invalid_mask_geometry"
                raise ValueError("method returned invalid mask geometry")
            name = f"masks/{row['sample_id']}-{row['label']}-{self.name}.png"
            path = _inside(self.out, name)
            path.parent.mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", (result.mask > 0).astype(np.uint8) * 255)
            if not ok:
                raise ValueError("mask artifact encoding failed")
            path.write_bytes(encoded.tobytes())
            record.update({"mask_artifact": name, "mask_sha256": file_hash(path)})
        if row["gt_mask_path"]:
            with Image.open(self.adapter.root / row["gt_mask_path"]) as image:
                gt = np.asarray(image.convert("L"))
            record["pixel"] = pixel_metrics(result.mask, gt)
        return result


def _cohort_ledger_claimed(recorded: dict, manifest_sha256: str,
                           protocol_sha256: str, event_id: str) -> None:
    if (recorded.get("manifest_sha256") != manifest_sha256
            or recorded.get("protocol_sha256") != protocol_sha256):
        raise ValueError("locked cohort consumption ledger does not match this cohort")
    if recorded.get("event_id") != event_id:
        raise ValueError("locked cohort already consumed under a different event_id")


def _claim_locked_cohort(cohort_root: Path, *, manifest_sha256: str,
                         protocol_sha256: str, event_id: str) -> None:
    """Record one locked consumption per sealed manifest + protocol identity."""
    _plain_path(cohort_root)
    if not cohort_root.is_dir():
        raise ValueError("locked cohort root is missing or inaccessible")
    ledger_path = cohort_root / _COHORT_LEDGER
    identity = {"manifest_sha256": manifest_sha256,
                "protocol_sha256": protocol_sha256, "event_id": event_id}
    if ledger_path.is_file():
        try:
            recorded = _load_sealed(ledger_path)
        except ValueError as err:
            raise ValueError("locked cohort consumption ledger is invalid or corrupt") from err
        _cohort_ledger_claimed(recorded, manifest_sha256, protocol_sha256, event_id)
        return
    try:
        _write(ledger_path, _sealed(identity))
    except FileExistsError:
        recorded = _load_sealed(ledger_path)
        _cohort_ledger_claimed(recorded, manifest_sha256, protocol_sha256, event_id)
    except OSError as err:
        raise ValueError("locked cohort consumption ledger cannot be written") from err


def _lock_pair(contract_path, role, identity, manifest, config_sha, methods, protocol,
               cohort_root: Path):
    if contract_path is None or role not in {"baseline", "candidate"}:
        raise ValueError("locked inference requires --pair-contract and --role")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for paired_role in ("baseline", "candidate"):
        frozen = contract.get(paired_role) if isinstance(contract, dict) else None
        if (not isinstance(frozen, dict) or set(frozen) != {"source_sha256", "config_sha256"}
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
                       for value in frozen.values())):
            raise ValueError("complete frozen pair requires both source/config identities")
    if (contract.get("approved") is not True or not identity["working_tree_clean"]
            or contract.get("manifest_sha256") != manifest["sha256"]
            or contract.get("protocol_sha256") != protocol
            or contract.get("seed") != manifest["seed"] or contract.get("methods") != methods
            or contract.get(role) != {"source_sha256": identity["source_sha256"],
                                      "config_sha256": config_sha}):
        raise ValueError("frozen pair contract does not match this run")
    criteria_hash = contract.get("criteria_sha256")
    if not isinstance(criteria_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", criteria_hash):
        raise ValueError("locked event requires preregistered criteria hash")
    if manifest.get("dataset_kind") == casia_detector_protocol.DATASET_KIND:
        if (contract.get("cohort_proposal_sha256") != manifest["cohort_receipt"]["raw_sha256"]
                or contract.get("oracle_policy") != manifest["oracle_policy"]
                or contract.get("row_inventory_sha256") != digest(manifest["rows"])):
            raise ValueError("CASIA lock requires the approved cohort and oracle")
        _check_population_rows(manifest["rows"], contract.get("population"))
    event = contract.get("event_id", "")
    if not isinstance(event, str) or not re.fullmatch(r"[a-zA-Z0-9-]{1,80}", event):
        raise ValueError("invalid locked event ID")
    _claim_locked_cohort(cohort_root, manifest_sha256=manifest["sha256"],
                         protocol_sha256=protocol, event_id=event)
    _write(contract_path.parent / f"event-{event}" / f"{role}.json",
           {"pair_sha256": file_hash(contract_path), "role": role})
    return file_hash(contract_path), contract["criteria_sha256"]


def evaluate(manifest_path: Path, split: str, config: Path, methods: list,
             out: Path, *, root: Path | None = None, pair_contract=None,
             role=None, repo: Path = REPO) -> dict:
    manifest = _load_sealed(manifest_path)
    root = root or _inside(repo, manifest["dataset_root"])
    validate_manifest(manifest, root)
    if split not in {"diagnostic", "dev", "locked"}:
        raise ValueError("unknown split")
    names = [m.name for m in methods]
    if not names or len(set(names)) != len(names):
        raise ValueError("methods must be nonempty and unique")
    imports = own_checkout_imports()
    identity, runtime = source_identity(repo), runtime_identity()
    identity["imports"] = imports
    config_sha = file_hash(config)
    cfg = load_thresholds(config)
    protocol = protocol_hash(repo)
    _plain_path(out)
    out.mkdir(parents=True, exist_ok=False)
    pair_sha, criteria_sha = (_lock_pair(pair_contract, role, identity, manifest,
                                       config_sha, names, protocol, root)
                              if split == "locked" else (None, None))
    adapter, records = ManifestAdapter(manifest, root, split), []
    wrapped = [CaptureMethod(m, adapter, records, out) for m in methods]
    np.random.seed(manifest["seed"])
    cv2.setRNGSeed(manifest["seed"])
    report = run_eval(adapter, wrapped, cfg, split, manifest["seed"])
    if any(r["status"] == "invalid_mask_geometry" for r in records):
        _write(out / "failure.json", {"status": "invalid_mask_geometry", "rows": records})
        raise ValueError("invalid predicted mask geometry; evaluation rejected")
    selected = [r for r in manifest["rows"] if r["split"] == split]
    keys = {(r["image_path"], r["method"]) for r in records}
    for row in selected:
        for method in names:
            if (row["image_path"], method) not in keys:
                records.append({**row, "method": method, "status": "input_excluded"})
    counts = {}
    for method in names:
        traces = [r for r in records if r["method"] == method]
        metrics = report.per_method[method]
        if len(traces) != len(selected) or len({r["image_path"] for r in traces}) != len(traces):
            raise ValueError("incomplete/duplicate method accounting")
        count = {status: sum(r["status"] == status for r in traces)
                 for status in sorted({r["status"] for r in traces})}
        if metrics["n_scored"] + metrics["n_excluded"] + metrics["n_error"] + metrics[
                "n_not_applicable"] != len(selected):
            raise ValueError("aggregate accounting mismatch")
        counts[method] = count
    result = _sealed({"protocol": PROTOCOL, "protocol_sha256": protocol,
                      "manifest_sha256": manifest["sha256"], "seed": manifest["seed"],
                      "split": split, "methods": names, "runtime": runtime,
                      "source": identity, "config_sha256": config_sha,
                      "config_semantic_hash": cfg.snapshot_hash(),
                      "pair_sha256": pair_sha, "criteria_sha256": criteria_sha, "role": role,
                      "n_selected": len(selected), "counts": counts,
                      "eval": asdict(report), "rows": records})
    _write(out / "report.json", result)
    return result


def _check_population_rows(rows, population):
    """A fixed CASIA event cannot replace failures or narrow class/group N."""
    if (not isinstance(population, dict)
            or set(population) != {"authentic", "forged", "distinct_screened_originals"}
            or any(type(n) is not int or n < 1 for n in population.values())
            or len(set(population.values())) != 1):
        raise ValueError("formal population requires equal positive class/group counts")
    groups = {}
    for label in ("authentic", "forged"):
        selected = [row for row in rows if row["label"] == label]
        groups[label] = {row["base_image_id"] for row in selected}
        if (len(selected) != population[label]
                or len(groups[label]) != population["distinct_screened_originals"]
                or any(row["scene_id"] != row["base_image_id"] or row["split"] != "locked"
                       for row in selected)):
            raise ValueError("formal population or original-ID groups changed")
    if (groups["authentic"] != groups["forged"] or len(rows) != sum(
            population[label] for label in ("authentic", "forged"))):
        raise ValueError("formal population needs exactly one complete pair per group")


def _check_formal_population(report, criteria):
    population = criteria.get("population")
    if population is None:
        return
    for method in report["methods"]:
        rows = [row for row in report["rows"] if row["method"] == method]
        _check_population_rows(rows, population)
        if any(row["status"] not in _SCORED_STATUS for row in rows):
            raise ValueError("formal population requires every row scored with zero exclusions/errors")
        inventory = criteria.get("row_inventory")
        if not inventory or not isinstance(inventory, list):
            raise ValueError("formal population requires preregistered row inventory")
        _check_population_rows(inventory, population)
        expected = {row["image_path"]: row for row in inventory}
        if (len(expected) != len(inventory)
                or {row["image_path"] for row in rows} != set(expected)
                or any(any(row.get(key) != value for key, value in expected[row["image_path"]].items())
                       for row in rows)):
            raise ValueError("formal row inventory differs from preregistered image/group mapping")


def _rates(report, method):
    metric = report["eval"]["per_method"][method]
    controls = metric["fp"] + metric["tn"]
    traces = [r for r in report["rows"] if r["method"] == method]
    pixel = _pixel_summary(traces)
    return {"fp": metric["fp"], "tp": metric["tp"],
            "fpr": metric["fp"] / controls if controls else None,
            "recall": metric["recall"],
            "abstention": metric["n_inconclusive"] / report["n_selected"],
            **{f"{label}_inconclusive": sum(r["label"] == label and r["status"] ==
                                             Verdict.INCONCLUSIVE.value for r in traces)
               for label in ("authentic", "forged")},
            "macro_iou": pixel["macro"]["iou"],
            "micro_iou": pixel["micro"]["iou"]}


def _pixel_summary(rows):
    pixel_rows = []
    for row in rows:
        if row["label"] != "forged" or row["status"] not in _SCORED_STATUS:
            continue
        if type(row.get("mask_present")) is not bool:
            raise ValueError("scored forged row requires a boolean mask presence flag")
        pixel = row.get("pixel")
        if not isinstance(pixel, dict) or pixel.get("skipped") is not None:
            raise ValueError("scored forged row is missing valid pixel evidence")
        counts = [pixel.get(name) for name in ("tp", "fp", "fn")]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("scored forged pixel counts must be nonnegative integers")
        tp, fp, fn = counts
        if tp + fn == 0:
            raise ValueError("scored forged pixel evidence requires nonempty ground truth")
        expected_iou = tp / (tp + fp + fn)
        row_iou = pixel.get("iou")
        if (type(row_iou) not in (int, float) or not 0 <= row_iou <= 1
                or abs(row_iou - expected_iou) > _PIXEL_REPORT_TOLERANCE):
            raise ValueError("scored forged row IoU disagrees with pixel counts")
        if not row["mask_present"] and (tp != 0 or fp != 0):
            raise ValueError("missing prediction must retain zero-IoU R2 pixel evidence")
        pixel_rows.append({"skipped": None, "precision": None, "recall": None,
                           "f1": None, "iou": expected_iou,
                           "tp": tp, "fp": fp, "fn": fn})
    return aggregate_pixel(pixel_rows)


def _check_report(report):
    methods = report["methods"]
    if (not methods or len(set(methods)) != len(methods)
            or set(report["counts"]) != set(methods)
            or set(report["eval"]["per_method"]) != set(methods)
            or any(r["method"] not in methods for r in report["rows"])):
        raise ValueError("invalid report method inventory")
    coverage = {}
    for method in report["methods"]:
        rows = [r for r in report["rows"] if r["method"] == method]
        if (len(rows) != report["n_selected"] or not rows
                or len({r["image_path"] for r in rows}) != len(rows)):
            raise ValueError("incomplete/duplicate report rows")
        if {r["label"] for r in rows} != {"authentic", "forged"}:
            raise ValueError("paired evaluation needs both classes")
        if any(r["status"] not in _SCORED_STATUS | _UNSCORED_STATUS for r in rows):
            raise ValueError("unknown report status")
        histogram = {status: sum(r["status"] == status for r in rows)
                     for status in {r["status"] for r in rows}}
        if (any(type(count) is not int or count < 0
                for count in report["counts"][method].values())
                or report["counts"][method] != histogram):
            raise ValueError("report status histogram mismatch")
        scored = [r for r in rows if r["status"] in _SCORED_STATUS]
        expected = image_metrics([r["status"] == Verdict.SUSPICIOUS_INDICATION.value
                                  for r in scored], [r["label"] == "forged" for r in scored])
        aggregate = report["eval"]["per_method"][method]
        if (not isinstance(aggregate, dict)
                or any(type(aggregate[k]) is not int or aggregate[k] != expected[k]
                       for k in ("tp", "fp", "tn", "fn", "n"))):
            raise ValueError("row/aggregate confusion counts disagree")
        expected_counts = {"n_scored": len(scored),
                           "n_excluded": histogram.get("method_exception", 0)
                           + histogram.get("input_excluded", 0),
                           "n_inconclusive": histogram.get(Verdict.INCONCLUSIVE.value, 0),
                           "n_error": histogram.get(Verdict.ERROR.value, 0),
                           "n_not_applicable": histogram.get(Verdict.NOT_APPLICABLE.value, 0)}
        if any(type(aggregate.get(k)) is not int or aggregate[k] != count
               for k, count in expected_counts.items()):
            raise ValueError("row/aggregate status accounting mismatch")
        expected_pixel = _pixel_summary(scored)
        if (type(aggregate.get("pixel_n_images")) is not int
                or aggregate["pixel_n_images"] != expected_pixel["n_images_scored"]):
            raise ValueError("pixel_n_images disagrees with scored forged rows")
        for field, expected_iou in (("pixel_macro", expected_pixel["macro"]["iou"]),
                                    ("pixel_micro", expected_pixel["micro"]["iou"])):
            pixel_aggregate = aggregate.get(field)
            if not isinstance(pixel_aggregate, dict) or "iou" not in pixel_aggregate:
                raise ValueError("pixel aggregate IoU is malformed")
            reported_iou = pixel_aggregate["iou"]
            if expected_iou is None:
                if reported_iou is not None:
                    raise ValueError("pixel aggregate IoU disagrees with scored forged rows")
            elif (type(reported_iou) not in (int, float) or not 0 <= reported_iou <= 1
                    or not math.isfinite(reported_iou)
                    or abs(reported_iou - expected_iou) > _PIXEL_REPORT_TOLERANCE):
                raise ValueError("pixel aggregate IoU disagrees with scored forged rows")
        coverage[method] = {label: {"eligible": sum(r["label"] == label for r in rows),
                                    "scored": sum(r["label"] == label for r in scored)}
                            for label in ("authentic", "forged")}
    return coverage


def compare(baseline_path: Path, candidate_path: Path, criteria_path: Path, out: Path):
    baseline = _load_sealed(baseline_path / "report.json")
    candidate = _load_sealed(candidate_path / "report.json")
    versions = {baseline.get("protocol"), candidate.get("protocol")}
    if 1 in versions:
        raise HistoricalProtocolError("Protocol v1 reports are historical and cannot be compared.")
    if versions != {PROTOCOL}:
        raise ValueError("unsupported report protocol")
    baseline_coverage = _check_report(baseline)
    candidate_coverage = _check_report(candidate)
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    _check_formal_population(baseline, criteria)
    _check_formal_population(candidate, criteria)
    for field in ("protocol", "protocol_sha256", "manifest_sha256", "seed", "methods",
                  "runtime", "n_selected", "pair_sha256", "criteria_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"baseline/candidate mismatch: {field}")
    if (baseline["split"] != "locked" or candidate["split"] != "locked"
            or not baseline["pair_sha256"] or baseline["role"] != "baseline"
            or candidate["role"] != "candidate" or criteria.get("approved") is not True
            or baseline["criteria_sha256"] != file_hash(criteria_path)
            or not criteria.get("approved_by")
            or set(criteria.get("methods", {})) != set(baseline["methods"])):
        raise ValueError("incomplete locked pair or unapproved criteria")
    def row_key(row):
        return (row["image_path"], row["base_image_id"], row["scene_id"], row["label"], row["method"])

    if sorted(map(row_key, baseline["rows"])) != sorted(map(row_key, candidate["rows"])):
        raise ValueError("paired rows differ")
    primary = criteria.get("primary", {})
    if (set(primary) != {"method", "objective", "minimum"}
            or primary["method"] not in baseline["methods"]
            or primary["objective"] not in {"fp_reduction", "tp_increase"}
            or type(primary["minimum"]) is not int or primary["minimum"] < 1):
        raise ValueError("explicit primary method/objective/correction count required")
    tolerance = criteria.get("tolerance")
    if not isinstance(tolerance, (float, int)) or not 0 <= tolerance <= 1e-9:
        raise ValueError("explicit numerical tolerance must be at most 1e-9")
    findings = {}
    for method in baseline["methods"]:
        targets = criteria["methods"][method]
        if set(targets) != _CRITERIA or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in targets.values()):
            raise ValueError("all numerical method criteria must be explicitly supplied")
        before, after = _rates(baseline, method), _rates(candidate, method)
        failures = []
        before_rows = {row_key(r): r for r in baseline["rows"] if r["method"] == method}
        after_rows = {row_key(r): r for r in candidate["rows"] if r["method"] == method}
        # A recovered failure is allowed; moving it to a different sample or
        # changing its kind introduces a new failure and must not hide in totals.
        if any(row["status"] in _UNSCORED_STATUS
               and row["status"] != before_rows[key]["status"]
               for key, row in after_rows.items()):
            failures.append("coverage:new_failed_rows")
        for label in ("authentic", "forged"):
            prior = baseline_coverage[method][label]
            current = candidate_coverage[method][label]
            if current["eligible"] != prior["eligible"] or current["scored"] < prior["scored"]:
                failures.append(f"coverage:{label}_denominator")
        for key, threshold in targets.items():
            rate, bound = key.rsplit("_", 1)
            rate = rate.removesuffix("_delta")
            actual = after[rate]
            if actual is None or before[rate] is None:
                failures.append(key)
                continue
            actual -= before[rate]
            if ((bound == "max" and actual > threshold + tolerance)
                    or (bound == "min" and actual < threshold - tolerance)):
                failures.append(key)
        for name in ("n_excluded", "n_error", "n_not_applicable"):
            if candidate["eval"]["per_method"][method][name] > baseline["eval"][
                    "per_method"][method][name]:
                failures.append(f"coverage:{name}")
        for status in ("method_exception", "input_excluded"):
            if candidate["counts"][method].get(status, 0) > baseline["counts"][method].get(status, 0):
                failures.append(f"coverage:{status}")
        if method == primary["method"]:
            gain = (before["fp"] - after["fp"] if primary["objective"] == "fp_reduction"
                    else after["tp"] - before["tp"])
            if gain < primary["minimum"]:
                failures.append("primary_correction_count")
        findings[method] = {"baseline": before, "candidate": after,
                            "coverage": {"baseline": baseline_coverage[method],
                                         "candidate": candidate_coverage[method]},
                            "failed": failures}
    result = _sealed({"protocol": PROTOCOL, "baseline_sha256": baseline["sha256"],
                      "candidate_sha256": candidate["sha256"],
                      "criteria_sha256": file_hash(criteria_path),
                      "status": "FAIL" if any(v["failed"] for v in findings.values()) else "PASS",
                      "methods": findings})
    _write(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exposed", nargs="*", default=[])
    p.add_argument("--prior-manifest", type=Path,
                   help="Preserve observed dev rows from a sealed protocol v1 manifest")
    p = commands.add_parser("freeze-casia", help="Freeze the approved native CASIA cohort without inference")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--split", choices=["diagnostic", "dev", "locked"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--methods", default="keypoint,block")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair-contract", type=Path)
    p.add_argument("--role", choices=["baseline", "candidate"])
    p = commands.add_parser("compare")
    for name in ("baseline", "candidate", "criteria", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze(args.root, args.out, seed=args.seed, exposed=args.exposed,
                            prior_manifest=args.prior_manifest)
        elif args.command == "freeze-casia":
            result = casia_detector_protocol.build_manifest(args.cohort, args.root, repo=REPO)
            _write(args.out, result)
        elif args.command == "evaluate":
            result = evaluate(args.manifest, args.split, args.config,
                              [load_method(name) for name in args.methods.split(",")], args.out,
                              root=args.root, pair_contract=args.pair_contract, role=args.role)
        else:
            result = compare(args.baseline, args.candidate, args.criteria, args.out)
    except HistoricalProtocolError as error:
        parser.exit(2, f"{error}\n")
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Protocol rejected request ({type(error).__name__}); inspect inputs locally.\n")
    print(json.dumps({"command": args.command, "sha256": result["sha256"],
                      "status": result.get("status", "PASS")}))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
