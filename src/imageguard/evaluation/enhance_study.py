"""Run controlled enhancement studies against synthetic image pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.evaluation.datasets import SynthAdapter
from imageguard.evaluation.enhance_metrics import align_mask, iou, psnr, ssim
from imageguard.evaluation.enhance_ops import degrade, restore, wrong_kernel
from imageguard.forensics.registry import load_method

try:
    from imageguard.evaluation.runner import _git_snapshot as _runner_git_snapshot
except ImportError:  # pragma: no cover - exercised only in a partial install
    _runner_git_snapshot = None


READING_RULE = (
    "PSNR/SSIM gain never establishes a detection gain; negative results are successful"
)
DEFAULT_DEGRADATIONS = ("gaussian", "disk", "motion", "awgn", "jpeg", "downsample")
DEFAULT_RESTORERS = (
    "unsharp",
    "clahe",
    "nlm",
    "richardson_lucy",
    "wiener",
)
DEFAULT_METHODS = ("keypoint", "block", "ela", "noise")
BLUR_DEGRADATIONS = frozenset(("gaussian", "disk", "motion"))
KERNEL_RESTORERS = frozenset(("richardson_lucy", "wiener"))

DEGRADATION_PARAMS: dict[str, dict[str, Any]] = {
    "gaussian": {"ksize": 7, "sigma": 1.4},
    "disk": {"radius": 4},
    "motion": {"length": 9, "angle_deg": 32.0},
    "awgn": {"sigma": 12.0},
    "jpeg": {"quality": 55},
    "downsample": {"factor": 2},
}
RESTORER_PARAMS: dict[str, dict[str, Any]] = {
    "unsharp": {"ksize": 5, "sigma": 1.0, "amount": 0.7},
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "nlm": {
        "h": 3.0,
        "hColor": 3.0,
        "templateWindowSize": 7,
        "searchWindowSize": 21,
    },
    "richardson_lucy": {"iterations": 10},
    "wiener": {"nsr": 0.01},
}
_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _names(
    value: Iterable[str] | str | None,
    default: tuple[str, ...],
    allowed: Iterable[str],
    label: str,
) -> list[str]:
    if value is None:
        selected = list(default)
    elif isinstance(value, str):
        selected = [part.strip() for part in value.split(",") if part.strip()]
    else:
        selected = [str(part).strip() for part in value if str(part).strip()]

    supported = set(allowed)
    unknown = [name for name in selected if name not in supported]
    if unknown:
        raise ValueError(f"unsupported {label}: {', '.join(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label} must not contain duplicates")
    return selected


def _sample_id(entry: dict[str, Any]) -> str:
    value = entry.get("id")
    if not isinstance(value, str) or not _SAMPLE_ID.fullmatch(value):
        raise ValueError("manifest sample id must be a simple filename component")
    if value in (".", ".."):
        raise ValueError("manifest sample id must not be a path component")
    return value


def _find_image(set_dir: Path, sample_id: str, suffix: str, preferred: str) -> Path:
    extensions = (preferred, ".png", ".jpg", ".jpeg")
    seen: set[str] = set()
    for extension in extensions:
        if extension in seen:
            continue
        seen.add(extension)
        candidate = set_dir / f"{sample_id}_{suffix}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {suffix} image for sample {sample_id}")


def _read_image(path: Path, flags: int) -> np.ndarray:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"could not decode {path.name}")
    return image


def _as_json_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _image_document(
    image: np.ndarray, file_name: str, image_format: str
) -> ImageDocument:
    array = np.ascontiguousarray(image.copy())
    data = array.tobytes()
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else int(array.shape[2])
    return ImageDocument(
        array=array,
        width=width,
        height=height,
        channels=channels,
        image_format=image_format,
        sha256=hashlib.sha256(data).hexdigest(),
        file_bytes=len(data),
        file_name=file_name,
    )


def _alignment(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, dict]:
    source_shape = tuple(int(value) for value in mask.shape[:2])
    evaluated_shape = tuple(int(value) for value in image.shape[:2])
    aligned = align_mask(mask, image.shape)
    source_height, source_width = source_shape
    evaluated_height, evaluated_width = evaluated_shape
    factors = {
        "y": evaluated_height / source_height,
        "x": evaluated_width / source_width,
    }
    return aligned, {
        "source_shape": list(source_shape),
        "evaluated_shape": list(evaluated_shape),
        "scale_factors": factors,
    }


def _method_results(
    image: np.ndarray,
    sample_id: str,
    role: str,
    method_specs: list[str],
    cfg: Any,
    gt_mask: np.ndarray | None,
    image_format: str,
) -> tuple[dict[str, dict], dict | None, int]:
    aligned_gt: np.ndarray | None = None
    alignment: dict | None = None
    if gt_mask is not None:
        aligned_gt, alignment = _alignment(gt_mask, image)

    results: dict[str, dict] = {}
    n_align_failed = 0
    for spec in method_specs:
        mask: np.ndarray | None = None
        error_type: str | None = None
        try:
            method = load_method(spec)
            result: AlgorithmResult = method.analyze(
                _image_document(image, f"{sample_id}_{role}.png", image_format),
                cfg,
            )
            verdict = str(getattr(result.verdict, "value", result.verdict))
            score = _as_json_number(result.score)
            mask = result.mask
        except Exception as exc:  # noqa: BLE001 - isolate detector failures per cell
            verdict = Verdict.ERROR.value
            score = None
            error_type = type(exc).__name__

        mask_present = mask is not None
        mask_nonempty = bool(mask_present and np.any(np.asarray(mask) > 0))
        if mask_present:
            mask_array = np.asarray(mask)
            mask_area_ratio = (
                float(np.count_nonzero(mask_array > 0) / mask_array.size)
                if mask_array.size
                else None
            )
        else:
            mask_array = None
            mask_area_ratio = None

        if gt_mask is None:
            overlap = None
        elif mask_array is None:
            overlap = 0.0
        elif aligned_gt is None or aligned_gt.shape != mask_array.shape:
            overlap = None
            n_align_failed += 1
        else:
            overlap = iou(mask_array, aligned_gt)

        cell = {
            "verdict": verdict,
            "score": score,
            "iou": overlap,
            "mask_present": mask_present,
            "mask_nonempty": mask_nonempty,
            "mask_area_ratio": mask_area_ratio,
        }
        if role == "matched_original":
            cell["suspicious_indication"] = (
                verdict == Verdict.SUSPICIOUS_INDICATION.value
            )
        if error_type is not None:
            cell["error_type"] = error_type
        results[spec] = cell
    return results, alignment, n_align_failed


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("OpenCV could not encode a study image")
    path.write_bytes(encoded.tobytes())


def _git_snapshot() -> tuple[str, bool | None]:
    if _runner_git_snapshot is not None:
        return _runner_git_snapshot()

    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).strip()
        clean = not subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).strip()
        return sha, clean
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def _arm_id(arm: dict[str, Any]) -> str:
    if arm["arm"] == "degradation":
        return f"degradation:{arm['degradation']}"
    return ":".join(
        (
            "restoration",
            arm["degradation"],
            arm["restorer"],
            arm["kernel_role"] or "single",
        )
    )


def _count_transitions(
    counts: dict,
    role: str,
    arm: dict[str, Any],
    sharp_methods: dict[str, dict],
    arm_methods: dict[str, dict],
) -> None:
    by_arm = counts.setdefault(role, {}).setdefault(_arm_id(arm), {})
    for method_name, result in arm_methods.items():
        if method_name not in sharp_methods:
            continue
        from_verdict = sharp_methods[method_name]["verdict"]
        to_verdict = result["verdict"]
        cell = by_arm.setdefault(method_name, {"n": 0, "changed": 0, "counts": {}})
        transition = f"{from_verdict}->{to_verdict}"
        cell["n"] += 1
        cell["changed"] += int(from_verdict != to_verdict)
        cell["counts"][transition] = cell["counts"].get(transition, 0) + 1


def _restoration_params(
    degradation: str,
    degradation_metadata: dict[str, Any],
    restorer: str,
    kernel_role: str,
) -> dict[str, Any]:
    params = deepcopy(RESTORER_PARAMS[restorer])
    kernel_params = dict(degradation_metadata)
    if kernel_role == "wrong":
        kernel = wrong_kernel(degradation, kernel_params)
    else:
        kernel = np.asarray(degradation_metadata["kernel"], dtype=np.float32)
    params.update(
        kernel=np.asarray(kernel, dtype=np.float32),
        kernel_role=kernel_role,
    )
    for name in ("wrong_sigma", "wrong_radius", "wrong_angle_deg"):
        if name in kernel_params:
            params[name] = kernel_params[name]
    return params


def _write_reports(
    payload: dict[str, Any], out_dir: Path, report_stem: str
) -> tuple[Path, Path]:
    json_path = out_dir / f"{report_stem}.json"
    markdown_path = out_dir / f"{report_stem}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _render_markdown(payload: dict[str, Any]) -> str:
    provenance = payload["provenance"]
    lines = [
        "# Classical Enhancement Study",
        "",
        READING_RULE,
        "",
        "## Scope",
        "",
        f"- Sample pairs (`n`): {payload['n']}",
        "- Roles: manipulated sample and matched original.",
        f"- Degradations: {', '.join(payload['degradations']) or 'none'}.",
        f"- Restorers: {', '.join(payload['restorers']) or 'none'}.",
        f"- Methods: {', '.join(payload['methods']) or 'none'}.",
        "- IoU ground truth is aligned once with nearest-neighbor interpolation.",
        "- Similarity metrics compare each changed arm with the sharp image of the same role.",
        "",
        "## Provenance",
        "",
        f"- Seed: `{provenance['seed']}`",
        f"- Threshold snapshot: `{provenance['config_hash']}`",
        f"- Git SHA: `{provenance['git_sha']}`",
        f"- Working tree clean: `{provenance['working_tree_clean']}`",
        f"- Dataset version: `{provenance['dataset_version']}`",
        f"- Alignment failures: `{payload['n_align_failed']}`",
        "",
        "## Verdict transitions",
        "",
    ]
    if payload["transition_counts"]:
        for role, arms in payload["transition_counts"].items():
            lines.extend((f"### {role.replace('_', ' ').title()}", ""))
            for arm_name, methods in arms.items():
                lines.append(f"- `{arm_name}`")
                for method_name, counts in methods.items():
                    lines.append(
                        f"  - `{method_name}`: n={counts['n']}, changed={counts['changed']}, "
                        f"transitions={json.dumps(counts['counts'], sort_keys=True)}"
                    )
            lines.append("")
    else:
        lines.extend(("No comparison arms were requested.", ""))

    lines.extend(
        (
            "## Status",
            "",
            "Complete: all manifest pairs were processed; the source set was read only.",
            "",
            "## Summary",
            "",
            f"Processed {payload['n']} sample pairs with {len(payload['methods'])} requested method(s).",
        )
    )
    return "\n".join(lines) + "\n"


def run_study(
    set_dir: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    degradations: Iterable[str] | str | None = None,
    restorers: Iterable[str] | str | None = None,
    methods: Iterable[str] | str | None = None,
    *,
    report_stem: str = "enhance_study",
) -> dict[str, Any]:
    """Evaluate requested operators without writing to the synthforge set."""
    source_dir = Path(set_dir).resolve()
    output_dir = Path(out_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError("synthforge set directory does not exist")
    manifest_path = source_dir / "meta.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise TypeError("meta.json must contain a list of sample entries")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if (
        not report_stem
        or Path(report_stem).name != report_stem
        or report_stem in (".", "..")
    ):
        raise ValueError("report_stem must be a simple filename stem")

    selected_degradations = _names(
        degradations, DEFAULT_DEGRADATIONS, DEGRADATION_PARAMS, "degradations"
    )
    selected_restorers = _names(
        restorers, DEFAULT_RESTORERS, RESTORER_PARAMS, "restorers"
    )
    selected_methods = _names(
        methods,
        DEFAULT_METHODS,
        (
            "keypoint",
            "block",
            "ela",
            "noise",
            "noise_local",
            "freq",
            "splicing",
        ),
        "methods",
    )

    entries: list[dict[str, Any]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise TypeError("each meta.json entry must be an object")
        _sample_id(entry)
        entries.append(entry)

    output_root = output_dir.resolve()
    report_root = output_root / report_stem
    json_path = output_root / f"{report_stem}.json"
    markdown_path = output_root / f"{report_stem}.md"
    for target in (report_root, json_path, markdown_path):
        resolved = target.resolve()
        if resolved == source_dir or source_dir in resolved.parents:
            raise ValueError("output paths must be outside the synthforge set")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_thresholds()
    adapter = SynthAdapter(source_dir)
    git_sha, working_tree_clean = _git_snapshot()
    transition_counts: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    n_align_failed = 0

    for sample_index, entry in enumerate(entries):
        sample_id = _sample_id(entry)
        preferred = ".jpg" if entry.get("format") == "jpeg" else ".png"
        edited_path = _find_image(source_dir, sample_id, "F", preferred)
        original_path = _find_image(source_dir, sample_id, "O", ".png")
        mask_path = source_dir / f"{sample_id}_M.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing M mask for sample {sample_id}")

        mask = _read_image(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        roles = (
            ("manipulated_sample", edited_path, mask.astype(np.uint8) * 255, 0),
            ("matched_original", original_path, None, 1),
        )
        for role, image_path, gt_mask, role_index in roles:
            sharp = _read_image(image_path, cv2.IMREAD_COLOR)
            sharp_format = (
                "jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "png"
            )
            sharp_methods, sharp_alignment, failed = _method_results(
                sharp,
                sample_id,
                "sharp",
                selected_methods,
                cfg,
                gt_mask,
                sharp_format,
            )
            n_align_failed += failed
            sample_record = {
                "id": sample_id,
                "role": role,
                "source_name": image_path.name,
                "image_shape": list(sharp.shape),
                "arms": [
                    {
                        "arm": "sharp",
                        "degradation": None,
                        "restorer": None,
                        "kernel_role": None,
                        "status": "RUN",
                        "reason": None,
                        "operator_params": {},
                        "image_path": None,
                        "image_shape": list(sharp.shape),
                        "metrics": None,
                        "gt_alignment": sharp_alignment,
                        "methods": sharp_methods,
                    }
                ],
            }

            for degradation_index, degradation_name in enumerate(selected_degradations):
                rng_seed = int(
                    np.random.SeedSequence(
                        [seed, sample_index, role_index, degradation_index]
                    ).generate_state(1)[0]
                )
                degraded, degradation_metadata = degrade(
                    sharp,
                    degradation_name,
                    deepcopy(DEGRADATION_PARAMS[degradation_name]),
                    rng=np.random.default_rng(rng_seed),
                )
                degraded_format = (
                    "jpeg"
                    if degradation_name == "jpeg" or sharp_format == "jpeg"
                    else "png"
                )
                degraded_relative = Path(report_stem) / "images" / sample_id / role
                degraded_relative /= f"{degradation_name}.png"
                _save_image(output_root / degraded_relative, degraded)
                degraded_methods, degraded_alignment, failed = _method_results(
                    degraded,
                    sample_id,
                    f"{role}_{degradation_name}",
                    selected_methods,
                    cfg,
                    gt_mask,
                    degraded_format,
                )
                n_align_failed += failed
                degraded_arm = {
                    "arm": "degradation",
                    "degradation": degradation_name,
                    "restorer": None,
                    "kernel_role": None,
                    "status": "RUN",
                    "reason": None,
                    "operator_params": degradation_metadata,
                    "image_path": degraded_relative.as_posix(),
                    "image_shape": list(degraded.shape),
                    "metrics": {
                        "psnr": _as_json_number(psnr(degraded, sharp)),
                        "ssim": _as_json_number(ssim(degraded, sharp)),
                    },
                    "gt_alignment": degraded_alignment,
                    "methods": degraded_methods,
                }
                sample_record["arms"].append(degraded_arm)
                _count_transitions(
                    transition_counts,
                    role,
                    degraded_arm,
                    sharp_methods,
                    degraded_methods,
                )

                for restorer_name in selected_restorers:
                    if restorer_name in KERNEL_RESTORERS:
                        if degradation_name not in BLUR_DEGRADATIONS:
                            sample_record["arms"].append(
                                {
                                    "arm": "restoration",
                                    "degradation": degradation_name,
                                    "restorer": restorer_name,
                                    "kernel_role": None,
                                    "status": "NOT_RUN",
                                    "reason": "no generating blur PSF",
                                    "operator_params": {},
                                    "image_path": None,
                                    "image_shape": None,
                                    "metrics": {
                                        "psnr": None,
                                        "ssim": None,
                                    },
                                    "gt_alignment": None,
                                    "methods": {},
                                }
                            )
                            continue
                        variants = (
                            ("oracle", "wrong")
                        )
                    else:
                        variants = (None,)

                    for kernel_role in variants:
                        if kernel_role is None:
                            restoration_params = deepcopy(
                                RESTORER_PARAMS[restorer_name]
                            )
                        else:
                            restoration_params = _restoration_params(
                                degradation_name,
                                degradation_metadata,
                                restorer_name,
                                kernel_role,
                            )
                        restored, restoration_metadata = restore(
                            degraded, restorer_name, restoration_params
                        )
                        role_label = kernel_role or "single"
                        restored_relative = (
                            Path(report_stem)
                            / "images"
                            / sample_id
                            / role
                            / f"{degradation_name}_{restorer_name}_{role_label}.png"
                        )
                        _save_image(output_root / restored_relative, restored)
                        restoration_methods, restoration_alignment, failed = (
                            _method_results(
                                restored,
                                sample_id,
                                f"{role}_{degradation_name}_{restorer_name}_{role_label}",
                                selected_methods,
                                cfg,
                                gt_mask,
                                degraded_format,
                            )
                        )
                        n_align_failed += failed
                        restoration_arm = {
                            "arm": "restoration",
                            "degradation": degradation_name,
                            "restorer": restorer_name,
                            "kernel_role": kernel_role,
                            "status": "RUN",
                            "reason": None,
                            "operator_params": restoration_metadata,
                            "image_path": restored_relative.as_posix(),
                            "image_shape": list(restored.shape),
                            "metrics": {
                                "psnr": _as_json_number(psnr(restored, sharp)),
                                "ssim": _as_json_number(ssim(restored, sharp)),
                            },
                            "gt_alignment": restoration_alignment,
                            "methods": restoration_methods,
                        }
                        sample_record["arms"].append(restoration_arm)
                        _count_transitions(
                            transition_counts,
                            role,
                            restoration_arm,
                            sharp_methods,
                            restoration_methods,
                        )

            samples.append(sample_record)

    provenance = {
        "n": len(entries),
        "seed": seed,
        "config_hash": cfg.snapshot_hash(),
        "git_sha": git_sha,
        "working_tree_clean": working_tree_clean,
        "dataset_version": adapter.version(),
    }
    payload = {
        "study": "enhance_study",
        "n": len(entries),
        "seed": seed,
        "provenance": provenance,
        "reading_rule": READING_RULE,
        "degradations": selected_degradations,
        "restorers": selected_restorers,
        "methods": selected_methods,
        "operator_params": {
            "degradations": {
                name: deepcopy(DEGRADATION_PARAMS[name])
                for name in selected_degradations
            },
            "restorers": {
                name: deepcopy(RESTORER_PARAMS[name])
                for name in selected_restorers
            },
        },
        "n_align_failed": n_align_failed,
        "transition_counts": transition_counts,
        "samples": samples,
    }
    _write_reports(payload, output_dir, report_stem)
    return payload


def _list_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="enhance_study")
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--degradations",
        type=_list_arg,
        default=list(DEFAULT_DEGRADATIONS),
    )
    parser.add_argument(
        "--restorers",
        type=_list_arg,
        default=list(DEFAULT_RESTORERS),
    )
    parser.add_argument("--methods", type=_list_arg, default=list(DEFAULT_METHODS))
    args = parser.parse_args(argv)

    raw_out = str(args.out)
    is_directory = raw_out.endswith(("/", "\\")) or Path(raw_out).is_dir()
    if is_directory:
        output_dir = Path(raw_out)
        report_stem = "enhance_study"
    else:
        output_path = Path(raw_out)
        output_dir = output_path.parent if str(output_path.parent) else Path(".")
        report_stem = output_path.name
    try:
        run_study(
            args.set,
            output_dir,
            seed=args.seed,
            degradations=args.degradations,
            restorers=args.restorers,
            methods=args.methods,
            report_stem=report_stem,
        )
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(output_dir / f"{report_stem}.json")
    print(output_dir / f"{report_stem}.md")


if __name__ == "__main__":
    main()
