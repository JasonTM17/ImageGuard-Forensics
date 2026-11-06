"""Run the NAFNet viewing arm using the established synthetic study protocol."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from imageguard.enhancement.derive import derive
from imageguard.enhancement.nafnet import MAX_LONG_SIDE, MODEL_BYTES
from imageguard.evaluation import enhance_study
from imageguard.evaluation.synthforge import TIERS, generate_set

DEFAULT_SEED = 20261106
DEFAULT_SAMPLE_COUNT = 24
DEFAULT_TIER = "easy"
DEFAULT_SIZE = 256
MAX_PREFLIGHT_SECONDS = 30.0
MAX_ESTIMATED_INFERENCE_SECONDS = 20 * 60
RESTORER = "nafnet"
METHODS = ("keypoint", "block", "ela", "noise")


def _preflight() -> tuple[float, dict[str, Any] | None, Exception | None]:
    sample = np.zeros((256, 256, 3), dtype=np.uint8)
    started = time.perf_counter()
    try:
        _result, metadata = derive(sample, RESTORER, {})
    except Exception as exc:  # noqa: BLE001 - report the measured failure safely
        return time.perf_counter() - started, None, exc
    return time.perf_counter() - started, metadata, None


def _write_payload(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nafnet_arm.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if payload["status"] == "NOT_RUN":
        markdown = "\n".join(
            (
                "# NAFNet E3 Viewing Arm",
                "",
                f"- Status: `{payload['status']}`",
                f"- Reason: {payload['reason']}",
                f"- Seed: `{payload['seed']}`",
                "- No study measurements were produced.",
                "",
                (
                    "NAFNet is a regression network and can synthesize texture. "
                    "This synthetic study is not an IoT camera result. "
                    "PSNR is an image-similarity measure, not a detection gain."
                ),
                "",
            )
        )
    else:
        markdown = enhance_study._render_markdown(payload).replace(
            "# Classical Enhancement Study",
            "# NAFNet E3 Viewing Arm",
            1,
        )
        markdown += "\n## E3 status\n\n"
        markdown += f"- Status: `{payload['status']}`\n"
        markdown += (
            f"- One 256×256 CPU inference: "
            f"{payload['runtime']['preflight_seconds']:.3f} seconds.\n"
        )
        markdown += (
            f"- Estimated NAFNet inference time for the requested run: "
            f"{payload['runtime']['estimated_inference_seconds']:.1f} seconds.\n"
        )
        if payload.get("runtime_note"):
            markdown += f"- Runtime note: {payload['runtime_note']}\n"
        markdown += (
            "\n## Viewing model caveat\n\n"
            "NAFNet is a regression network and can synthesize texture. "
            "This synthetic study is not an IoT camera result. "
            "PSNR is an image-similarity measure and does not establish a "
            "detection gain.\n"
        )
    (out_dir / "nafnet_arm.md").write_text(markdown, encoding="utf-8")


def run_nafnet_arm(
    out_dir: str | Path,
    *,
    n: int = DEFAULT_SAMPLE_COUNT,
    tier: str = DEFAULT_TIER,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the paired NAFNet arm; generated inputs and scored images stay temporary."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be a positive integer")
    if tier not in TIERS:
        raise ValueError("tier is not supported")
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 1:
        raise ValueError("size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    output_dir = Path(out_dir)
    preflight_seconds, model_metadata, preflight_error = _preflight()
    if preflight_error is not None:
        detail = ""
        if isinstance(preflight_error, FileNotFoundError):
            detail = ": NAFNet weight is missing"
        elif "Cannot use 'edge' mode to pad dimension with a value of 0" in str(
            preflight_error
        ):
            detail = ": the model graph produced zero-sized spatial features"
        payload: dict[str, Any] = {
            "study": "nafnet_arm",
            "status": "NOT_RUN",
            "reason": (
                f"256×256 NAFNet preflight failed "
                f"({type(preflight_error).__name__}){detail}"
            ),
            "n": 0,
            "n_requested": n,
            "seed": seed,
            "degradations": list(enhance_study.DEFAULT_DEGRADATIONS),
            "restorers": [RESTORER],
            "methods": list(METHODS),
            "model": {
                "name": "NAFNet",
                "model_bytes": MODEL_BYTES,
                "max_long_side": MAX_LONG_SIDE,
            },
            "runtime": {"preflight_seconds": preflight_seconds},
        }
        _write_payload(payload, output_dir)
        return payload

    if preflight_seconds > MAX_PREFLIGHT_SECONDS:
        payload = {
            "study": "nafnet_arm",
            "status": "NOT_RUN",
            "reason": (
                "256×256 NAFNet preflight exceeded the 30-second runtime limit"
            ),
            "n": 0,
            "n_requested": n,
            "seed": seed,
            "degradations": list(enhance_study.DEFAULT_DEGRADATIONS),
            "restorers": [RESTORER],
            "methods": list(METHODS),
            "model": model_metadata,
            "runtime": {"preflight_seconds": preflight_seconds},
        }
        _write_payload(payload, output_dir)
        return payload

    inference_count_per_sample = 2 * len(enhance_study.DEFAULT_DEGRADATIONS)
    estimated_seconds = preflight_seconds * n * inference_count_per_sample
    n_used = n
    runtime_note = None
    if estimated_seconds > MAX_ESTIMATED_INFERENCE_SECONDS:
        n_used = max(
            1,
            min(
                n,
                math.floor(
                    MAX_ESTIMATED_INFERENCE_SECONDS
                    / (preflight_seconds * inference_count_per_sample)
                ),
            ),
        )
        runtime_note = (
            f"Sample count reduced from {n} to {n_used} using the timed "
            "single-call estimate to leave time for detector scoring."
        )
        estimated_seconds = preflight_seconds * n_used * inference_count_per_sample

    try:
        with tempfile.TemporaryDirectory(prefix="imageguard-nafnet-") as temp_root:
            temp_path = Path(temp_root)
            set_dir = temp_path / "set"
            eval_dir = temp_path / "evaluation"
            generate_set(
                n_used,
                seed=seed,
                out_dir=set_dir,
                size=(size, size),
                tier=tier,
            )

            restorer_params = {
                **enhance_study.RESTORER_PARAMS,
                RESTORER: {},
            }

            def restore_nafnet(
                image: np.ndarray, kind: str, params: dict
            ) -> tuple[np.ndarray, dict]:
                if kind != RESTORER:
                    raise ValueError("unexpected restorer in NAFNet study arm")
                return derive(image, RESTORER, params)

            with (
                patch.object(enhance_study, "RESTORER_PARAMS", restorer_params),
                patch.object(enhance_study, "restore", restore_nafnet),
            ):
                payload = enhance_study.run_study(
                    set_dir,
                    eval_dir,
                    seed=seed,
                    restorers=[RESTORER],
                    methods=METHODS,
                    report_stem="nafnet_arm",
                )
    except Exception as exc:  # noqa: BLE001 - retain a truthful study-arm status
        payload = {
            "study": "nafnet_arm",
            "status": "NOT_RUN",
            "reason": f"synthetic study arm failed ({type(exc).__name__})",
            "n": 0,
            "n_requested": n,
            "seed": seed,
            "degradations": list(enhance_study.DEFAULT_DEGRADATIONS),
            "restorers": [RESTORER],
            "methods": list(METHODS),
            "model": model_metadata,
        }
        _write_payload(payload, output_dir)
        return payload

    payload["study"] = "nafnet_arm"
    payload["status"] = "DONE_WITH_CONCERNS" if runtime_note else "DONE"
    payload["n_requested"] = n
    payload["model"] = model_metadata
    payload["runtime"] = {
        "preflight_seconds": preflight_seconds,
        "estimated_inference_seconds": estimated_seconds,
        "inference_count_per_sample": inference_count_per_sample,
    }
    if runtime_note:
        payload["runtime_note"] = runtime_note
    _write_payload(payload, output_dir)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nafnet_arm")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", default=DEFAULT_SAMPLE_COUNT, type=int)
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=TIERS)
    parser.add_argument("--size", default=DEFAULT_SIZE, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args(argv)
    payload = run_nafnet_arm(
        args.out,
        n=args.n,
        tier=args.tier,
        size=args.size,
        seed=args.seed,
    )
    print(f"NAFNet arm: {payload['status']}")
    print("Study report: nafnet_arm.md")


if __name__ == "__main__":
    main()
