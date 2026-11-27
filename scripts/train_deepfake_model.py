"""Train a small DIP feature model on dev; report source-group-held-out results.

No weights or deepfake accuracy are supplied by this script. Training requires
independent labels and at least two usable source groups per class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yaml import YAMLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imageguard.core.config import load_thresholds
from imageguard.evaluation.deepfake_study import run_deepfake_study
from imageguard.evaluation.deepfake_training import fit_model


def _save_outputs(outputs: list[tuple[Path, dict]]) -> None:
    """Reserve new files exclusively; roll back only files this call created."""
    reserved: list[Path] = []
    try:
        for path, payload in outputs:
            text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                reserved.append(path)
                stream.write(text)
    except (OSError, ValueError):
        for path in reserved:
            path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="read-only dataset root")
    parser.add_argument("--config", type=Path, help="fixed extractor configuration")
    parser.add_argument("--folds", type=int, default=5, help="number of source-group folds within dev")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--l2", type=float, default=1.0, help="fixed regularization; never tuned on test")
    parser.add_argument("--output", type=Path, required=True, help="new model JSON file")
    parser.add_argument("--report", type=Path, required=True, help="new dev/OOF audit JSON file")
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.report.resolve():
            raise ValueError("model and report paths must differ")
        for path in (args.output, args.report):
            if path.exists():
                raise ValueError(f"output already exists: {path.name}")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        study = run_deepfake_study(manifest, args.root, split="dev", cfg=load_thresholds(args.config))
        included, excluded = [], []
        for row in study["samples"]:
            if row["exclusion_reason"] is None and row["features"] is not None:
                included.append(row)
            else:
                excluded.append({**row, "exclusion_reason": row["exclusion_reason"]
                                 or row["feature_exclusion_reason"] or "missing_features"})
        model, training = fit_model(
            included, config_sha256=study["config_sha256"], manifest_sha256=study["manifest_sha256"],
            folds=args.folds, seed=args.seed, l2=args.l2, excluded_dev=excluded,
        )
        report = {"study": study, "training": training,
                  "scope": "development only; grouped OOF is not an independent final test"}
        _save_outputs([(args.output, model), (args.report, report)])
        print(f"model: {args.output}\ndev report: {args.report}\n"
              f"usable dev images: {len(included)}; excluded: {len(excluded)}\n"
              "Evaluate the fixed model on source-disjoint test images before reporting detection quality.")
    except (OSError, ValueError, KeyError, YAMLError) as error:
        parser.exit(2, f"training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
