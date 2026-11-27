"""Measure face-cue activation and coverage on a local labelled manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yaml import YAMLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imageguard.core.config import load_thresholds
from imageguard.evaluation.deepfake_study import run_deepfake_study
from imageguard.forensics.auxiliary.deepfake_model import config_digest, load_model


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="dataset root; images are read only")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--config", type=Path, help="fixed thresholds YAML; no automatic threshold tuning")
    parser.add_argument("--model", type=Path, help="fixed DIP JSON model; test only, audited against development data")
    parser.add_argument("--output", type=Path, help="new JSON report; omit to print to stdout")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cfg = load_thresholds(args.config)
        model = load_model(args.model, expected_config_sha256=config_digest(cfg.raw)) if args.model else None
        result = run_deepfake_study(manifest, args.root, split=args.split, cfg=cfg, model=model)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, YAMLError) as error:
        parser.exit(2, f"evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
