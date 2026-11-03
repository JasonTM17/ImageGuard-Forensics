"""CLI entry: `python -m imageguard [eval ...]`."""

from __future__ import annotations

import argparse
from pathlib import Path

from imageguard import __version__

_METHODS = {
    "keypoint": "imageguard.forensics.keypoint.pipeline:KeypointCopyMove",
    "block": "imageguard.forensics.block.pipeline:BlockCopyMove",
    "ela": "imageguard.forensics.auxiliary.ela:ELA",
}


def _load_method(spec: str):
    import importlib

    mod, cls = _METHODS[spec].split(":")
    return getattr(importlib.import_module(mod), cls)()


def main() -> None:
    ap = argparse.ArgumentParser(prog="imageguard")
    ap.add_argument("--version", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    ev = sub.add_parser("eval", help="chạy evaluation trên dataset")
    ev.add_argument("--dataset", choices=["synth", "comofod"], required=True)
    ev.add_argument("--root", type=Path, required=True)
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--methods", default="keypoint,block")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    if args.cmd == "eval":
        from imageguard.core.config import load_thresholds
        from imageguard.evaluation.comofod import CoMoFoDAdapter
        from imageguard.evaluation.datasets import SynthAdapter
        from imageguard.evaluation.runner import run_eval

        cfg = load_thresholds()
        adapter = (SynthAdapter(args.root) if args.dataset == "synth"
                   else CoMoFoDAdapter(args.root))
        methods = [_load_method(m.strip()) for m in args.methods.split(",")]
        report = run_eval(adapter, methods, cfg, args.split, args.seed)
        jp, cp = report.save(args.out)
        print(f"experiment_id: {report.experiment_id}")
        print(f"saved: {jp} | {cp}")
        for name, d in report.per_method.items():
            print(f"  {name}: f1={d['f1']} iou_macro={d['pixel_macro']['iou']} "
                  f"n={d['n']} excl={d['n_excluded']} inconc={d['n_inconclusive']}")
        return
    print(f"ImageGuard Forensics v{__version__} — python -m imageguard eval --help")


if __name__ == "__main__":
    main()
