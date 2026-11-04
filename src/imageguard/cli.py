"""CLI entry: `python -m imageguard [eval ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imageguard import __version__

from imageguard.forensics.registry import load_method as _load_method


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    _ver = argparse.ArgumentParser(add_help=False)
    _ver.add_argument("--version", action="version",
                      version=f"ImageGuard Forensics v{__version__}")
    ap = argparse.ArgumentParser(prog="imageguard", parents=[_ver])
    sub = ap.add_subparsers(dest="cmd")
    ev = sub.add_parser("eval", help="chạy evaluation trên dataset",
                        parents=[_ver])
    ev.add_argument("--dataset", choices=["synth", "comofod"], required=True)
    ev.add_argument("--root", type=Path, required=True)
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--methods", default="keypoint,block")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out", type=Path, default=Path("reports"))
    sub.add_parser("gui", help="mở desktop app (PySide6)", parents=[_ver])
    args = ap.parse_args()

    if args.cmd in (None, "gui"):
        try:
            from imageguard.ui.app import main as gui_main
        except ImportError as e:
            print(f"GUI khong kha dung ({e}); dung: python -m imageguard eval --help")
            return
        sys.exit(gui_main())

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
                  f"n={d['n']} excl={d['n_excluded']} inconc={d['n_inconclusive']} "
                  f"na={d['n_not_applicable']} err={d['n_error']}")
        return
    print(f"ImageGuard Forensics v{__version__} — python -m imageguard eval --help")


if __name__ == "__main__":
    main()
