"""CLI entry: `python -m imageguard [eval | gui | analyze | ingest ...]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imageguard import __version__
from imageguard.forensics.registry import METHOD_SPECS
from imageguard.forensics.registry import load_method as _load_method
from imageguard.services.analysis_service import DEFAULT_METHODS

_DIRECT_CMDS = {"eval", "gui", "analyze", "ingest"}
_METHODS_DEFAULT_STR = ",".join(DEFAULT_METHODS)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in _DIRECT_CMDS:
        sys.argv.insert(1, "gui")

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
    gui_p = sub.add_parser("gui", help="mở desktop app (PySide6)", parents=[_ver])
    gui_p.add_argument("image", nargs="?", type=Path, help="ảnh mở sẵn")
    analyze_p = sub.add_parser(
        "analyze",
        help="phân tích một ảnh, in báo cáo JSON (dấu hiệu) ra stdout",
        parents=[_ver],
    )
    analyze_p.add_argument("file", type=Path, help="file ảnh cần phân tích")
    analyze_p.add_argument(
        "--methods",
        default=_METHODS_DEFAULT_STR,
        metavar="LIST",
        help=(
            "danh sách method, phân cách bằng dấu phẩy "
            f"(mặc định: {_METHODS_DEFAULT_STR})"
        ),
    )
    analyze_p.add_argument(
        "--no-persist",
        action="store_true",
        help="không ghi cơ sở dữ liệu hay artifacts",
    )
    ingest_p = sub.add_parser("ingest", help="theo dõi thư mục ảnh tĩnh", parents=[_ver])
    ingest_sub = ingest_p.add_subparsers(dest="ingest_mode", required=True)
    for mode in ("watch", "once"):
        mode_p = ingest_sub.add_parser(mode, parents=[_ver])
        mode_p.add_argument("directory", type=Path)
        mode_p.add_argument("--config", type=Path, help="tệp cấu hình ingest")
    args = ap.parse_args()

    if args.cmd in (None, "gui"):
        try:
            from imageguard.ui.app import main as gui_main
        except ImportError as e:
            print(f"GUI khong kha dung ({e}); dung: python -m imageguard eval --help")
            return
        sys.exit(gui_main(getattr(args, "image", None)))

    if args.cmd == "eval":
        from imageguard.core.config import load_thresholds
        from imageguard.evaluation.comofod import CoMoFoDAdapter
        from imageguard.evaluation.datasets import SynthAdapter
        from imageguard.evaluation.runner import run_eval

        specs = [m.strip() for m in args.methods.split(",") if m.strip()]
        methods = []
        for spec in specs:
            try:
                methods.append(_load_method(spec))
            except KeyError:
                valid = ", ".join(sorted(METHOD_SPECS))
                print(
                    f"error: unknown eval method {spec!r}; valid: {valid}",
                    file=sys.stderr,
                )
                raise SystemExit(2) from None

        cfg = load_thresholds()
        adapter = (SynthAdapter(args.root) if args.dataset == "synth"
                   else CoMoFoDAdapter(args.root))
        report = run_eval(adapter, methods, cfg, args.split, args.seed)
        jp, cp = report.save(args.out)
        print(f"experiment_id: {report.experiment_id}")
        print(f"saved: {jp} | {cp}")
        for name, d in report.per_method.items():
            print(f"  {name}: f1={d['f1']} iou_macro={d['pixel_macro']['iou']} "
                  f"n={d['n']} excl={d['n_excluded']} inconc={d['n_inconclusive']} "
                  f"na={d['n_not_applicable']} err={d['n_error']}")
        return

    if args.cmd == "analyze":
        _run_analyze_command(args)
        return

    if args.cmd == "ingest":
        from imageguard.ingest.commands import run_ingest

        raise SystemExit(
            run_ingest(args.ingest_mode, args.directory, args.config)
        )
    print(f"ImageGuard Forensics v{__version__} — python -m imageguard eval --help")


def _run_analyze_command(args: argparse.Namespace) -> None:
    from imageguard.core.config import load_thresholds
    from imageguard.core.errors import ErrorResult
    from imageguard.core.models import ForensicReport
    from imageguard.services.analysis_service import (
        AnalysisRequest,
        AnalysisService,
        report_to_dict,
    )
    from imageguard.storage.artifacts import ArtifactStore
    from imageguard.storage.db import make_session_factory
    from imageguard.storage.repository import AnalysisRepository, ImageRepository

    cfg = load_thresholds()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        methods = list(DEFAULT_METHODS)

    if args.no_persist:
        svc = AnalysisService(cfg)
    else:
        db_path = Path("data") / "imageguard.db"
        sf = make_session_factory(db_path)
        store = ArtifactStore()
        svc = AnalysisService(cfg, ImageRepository(sf), AnalysisRepository(sf), store)

    req = AnalysisRequest(path=args.file, methods=methods)
    result = svc.analyze(req)

    if isinstance(result, ForensicReport):
        payload = report_to_dict(result)
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=1))
        sys.exit(0)
    if isinstance(result, ErrorResult):
        err_payload = {
            "code": result.code.value,
            "message": result.message,
            "detail": result.detail,
        }
        print(json.dumps(err_payload, ensure_ascii=False, default=str, indent=1))
        sys.exit(1)
    print(json.dumps(
        {"code": "INTERNAL", "message": "Unexpected result type",
         "detail": {"type": type(result).__name__}},
        ensure_ascii=False, default=str, indent=1,
    ))
    sys.exit(1)


if __name__ == "__main__":
    main()
