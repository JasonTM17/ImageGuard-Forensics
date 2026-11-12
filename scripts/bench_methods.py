"""Benchmark wall-clock runtime của các forensic method — tái lập được (seed cố định).

Đo từng method qua registry (cùng entry `analyze` như AnalysisService), và một
lần chạy end-to-end AnalysisService (không persist) với DEFAULT_METHODS.
Bộ nhớ đỉnh không báo cáo — tracemalloc không phản ánh OpenCV/native alloc.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.evaluation.synthforge import _base_image
from imageguard.forensics.registry import METHOD_SPECS, load_method
from imageguard.services.analysis_service import (
    DEFAULT_METHODS,
    AnalysisRequest,
    AnalysisService,
)

BENCH_SEED = 424242
FULL_SIZES = (256, 512, 1024, 2048)
QUICK_SIZES = (256, 512)
JPEG_QUALITY = 85


def _validate_sizes(sizes: tuple[int, ...]) -> tuple[int, ...]:
    if not sizes:
        raise ValueError("sizes không được rỗng")
    for s in sizes:
        if s < 32 or s > 8192:
            raise ValueError(f"kích thước không hợp lệ: {s} (cần 32–8192)")
    return sizes


def _validate_repetitions(n: int) -> int:
    if n < 1:
        raise ValueError(f"repetitions phải >= 1, nhận {n}")
    return n


def _synth_square(seed: int, side: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed + side * 1_000_003))
    return _base_image(rng, (side, side))


def _write_assets(
    work_dir: Path, seed: int, sizes: tuple[int, ...]
) -> dict[int, dict[str, Path]]:
    """PNG cho hầu hết method; JPEG riêng cho ELA."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, dict[str, Path]] = {}
    for side in sizes:
        bgr = _synth_square(seed, side)
        png = work_dir / f"synth_{side}.png"
        jpg = work_dir / f"synth_{side}.jpg"
        ok_png, enc_png = cv2.imencode(".png", bgr)
        if not ok_png:
            raise ValueError(f"imencode PNG thất bại tại size {side}")
        png.write_bytes(enc_png.tobytes())
        ok_jpg, enc_jpg = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok_jpg:
            raise ValueError(f"imencode JPEG thất bại tại size {side}")
        jpg.write_bytes(enc_jpg.tobytes())
        out[side] = {"png": png, "jpeg": jpg}
    return out


def _open_doc(path: Path, cfg) -> ImageDocument:
    doc = safe_open(path, cfg)
    if isinstance(doc, ErrorResult):
        raise ValueError(f"safe_open thất bại cho {path.name}: {doc.message}")  # noqa: TRY004
    return doc


def _time_method(
    spec: str, doc: ImageDocument, cfg, repetitions: int
) -> dict[str, Any]:
    method = load_method(spec)
    samples: list[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        method.analyze(doc, cfg, None)
        samples.append(time.perf_counter() - t0)
    return {
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
    }


def _time_service(
    path: Path, cfg, repetitions: int, methods: list[str]
) -> dict[str, Any]:
    svc = AnalysisService(cfg)
    samples: list[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        out = svc.analyze(
            AnalysisRequest(path=path, methods=list(methods), include_exif=True)
        )
        if not hasattr(out, "report_id"):
            raise ValueError(f"AnalysisService lỗi: {getattr(out, 'message', out)}")
        samples.append(time.perf_counter() - t0)
    return {
        "methods": list(methods),
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
    }


def _platform_info() -> dict[str, str]:
    proc = platform.processor() or "unknown"
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": proc,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "opencv_threads": str(cv2.getNumThreads()),
    }


def run_benchmark(
    *,
    seed: int = BENCH_SEED,
    sizes: tuple[int, ...] = FULL_SIZES,
    repetitions: int = 3,
    work_dir: Path | None = None,
    method_specs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Chạy benchmark trong process; trả dict JSON-serializable."""
    sizes = _validate_sizes(sizes)
    repetitions = _validate_repetitions(repetitions)
    specs = method_specs or tuple(METHOD_SPECS.keys())

    cfg = load_thresholds()
    owned = work_dir is not None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="ig_bench_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)

    assets = _write_assets(work_dir, seed, sizes)
    method_results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec not in METHOD_SPECS:
            raise ValueError(f"method không có trong registry: {spec}")
        by_size: dict[str, Any] = {}
        for side in sizes:
            fmt = "jpeg" if spec == "ela" else "png"
            path = assets[side][fmt]
            doc = _open_doc(path, cfg)
            by_size[str(side)] = _time_method(spec, doc, cfg, repetitions)
        method_results[spec] = by_size

    service_results: dict[str, Any] = {}
    for side in sizes:
        path = assets[side]["png"]
        service_results[str(side)] = _time_service(
            path, cfg, repetitions, list(DEFAULT_METHODS)
        )

    payload: dict[str, Any] = {
        "meta": {
            "seed": seed,
            "repetitions": repetitions,
            "sizes": list(sizes),
            "jpeg_quality": JPEG_QUALITY,
            "method_specs": list(specs),
            "default_service_methods": list(DEFAULT_METHODS),
            **_platform_info(),
        },
        "per_method": method_results,
        "analysis_service_no_persist": service_results,
        "assets_dir": str(work_dir) if owned else None,
    }
    if tmp is not None:
        tmp.cleanup()
    return payload


def _print_table(data: dict[str, Any]) -> None:
    sizes = [str(s) for s in data["meta"]["sizes"]]
    specs = data["meta"]["method_specs"]
    header = ["method", *sizes, "unit"]
    rows = [header]
    for spec in specs:
        row = [spec]
        for side in sizes:
            med = data["per_method"][spec][side]["median_s"]
            row.append(f"{med:.6f}")
        row.append("s (median)")
        rows.append(row)
    rows.append([])
    rows.append(["analysis_service (no persist)", *[
        f"{data['analysis_service_no_persist'][s]['median_s']:.6f}" for s in sizes
    ], "s (median)"])
    col_w = [max(len(r[i]) for r in rows if i < len(r)) for i in range(len(header))]
    for row in rows:
        if not row:
            print()
            continue
        print("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Benchmark forensic method runtime")
    ap.add_argument("--seed", type=int, default=BENCH_SEED)
    ap.add_argument(
        "-n",
        "--repetitions",
        type=int,
        default=3,
        help="số lần lặp mỗi method/size (mặc định 3)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="file JSON đầu ra (mặc định: thư mục tạm, không ghi vào repo data/)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke: sizes 256,512 và 1 repetition",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="giữ file ảnh synth (debug); mặc định xóa sau khi đo",
    )
    args = ap.parse_args(argv)

    if args.quick:
        sizes: tuple[int, ...] = QUICK_SIZES
        repetitions = 1
    else:
        sizes = FULL_SIZES
        repetitions = args.repetitions

    _validate_repetitions(repetitions)

    data = run_benchmark(
        seed=args.seed,
        sizes=sizes,
        repetitions=repetitions,
        work_dir=args.work_dir,
    )

    out_path = args.out
    if out_path is None:
        out_path = Path(tempfile.gettempdir()) / "imageguard_bench_methods.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    _print_table(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
