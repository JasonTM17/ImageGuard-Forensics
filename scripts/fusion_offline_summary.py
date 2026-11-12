"""Offline summary: map saved detector rows to fusion verdicts (no re-run).

Reads a CASIA formal ``report.json`` (per-method rows only) and reports counts
plus Wilson 95% intervals for originals vs edits, per method and fused.
Exploratory / same scored event — not a held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult
from imageguard.forensics.fusion import combine

METHOD_KEYPOINT = MethodName.COPY_MOVE_KEYPOINT.value
METHOD_BLOCK = MethodName.COPY_MOVE_BLOCK.value
LABEL_AUTHENTIC = "authentic"
LABEL_FORGED = "forged"
VERDICT_KEYS = (
    Verdict.SUSPICIOUS_INDICATION.value,
    Verdict.INCONCLUSIVE.value,
    Verdict.NO_INDICATION.value,
)
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (0..1 each endpoint)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def status_to_verdict(status: str) -> Verdict:
    if not status or not isinstance(status, str):
        raise ValueError("row status must be a non-empty string")
    try:
        return Verdict(status)
    except ValueError as exc:
        raise ValueError(f"unknown verdict status: {status!r}") from exc


def row_to_result(row: dict[str, Any]) -> AlgorithmResult:
    method_raw = row.get("method")
    if method_raw not in (METHOD_KEYPOINT, METHOD_BLOCK):
        raise ValueError(f"unsupported method in row: {method_raw!r}")
    method = MethodName(method_raw)
    verdict = status_to_verdict(str(row["status"]))
    score = row.get("score")
    if score is not None and not isinstance(score, (int, float)):
        raise ValueError("score must be numeric or null")
    return AlgorithmResult(
        method=method,
        verdict=verdict,
        score=float(score) if score is not None else None,
        score_definition=str(row.get("score_definition") or ""),
        evidence=dict(row.get("evidence") or {}),
        limitations=list(row.get("limitations") or []),
        params_hash=str(row.get("params_hash") or ""),
        duration_ms=int(row.get("duration_ms") or 0),
    )


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"report file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in report: {path}") from exc
    if not isinstance(data, dict) or "rows" not in data:
        raise ValueError("report must be an object with a 'rows' array")
    rows = data["rows"]
    if not isinstance(rows, list):
        raise TypeError("'rows' must be a list")
    return data


@dataclass(frozen=True)
class SampleVerdicts:
    sample_id: str
    label: str
    keypoint: Verdict
    block: Verdict
    fused: Verdict


def aggregate_samples(rows: list[dict[str, Any]]) -> list[SampleVerdicts]:
    by_sample: dict[str, dict[str, Any]] = defaultdict(dict)
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each row must be an object")
        sid = row.get("sample_id")
        if not sid or not isinstance(sid, str):
            raise ValueError("row missing sample_id")
        label = row.get("label")
        if label not in (LABEL_AUTHENTIC, LABEL_FORGED):
            raise ValueError(f"unexpected label for {sid}: {label!r}")
        if sid in labels and labels[sid] != label:
            raise ValueError(f"conflicting labels for sample_id {sid}")
        labels[sid] = label
        method = row.get("method")
        if method in (METHOD_KEYPOINT, METHOD_BLOCK):
            if method in by_sample[sid]:
                raise ValueError(f"duplicate row for {sid} method {method}")
            by_sample[sid][method] = row

    out: list[SampleVerdicts] = []
    for sid in sorted(by_sample):
        methods = by_sample[sid]
        if METHOD_KEYPOINT not in methods or METHOD_BLOCK not in methods:
            raise ValueError(f"sample {sid} missing keypoint or block row")
        kp = row_to_result(methods[METHOD_KEYPOINT])
        bl = row_to_result(methods[METHOD_BLOCK])
        fused, _, _ = combine([kp, bl])
        out.append(
            SampleVerdicts(
                sample_id=sid,
                label=labels[sid],
                keypoint=kp.verdict,
                block=bl.verdict,
                fused=fused,
            )
        )
    return out


def _verdict_for_view(sample: SampleVerdicts, view: str) -> Verdict:
    if view == "keypoint":
        return sample.keypoint
    if view == "block":
        return sample.block
    if view == "fused_keypoint_block":
        return sample.fused
    raise ValueError(f"unknown view: {view!r}")


def summarize(samples: list[SampleVerdicts]) -> dict[str, Any]:
    views = ("keypoint", "block", "fused_keypoint_block")
    result: dict[str, Any] = {
        "exploratory_note": (
            "Offline replay of fusion.combine on saved keypoint+block rows only; "
            "aux methods (ELA/noise/frequency/splicing) absent — GUI default list "
            "may yield more INCONCLUSIVE. Same 103-pair CASIA formal event; "
            "not held-out; not the 34 held-aside pairs."
        ),
        "n_samples": len(samples),
        "by_view": {},
    }
    for view in views:
        block: dict[str, Any] = {}
        for cohort, label in (("originals", LABEL_AUTHENTIC), ("edits", LABEL_FORGED)):
            cohort_samples = [s for s in samples if s.label == label]
            n = len(cohort_samples)
            counts = {k: 0 for k in VERDICT_KEYS}
            for s in cohort_samples:
                v = _verdict_for_view(s, view)
                key = v.value
                if key not in counts:
                    raise ValueError(f"unexpected verdict {key} in view {view}")
                counts[key] += 1
            rates: dict[str, Any] = {}
            for vk, c in counts.items():
                low, high = wilson_interval(c, n)
                rates[vk] = {
                    "count": c,
                    "proportion": (c / n) if n else None,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                }
            block[cohort] = {"n": n, "verdicts": rates}
        result["by_view"][view] = block
    return result


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fusion offline summary (exploratory)",
        "",
        summary.get("exploratory_note", ""),
        "",
        (
            f"Total paired samples: **{summary.get('n_samples', 0)}** "
            "(each original/edit image appears once per view)."
        ),
        "",
    ]
    view_titles = {
        "keypoint": "Keypoint alone",
        "block": "Block alone",
        "fused_keypoint_block": "Fused (keypoint + block)",
    }
    for view, title in view_titles.items():
        block = summary["by_view"][view]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Cohort | n | suspicious | inconclusive | no_indication |"
        )
        lines.append("| --- | ---: | --- | --- | --- |")
        for cohort in ("originals", "edits"):
            part = block[cohort]
            n_cohort = part["n"]
            verdicts = part["verdicts"]

            def _cell(key: str, *, _n: int = n_cohort, _v: dict = verdicts) -> str:
                r = _v[key]
                c = r["count"]
                if _n == 0:
                    return f"{c}"
                lo = r["wilson_95_low"]
                hi = r["wilson_95_high"]
                if math.isnan(lo):
                    return f"{c}"
                return f"{c} ({lo:.3f}–{hi:.3f})"

            lines.append(
                f"| {cohort} | {n_cohort} | "
                f"{_cell(Verdict.SUSPICIOUS_INDICATION.value)} | "
                f"{_cell(Verdict.INCONCLUSIVE.value)} | "
                f"{_cell(Verdict.NO_INDICATION.value)} |"
            )
        lines.append("")
        lines.append(
            "_Counts shown; parentheses = Wilson 95% interval for that verdict’s "
            "proportion within the cohort._"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline fused-verdict summary from a saved detector report.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to report.json with per-method rows",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "both"),
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args(argv)
    data = load_report(args.report)
    samples = aggregate_samples(data["rows"])
    summary = summarize(samples)
    if args.format in ("json", "both"):
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.format in ("markdown", "both"):
        if args.format == "both":
            print()
        print(format_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
