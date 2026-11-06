"""Focused fixture for the enhancement study runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from imageguard.evaluation.enhance_study import run_study
from imageguard.evaluation.synthforge import generate_set

READING_RULE = "PSNR/SSIM gain never establishes a detection gain; negative results are successful"
BANNED_MARKDOWN = (
    "forged",
    "fake",
    "authentic",
    "restored the truth",
    "đã khôi phục",
    "ảnh rõ là ảnh thật",
    "enhancement xác nhận",
)


def _files_with_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_enhance_study_keeps_source_immutable_and_records_limits(tmp_path: Path) -> None:
    set_dir = tmp_path / "source"
    generate_set(
        2,
        seed=11,
        out_dir=set_dir,
        size=(128, 128),
        tier="easy",
    )
    before = _files_with_bytes(set_dir)
    out_dir = tmp_path / "out"

    run_study(
        set_dir,
        out_dir,
        seed=11,
        degradations=["gaussian", "jpeg"],
        restorers=["unsharp", "richardson_lucy"],
        methods=["noise"],
    )

    assert _files_with_bytes(set_dir) == before
    payload = json.loads((out_dir / "enhance_study.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "enhance_study.md").read_text(encoding="utf-8")
    assert payload["reading_rule"] == READING_RULE

    manipulated = next(
        sample for sample in payload["samples"]
        if sample["role"] == "manipulated_sample"
    )
    gaussian_degraded = next(
        arm for arm in manipulated["arms"]
        if arm["arm"] == "degradation" and arm["degradation"] == "gaussian"
    )
    assert "psnr" in gaussian_degraded["metrics"]
    assert "ssim" in gaussian_degraded["metrics"]

    jpeg_lucy = next(
        arm for arm in manipulated["arms"]
        if arm["arm"] == "restoration"
        and arm["degradation"] == "jpeg"
        and arm["restorer"] == "richardson_lucy"
    )
    assert jpeg_lucy["status"] == "NOT_RUN"
    assert jpeg_lucy["reason"] == "no generating blur PSF"

    gaussian_lucy_oracle = next(
        arm for arm in manipulated["arms"]
        if arm["arm"] == "restoration"
        and arm["degradation"] == "gaussian"
        and arm["restorer"] == "richardson_lucy"
        and arm["kernel_role"] == "oracle"
    )
    psnr_value = gaussian_lucy_oracle["metrics"]["psnr"]
    assert psnr_value is None or isinstance(psnr_value, (int, float))
    assert psnr_value != "0"

    matched_original = next(
        sample for sample in payload["samples"]
        if sample["role"] == "matched_original"
    )
    assert all(
        result["iou"] is None
        for arm in matched_original["arms"]
        for result in arm["methods"].values()
    )

    combined = json.dumps(payload, ensure_ascii=False).lower() + "\n" + markdown.lower()
    assert READING_RULE.lower() in combined
    assert all(phrase not in markdown.lower() for phrase in BANNED_MARKDOWN)
    assert all(phrase not in json.dumps(payload, ensure_ascii=False).lower()
               for phrase in BANNED_MARKDOWN)
    assert _files_with_bytes(set_dir) == before


def test_output_junction_into_source_is_refused(tmp_path: Path) -> None:
    set_dir = tmp_path / "source"
    generate_set(1, seed=3, out_dir=set_dir, size=(64, 64), tier="easy")
    before = _files_with_bytes(set_dir)
    out_dir = tmp_path / "out"
    link = out_dir / "enhance_study" / "images"
    link.parent.mkdir(parents=True)
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(set_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"junction was not created: {created.stderr.strip()}")

    with pytest.raises(ValueError, match="source set"):
        run_study(
            set_dir,
            out_dir,
            seed=3,
            degradations=["jpeg"],
            restorers=["unsharp"],
            methods=["noise"],
        )

    assert _files_with_bytes(set_dir) == before
