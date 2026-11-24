"""Ranh giới import ipkit: module giảng dạy không đi theo detector."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

_IPKIT = Path(__file__).resolve().parents[1] / "src" / "imageguard" / "ipkit"
_BANNED_ROOTS = {"cv2", "scipy", "skimage", "PIL", "torch", "sklearn"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_ipkit_sources_do_not_import_banned_libraries() -> None:
    offenders: list[str] = []
    for path in sorted(_IPKIT.rglob("*.py")):
        hit = _imported_roots(path) & _BANNED_ROOTS
        if hit:
            offenders.append(f"{path.name}: {sorted(hit)}")
    assert offenders == []


def test_forensic_named_import_skips_new_teaching_modules() -> None:
    script = """
import sys
from imageguard.ipkit import features_basic
missing = [
    name
    for name in (
        "imageguard.ipkit.geometry",
        "imageguard.ipkit.color",
        "imageguard.ipkit.features_harris",
    )
    if name in sys.modules
]
raise SystemExit("loaded " + ",".join(missing) if missing else 0)
"""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
