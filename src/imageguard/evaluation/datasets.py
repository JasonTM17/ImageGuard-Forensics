"""Dataset adapters — protocol + SynthAdapter.

Sample: {image_path, gt_mask_path(None=authentic), label, category, base_image_id}.
Split by base_image_id → chống leakage (biến thể cùng ảnh gốc cùng phía split).
Deterministic: split = hash(base_image_id) % 2 → dev|test, không random state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True, slots=True)
class Sample:
    image_path: Path
    gt_mask_path: Path | None  # None = authentic
    label: str  # "forged" | "authentic"
    category: str  # ví dụ "translate/none", "rotate/jpeg80" — per-condition report
    base_image_id: str


class DatasetAdapter(Protocol):
    name: str

    def samples(self, split: str = "dev") -> Iterable[Sample]:
        ...

    def version(self) -> str:
        ...


def split_of(base_id: str) -> str:
    """Deterministic split: hash ổn định của base_image_id → dev hoặc test.
    ~50/50 theo nghịch lý sinh nhật — đủ cho dev calibration vs locked test."""
    h = int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16)
    return "dev" if h % 2 == 0 else "test"


class SynthAdapter:
    """Đọc manifest synthforge (meta.json + *_O/_F/_M.png)."""

    name = "synthforge"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        mf = self.root / "meta.json"
        self.entries = json.loads(mf.read_text()) if mf.exists() else []

    def version(self) -> str:
        mf = self.root / "meta.json"
        if not mf.exists():
            return "absent"
        return hashlib.sha256(mf.read_bytes()).hexdigest()[:12]

    def samples(self, split: str = "dev") -> Iterable[Sample]:
        for e in self.entries:
            stem = e["id"]
            spec = e["spec"]
            cat = f"{spec['shape']}/{spec['postproc']}/rot{spec['rotate_deg']}"
            base_id = f"{stem}-seed{e['seed']}"
            if split_of(base_id) != split:
                continue
            f = self.root / f"{stem}_F.png"
            m = self.root / f"{stem}_M.png"
            o = self.root / f"{stem}_O.png"
            if f.exists() and m.exists():
                yield Sample(f, m, "forged", cat, base_id)
            if o.exists():
                # Ảnh gốc = authentic, gt mask rỗng — đo false-positive.
                yield Sample(o, None, "authentic", cat, base_id)
