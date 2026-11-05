"""CoMoFoD adapter — discovery-based, không redistribute dataset.

Quy ước file CoMoFoD (Tralic et al., ELMAR-2013):
  `NNN_F.png` forged · `NNN_O.png` original · `NNN_B.png` mask binary ·
  `NNN_C.png` mask color (một số bản `_M`). Discovery = scan dir, parse
  suffix, build manifest từ file thật — không hardcode số lượng.

GATE verify thủ công (bắt buộc trước batch eval đầu):
  - assert dims mask == dims ảnh; lệch → reject+log, KHÔNG silent resize.
  - semantic mask (vùng nào là forged: trắng hay đen) phải đọc từ readme/
    verify bằng mắt — adapter ghi `mask_convention` vào manifest, không đoán.
Thiếu dataset → samples() rỗng + version="absent" — KHÔNG fail suite.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from imageguard.evaluation.datasets import Sample, split_of

log = logging.getLogger(__name__)

_PATTERN = re.compile(r"^(?P<id>.+?)_(?P<kind>F|O|B|C|M)\.(?P<ext>png|jpg|jpeg|tif|bmp)$", re.IGNORECASE)


class CoMoFoDAdapter:
    name = "comofod"

    def __init__(self, root: Path, mask_convention: str = "binary_white_is_forged") -> None:
        self.root = Path(root)
        self.mask_convention = mask_convention
        self._manifest: list[Sample] | None = None

    def version(self) -> str:
        return "absent" if not self.root.exists() else "present-unverified"

    def _discover(self) -> list[Sample]:
        forged: dict[str, Path] = {}
        originals: dict[str, Path] = {}
        masks: dict[str, Path] = {}
        for f in sorted(self.root.glob("*")):
            m = _PATTERN.match(f.name)
            if not m:
                continue
            kind = m["kind"].upper()
            if kind == "F":
                forged[m["id"]] = f
            elif kind == "O":
                originals[m["id"]] = f
            elif kind == "B":
                masks[m["id"]] = f  # binary mask — ưu tiên tuyệt đối
            elif kind in ("C", "M") and m["id"] not in masks:
                # Fallback khi không có _B: giữ C/M nhưng KHÔNG ghi đè _B.
                # docs/dataset-strategy: _B là GT binary chính thức.
                masks[m["id"]] = f
                log.info("comofod %s: thiếu _B — dùng _%s làm GT fallback",
                         m["id"], kind)
        out: list[Sample] = []
        for sid, fpath in forged.items():
            mp = masks.get(sid)
            if mp is None:
                log.warning("comofod %s: forged thiếu mask — skip", sid)
                continue
            cat = "comofod"
            out.append(Sample(fpath, mp, "forged", cat, sid))
            if sid in originals:
                out.append(Sample(originals[sid], None, "authentic", cat, sid))
        return out

    def samples(self, split: str = "dev") -> Iterable[Sample]:
        if not self.root.exists():
            log.info("CoMoFoD absent tại %s — skip (không fail)", self.root)
            return
        if self._manifest is None:
            self._manifest = self._discover()
            log.info("CoMoFoD manifest: %d samples", len(self._manifest))
        for s in self._manifest:
            if split_of(s.base_image_id) == split:
                yield s

    def verify_masks(self) -> dict:
        """Kiểm dims mask==image trên mọi forged pair. Trả report dict —
        caller (script) in ra cho user verify semantic bằng mắt."""
        report = {"checked": 0, "dim_mismatch": [], "mask_values": {}}
        for s in self.samples("dev"):
            if s.gt_mask_path is None:
                continue
            with Image.open(s.image_path) as im, Image.open(s.gt_mask_path) as mk:
                report["checked"] += 1
                if im.size != mk.size:
                    report["dim_mismatch"].append(s.image_path.name)
                    continue
                uniq = np.unique(np.asarray(mk.convert("L")))
                report["mask_values"][s.image_path.name] = uniq[:6].tolist()
        return report
