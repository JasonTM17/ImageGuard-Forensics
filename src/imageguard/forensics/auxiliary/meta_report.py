"""EXIF/metadata consistency audit — findings[], KHÔNG verdict.

Metadata là dữ kiện KHAI BÁO do camera/phần mềm ghi — dễ chỉnh sửa hoặc
bị xoá. Một finding (vd: Software=Photoshop) nghĩa là "ảnh đi qua công cụ
chỉnh sửa", KHÔNG nghĩa là nội dung bị giả mạo — crop/resize hợp pháp cũng
để lại vệt tương tự.
"""

from __future__ import annotations

import threading
import time

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument

EDITOR_HINTS = (
    "photoshop", "gimp", "lightroom", "snapseed", "paint", "pixlr",
    "affinity", "corel", "picsart", "canva", "photoscape",
)


def audit_exif(doc: ImageDocument, cfg: ThresholdSet | None = None,
               cancel: threading.Event | None = None) -> AlgorithmResult:
    """Đọc EXIF đã parse: Software, Make/Model, DateTime vs Original.

    Chủ đề: metadata forensics (không xử lý pixel). **Aux**: không
    SUSPICIOUS — INCONCLUSIVE nếu có finding, NO_INDICATION nếu trống;
    fusion chỉ ghi signal, không tự kết tội ảnh.
    """
    t0 = time.perf_counter()
    present = doc.exif.get("present", {})
    missing = doc.exif.get("missing", [])
    findings: list[str] = []

    if doc.exif.get("tag_count", 0) == 0:
        findings.append(
            "ảnh không có EXIF — phổ biến sau re-save/qua mạng xã hội; "
            "không phải bằng chứng giả mạo"
        )
    sw = str(present.get("Software", ""))
    if sw:
        hit = next((h for h in EDITOR_HINTS if h in sw.lower()), None)
        findings.append(
            f"Software='{sw}' — ảnh đã qua "
            + ("công cụ chỉnh sửa ảnh" if hit else "phần mềm xử lý")
        )
    if "Make" in missing and "Model" in missing:
        findings.append("thiếu Make/Model — không xác định được thiết bị chụp")
    dt, dto = present.get("DateTime"), present.get("DateTimeOriginal")
    if dt and dto and dt != dto:
        findings.append(
            f"DateTime={dt} ≠ DateTimeOriginal={dto} — mốc thời gian bị ghi đè "
            "(thường do re-save, cũng có thể chỉnh tay)"
        )

    verdict = Verdict.INCONCLUSIVE if findings else Verdict.NO_INDICATION
    return AlgorithmResult(
        method=MethodName.EXIF_AUDIT,
        verdict=verdict,
        score=None,
        score_definition="",
        evidence={"findings": findings, "exif_present": present,
                  "exif_missing": missing},
        limitations=[
            ("metadata là dữ kiện khai báo — dễ chỉnh/xoá; finding chỉ gợi ý "
             "hướng điều tra, không đứng một mình"),
            "không có EXIF ≠ ảnh giả — nhiều pipeline hợp pháp xoá metadata",
        ],
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
