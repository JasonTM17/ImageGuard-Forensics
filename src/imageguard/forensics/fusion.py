"""Fusion — luật kết hợp verdict minh bạch, KHÔNG weighted-score hộp đen.

Bảng luật (docs/evaluation-methodology.md là bản chính thức):
  1. NOT_APPLICABLE → loại khỏi xét (method không áp dụng cho ảnh này).
  2. ERROR → warning riêng, không tính vào quyết định.
  3. ≥1 SUSPICIOUS_INDICATION trong methods còn lại → SUSPICIOUS + liệt kê.
  4. Tất cả methods chạy được = NO_INDICATION (≥1) → NO_INDICATION.
  5. Mọi trường hợp còn lại → INCONCLUSIVE.

Nguyên tắc: chỉ detector conclusive (keypoint/block) mới đẩy được lên
SUSPICIOUS; aux signals (ELA/EXIF) không bao giờ trả SUSPICIOUS nên không thể
tự kết tội ảnh — thiết kế có chủ đích.
"""

from __future__ import annotations

from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult


def combine(
    results: list[AlgorithmResult],
) -> tuple[Verdict, list[str], list[str]]:
    """Trả (overall_verdict, signals, warnings)."""
    signals: list[str] = []
    warnings: list[str] = []
    active = [r for r in results if r.verdict != Verdict.NOT_APPLICABLE]
    usable = []
    for r in active:
        if r.verdict == Verdict.ERROR:
            warnings.append(f"{r.method.value}: method trả ERROR — loại khỏi fusion")
        else:
            usable.append(r)

    suspicious = [r.method.value for r in usable
                  if r.verdict == Verdict.SUSPICIOUS_INDICATION]
    if suspicious:
        signals.extend(suspicious)
        return Verdict.SUSPICIOUS_INDICATION, signals, warnings

    if usable and all(r.verdict == Verdict.NO_INDICATION for r in usable):
        return Verdict.NO_INDICATION, signals, warnings

    if not usable:
        warnings.append("không method nào chạy được — toàn NOT_APPLICABLE/ERROR")
    return Verdict.INCONCLUSIVE, signals, warnings
