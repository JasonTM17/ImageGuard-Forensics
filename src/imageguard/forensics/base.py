"""Method protocol — hợp đồng chung mọi phương pháp forensic phải tuân."""

from __future__ import annotations

import threading
from typing import Protocol

from imageguard.core.config import ThresholdSet
from imageguard.core.models import AlgorithmResult, ImageDocument


class Method(Protocol):
    """Protocol tối thiểu: nhận ImageDocument + thresholds, trả AlgorithmResult.

    `cancel` là cooperative cancellation — method KIỂM TRA giữa các stage
    (không bắt buộc mọi vòng lặp) và trả sớm verdict hiện có hoặc INCONCLUSIVE
    kèm limitations nếu bị huỷ giữa chừng.
    """

    name: str

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        ...


def cancelled(cancel: threading.Event | None) -> bool:
    return cancel is not None and cancel.is_set()
