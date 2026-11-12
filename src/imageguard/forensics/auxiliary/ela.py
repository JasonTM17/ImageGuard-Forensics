"""Error Level Analysis — chỉ có nghĩa với JPEG.

Nguyên lý: vùng ảnh được nén JPEG lần đầu có error level riêng; vùng bị chèn
sau đó (từ ảnh khác / chỉnh sửa rồi re-save) có error level khác. Re-encode
ảnh ở quality q rồi đo |I − R_q(I)| — vùng khác "lịch sử nén" lộ ra.

Giới hạn CỨNG (ghi vào mọi result):
- Ảnh đã re-save/save lại → vệt ELA đều nhau → vô nghĩa.
- Vùng phẳng có ELA thấp tự nhiên; cạnh sắc có ELA cao tự nhiên — heatmap
  KHÔNG phải bản đồ giả mạo.
- ELA không bao giờ cho SUSPICIOUS một mình — tối đa INCONCLUSIVE + đề nghị
  corroboration bằng method khác.
"""

from __future__ import annotations

import threading
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.base import cancelled

SCORE_DEF = "p99 của heatmap ELA (max qua sweep quality), scale [0,1] đã amplify"


class ELA:
    """Error Level Analysis — chỉ JPEG; re-save ở nhiều quality rồi lấy diff.

    Chủ đề: JPEG / nén có mất (artifact). **Aux**: không bao giờ
    ``SUSPICIOUS_INDICATION`` — tối đa INCONCLUSIVE khi p99 heat vượt ngưỡng;
    non-JPEG → NOT_APPLICABLE.
    """

    name = MethodName.ELA.value

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        """Sweeps quality: lưu JPEG q, đọc lại, diff trung bình kênh RGB.

        Score = p99 heat (sau amplify). Verdict INCONCLUSIVE nếu p99 ≥
        ``indication_p99``, ngược lại NO_INDICATION — không đẩy fusion lên
        SUSPICIOUS một mình.
        """
        t0 = time.perf_counter()
        base_lim = [
            ("ELA chỉ là bằng chứng khám phá — heatmap cao có thể do cạnh sắc/"
             "texture tự nhiên, không đồng nghĩa vùng bị chỉnh sửa"),
            "mất ý nghĩa nếu ảnh đã re-save nhiều lần sau chỉnh sửa",
        ]
        if doc.image_format != "jpeg":
            return self._done(
                Verdict.NOT_APPLICABLE, None, t0,
                [("ELA phân tích JPEG compression artifacts — ảnh "
                  f"{doc.image_format} lossless/khác định dạng không có tín hiệu")],
            )

        rgb = cv2.cvtColor(doc.array, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        qualities = list(cfg.value("ela", "qualities"))
        amplify = float(cfg.value("ela", "amplify"))
        maps: dict[int, np.ndarray] = {}
        for q in qualities:
            if cancelled(cancel):
                return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled mid-sweep"])
            buf = BytesIO()
            pil.save(buf, "JPEG", quality=int(q))
            buf.seek(0)
            with Image.open(buf) as im:
                re = np.asarray(im.convert("RGB"), dtype=np.float32)
            diff = np.abs(rgb.astype(np.float32) - re).mean(axis=2)
            # Amplify rồi clip về [0,1] — scale tuyệt đối để p99 so sánh được.
            maps[int(q)] = np.clip(diff * amplify / 255.0, 0, 1)

        heat = np.maximum.reduce(list(maps.values()))
        p99 = float(np.percentile(heat, 99))
        ind = float(cfg.value("ela", "indication_p99"))
        verdict = Verdict.INCONCLUSIVE if p99 >= ind else Verdict.NO_INDICATION
        if p99 >= ind:
            base_lim.append(
                f"p99={p99:.3f} ≥ {ind} — có vùng error-level bất thường; "
                "cần corroborate bằng keypoint/block trước khi nghi ngờ"
            )
        mask = np.where(heat >= 0.5, 255, 0).astype(np.uint8)
        return self._done(
            verdict, p99, t0, base_lim,
            mask=mask, heatmap=heat,
            evidence={
                "qualities": qualities,
                "p99": round(p99, 4),
                "per_quality_p99": {q: round(float(np.percentile(m, 99)), 4)
                                    for q, m in maps.items()},
            },
            params_hash=cfg.snapshot_hash(),
        )

    def _done(self, verdict, score, t0, limitations, evidence=None, mask=None,
              heatmap=None, params_hash=""):
        return AlgorithmResult(
            method=MethodName.ELA, verdict=verdict, score=score,
            score_definition=SCORE_DEF, mask=mask, heatmap=heatmap,
            evidence=evidence or {}, limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
