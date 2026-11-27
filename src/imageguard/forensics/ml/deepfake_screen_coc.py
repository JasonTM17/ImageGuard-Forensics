"""Screen-recapture deepfake detection — CoC Xception (ONNX, CPU, offline).

``deepfake_screen_coc`` dùng checkpoint Xception do nhóm ScreenDeepfakeBench
phát hành cho bài toán video deepfake phát trên màn hình rồi bị camera quay
lại (Caught on Camera). Khác GenConViT đa miền, model này đúng domain camera
recapture hơn nhưng **không** được xem là detector deepfake tổng quát.

Pipeline khớp bản evaluate của tác giả ở các điểm chính:

1. YuNet định vị mặt và 5 landmark; landmark được căn chỉnh similarity với
   margin scale 1.3 theo preprocessing DeepfakeBench/CoC.
2. Crop đã align 256×256 rồi resize 299×299, giữ thứ tự kênh BGR như pipeline OpenCV gốc.
3. ``/255`` rồi chuẩn hoá mean/std 0.5; ONNX output là 2 logits,
   ``softmax[..., 1]`` là điểm fake heuristic.

Checkpoint PyTorch gốc được chuyển ONNX cục bộ và pin SHA-256; file ONNX không
commit và không tải lúc runtime. Licence/trọng số: xem
``docs/third-party-models.md``. Model gốc CC BY-NC 4.0, chỉ phù hợp mục đích
phi thương mại/nghiên cứu.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.paths import model_root
from imageguard.forensics.base import cancelled
from imageguard.forensics.ml.face_detect import (
    detect_primary_face_details,
    sha256_file,
    yunet_integrity_error,
)

MODEL_PATH = model_root() / "models" / "coc_xception_screen_deepfake.onnx"
MODEL_BYTES = 83_165_816
MODEL_SHA256 = "800dff109c5bbb165b9b0404e1e1207bc521ec75fc51926247cbae2abe271b1e"
INPUT_SIZE = 299

SCORE_DEF = (
    "fake_prob = softmax(logits)[fake] của CoC Xception trên ảnh mặt align "
    "299×299 — điểm heuristic của classifier, không phải xác suất đã hiệu chuẩn"
)
INSTALL_HINT = (
    "chưa cài model CoC screen-recapture — tạo/đặt coc_xception_screen_deepfake.onnx "
    "đúng sha256 pin vào models/ theo docs/third-party-models.md"
)


def model_integrity_error() -> str | None:
    path = Path(MODEL_PATH)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != MODEL_BYTES:
        return "size-mismatch"
    if sha256_file(path) != MODEL_SHA256:
        return "sha256-mismatch"
    return None


@lru_cache(maxsize=1)
def _session() -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _alignment_destination(size: int) -> np.ndarray:
    """5 điểm đích của DeepfakeBench alignment cho output ``size`` và scale 1.3."""
    dst = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float64,
    )
    dst[:, 0] += 8.0
    dst *= size / 112.0
    margin = size * 0.3 / 2.0
    dst += margin
    dst *= size / (size + 2.0 * margin)
    return dst


def align_face_coc(bgr: np.ndarray, landmarks: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray | None:
    """Similarity-align 5 landmark theo preprocessing CoC; trả crop BGR ``size×size``.

    OpenCV ``estimateAffinePartial2D`` là phương thức tương đương similarity
    transform của pipeline gốc. Nếu ước lượng thất bại, method phải xử lý như
    không đủ điều kiện áp dụng thay vì fallback sang crop chặt sai domain.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(pts, _alignment_destination(size))
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return None
    return cv2.warpAffine(bgr, matrix, (size, size), flags=cv2.INTER_LINEAR)


def preprocess_aligned(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 299×299 → tensor (1,3,299,299), normalize mean/std 0.5."""
    if crop_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        crop_bgr = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = np.transpose(crop_bgr.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
    return (arr - np.float32(0.5)) / np.float32(0.5)


def infer_fake_prob(tensor: np.ndarray) -> float:
    (logits,) = _session().run(["logits"], {"image_bgr_normalized": tensor})
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (tensor.shape[0], 2)
        or not np.issubdtype(logits.dtype, np.floating)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid CoC logits")
    row = logits[0].astype(np.float64)
    exp = np.exp(row - row.max())
    return float((exp / exp.sum())[1])


def verdict_for_score(fake_prob: float, fake_thresh: float, real_thresh: float) -> Verdict:
    if fake_prob >= fake_thresh:
        return Verdict.SUSPICIOUS_INDICATION
    if fake_prob <= real_thresh:
        return Verdict.NO_INDICATION
    return Verdict.INCONCLUSIVE


class DeepfakeScreenCoC:
    """Method registry entry — spec ``deepfake_screen``."""

    name = "deepfake_screen"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            "model Xception được train cho deepfake phát qua màn hình/camera quay lại — không phải detector deepfake tổng quát",
            "score là heuristic của classifier, không phải xác suất đã hiệu chuẩn",
            "SUSPICIOUS_INDICATION là dấu hiệu nghi vấn cần đối chiếu — không phải bằng chứng giả mạo",
        ]
        evidence: dict[str, Any] = {"variant": "coc_xception_onnx", "model_available": False}

        integrity = model_integrity_error()
        if integrity is not None:
            if integrity == "missing":
                return self._done(Verdict.NOT_APPLICABLE, None, t0, [INSTALL_HINT] + lim, evidence)
            return self._done(
                Verdict.ERROR,
                None,
                t0,
                [f"file model CoC sai integrity ({integrity}) — tạo/đặt lại đúng bản pin"] + lim,
                evidence,
            )
        yunet_err = yunet_integrity_error()
        if yunet_err is not None:
            note = (
                "chưa cài model YuNet face detection — xem docs/third-party-models.md"
                if yunet_err == "missing"
                else f"file YuNet sai integrity ({yunet_err}) — tải lại đúng bản pin"
            )
            verdict = Verdict.NOT_APPLICABLE if yunet_err == "missing" else Verdict.ERROR
            return self._done(verdict, None, t0, [note] + lim, evidence)
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return self._done(
                Verdict.NOT_APPLICABLE,
                None,
                t0,
                ["thiếu onnxruntime — cài extra 'lama' hoặc 'pip install onnxruntime'"] + lim,
                evidence,
            )
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim, evidence)

        detection = detect_primary_face_details(doc.array, int(cfg.value("deepfake_screen_coc", "detect_max_side")))
        if detection is None:
            return self._done(
                Verdict.NOT_APPLICABLE,
                None,
                t0,
                ["không phát hiện khuôn mặt đủ lớn — model chỉ áp dụng cho ảnh có mặt người"] + lim,
                {**evidence, "face_detected": False},
            )
        rect, det_conf, landmarks = detection
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim, evidence)

        aligned = align_face_coc(doc.array, landmarks, size=256)
        if aligned is None:
            return self._done(Verdict.NOT_APPLICABLE, None, t0,
                ["không căn chỉnh được 5 landmark theo preprocessing CoC"] + lim,
                {**evidence, "face_detected": True, "alignment": "failed"})
        try:
            fake_prob = infer_fake_prob(preprocess_aligned(aligned))
        except Exception as exc:  # noqa: BLE001 - inference boundary: hide backend details
            return self._done(Verdict.ERROR, None, t0,
                ["CoC Xception không trả được đầu ra hợp lệ"] + lim,
                {**evidence, "error_stage": "inference", "error_type": type(exc).__name__})

        fake_thresh = float(cfg.value("deepfake_screen_coc", "fake_thresh"))
        real_thresh = float(cfg.value("deepfake_screen_coc", "real_thresh"))
        verdict = verdict_for_score(fake_prob, fake_thresh, real_thresh)
        evidence.update({
            "model_available": True,
            "face_detected": True,
            "roi_rect": [int(v) for v in rect],
            "detector": "yunet_2023mar",
            "detector_confidence": round(det_conf, 4),
            "alignment": "coc_5point_similarity_scale_1_3",
            "fake_prob": round(fake_prob, 4),
            "model_sha256_16": MODEL_SHA256[:16],
            "fake_thresh": fake_thresh,
            "real_thresh": real_thresh,
        })
        mask = None
        if verdict in {Verdict.SUSPICIOUS_INDICATION, Verdict.INCONCLUSIVE}:
            mask = np.zeros(doc.array.shape[:2], dtype=np.uint8)
            x, y, w, h = rect
            mask[y:y + h, x:x + w] = 255
        if verdict is Verdict.SUSPICIOUS_INDICATION:
            lim.insert(0, f"classifier screen-recapture ghi nhận dấu hiệu mạnh trên vùng mặt (fake_prob={fake_prob:.2f} ≥ {fake_thresh:.2f}) — cần đối chiếu chuyên gia")
        elif verdict is Verdict.INCONCLUSIVE:
            lim.insert(0, f"score vùng xám (fake_prob={fake_prob:.2f} trong ({real_thresh:.2f}, {fake_thresh:.2f})) — không đủ cơ sở kết luận")
        return self._done(verdict, round(fake_prob, 4), t0, lim, evidence, mask=mask, params_hash=cfg.snapshot_hash())

    def _done(self, verdict: Verdict, score: float | None, t0: float,
              limitations: list[str], evidence: dict | None = None,
              mask: np.ndarray | None = None, params_hash: str = "") -> AlgorithmResult:
        return AlgorithmResult(
            method=MethodName.DEEPFAKE_SCREEN_COC,
            verdict=verdict,
            score=score,
            score_definition=SCORE_DEF,
            mask=mask,
            evidence=evidence or {},
            limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
