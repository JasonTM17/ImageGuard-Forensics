"""Deepfake face-manipulation detection — GenConViT-ED (ONNX, CPU, offline).

Đây là method ML đầu tiên của pipeline và khác hẳn ``deepfake_cues``
(heuristic 6 trụ cột, không bao giờ SUSPICIOUS): GenConViT-ED là classifier
đã train trên DFDC / FaceForensics++ / Celeb-DF v2 / DeepfakeTIMIT (Wodajo
et al., Applied Sciences 2025), input là crop khuôn mặt 224×224, output là
2 logits [fake, real]. Vì là detector học từ dữ liệu deepfake thật, method
này CÓ THỂ trả ``SUSPICIOUS_INDICATION`` khi điểm vượt ngưỡng trên — vẫn là
*dấu hiệu nghi vấn*, không phải phán quyết sự thật.

Preprocessing giữ đúng pipeline gốc (erprogs/GenConViT, MIT):

1. Crop tight face rect → resize 224×224 ``INTER_AREA`` → RGB.
2. ``/255`` rồi chuẩn hoá ImageNet (mean/std).
3. ``sigmoid`` cả 2 logits; ``fake_frac = s_fake / (s_fake + s_real)`` —
   > 0.5 tương đương quy tắc argmax + xor của tác giả.

Sai lệch đã biết so với pipeline gốc: face detection dùng YuNet thay vì
dlib/face_recognition; YuNet là dependency sẵn trong OpenCV của dự án.
Score là heuristic của classifier, KHÔNG phải xác suất đã hiệu chuẩn.

Model là tuỳ chọn, pin SHA-256, CPU-only, không bao giờ tải lúc runtime.
Licence: code GenConViT MIT; weights gốc trên HuggingFace ghi CC BY-NC 4.0
(chỉ dùng phi thương mại) — xem ``docs/third-party-models.md``.
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
    crop_face,
    detect_primary_face,
    sha256_file,
    yunet_integrity_error,
)

MODEL_PATH = model_root() / "models" / "genconvit_ed_inference.onnx"
MODEL_BYTES = 122_433_215
MODEL_SHA256 = "59605cda6b03322d1c2d892b4ba058d0a004b6c79b886fdbc2f420c897f7b9ae"

INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

SCORE_DEF = (
    "fake_frac = sigmoid(logit_fake) / (sigmoid(logit_fake) + sigmoid(logit_real)) "
    "của GenConViT-ED trên crop mặt 224×224 — điểm heuristic của classifier, "
    "không phải xác suất đã hiệu chuẩn"
)

INSTALL_HINT = (
    "chưa cài model deepfake ML — tải genconvit_ed_inference.onnx "
    "(pin sha256) vào models/ theo docs/third-party-models.md"
)


def model_integrity_error() -> str | None:
    """None nếu model hợp lệ; 'missing' | 'size-mismatch' | 'sha256-mismatch'."""
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
    """Load ONNX session CPU deterministic (lru_cache: load một lần/process)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )


def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 crop bất kỳ → tensor (1,3,224,224) float32 chuẩn hoá ImageNet.

    Giữ đúng thứ tự pipeline gốc: resize INTER_AREA trước, đổi RGB sau,
    scale /255 rồi (x − mean) / std theo kênh.
    """
    if crop_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        crop_bgr = cv2.resize(
            crop_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
        )
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    arr = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
    return (arr - _MEAN) / _STD


def infer_sigmoids(tensor: np.ndarray) -> np.ndarray:
    """Trả sigmoid logits (N,2) — cột 0 = fake, cột 1 = real (quy ước tác giả)."""
    (logits,) = _session().run(["output"], {"input": tensor})
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (tensor.shape[0], 2)
        or not np.issubdtype(logits.dtype, np.floating)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid GenConViT logits")
    with np.errstate(over="ignore", under="ignore"):
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def aggregate_fake_frac(sigmoids: np.ndarray) -> tuple[float, float, float]:
    """Trung bình sigmoid qua các frame/crop rồi chuẩn hoá tỉ lệ fake.

    Trả (fake_frac, mean_sig_fake, mean_sig_real). fake_frac > 0.5 tương
    đương quyết định FAKE của pipeline gốc.
    """
    mean_sig = sigmoids.mean(axis=0)
    total = float(mean_sig[0] + mean_sig[1])
    frac = float(mean_sig[0] / total) if total > 0 else 0.5
    return frac, float(mean_sig[0]), float(mean_sig[1])


def verdict_for_score(fake_frac: float, fake_thresh: float, real_thresh: float) -> Verdict:
    if fake_frac >= fake_thresh:
        return Verdict.SUSPICIOUS_INDICATION
    if fake_frac <= real_thresh:
        return Verdict.NO_INDICATION
    return Verdict.INCONCLUSIVE


class DeepfakeGenConViT:
    """Method registry entry — spec ``deepfake_ml``."""

    name = "deepfake_ml"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            "classifier học sâu GenConViT-ED — score là heuristic, không phải xác suất đã hiệu chuẩn",
            "hiệu năng ngoài phân phối train (generator/codec/nén mạng xã hội mới) chưa được đảm bảo",
            "SUSPICIOUS_INDICATION là dấu hiệu nghi vấn cần đối chiếu — không phải bằng chứng giả mạo",
        ]
        evidence: dict[str, Any] = {"variant": "genconvit_ed_onnx", "model_available": False}

        integrity = model_integrity_error()
        if integrity is not None:
            if integrity == "missing":
                return self._done(
                    Verdict.NOT_APPLICABLE, None, t0, [INSTALL_HINT] + lim, evidence
                )
            return self._done(
                Verdict.ERROR,
                None,
                t0,
                [f"file model deepfake ML sai integrity ({integrity}) — tải lại đúng bản pin"] + lim,
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

        detect_max = int(cfg.value("deepfake_ml", "detect_max_side"))
        detection = detect_primary_face(doc.array, detect_max)
        if detection is None:
            return self._done(
                Verdict.NOT_APPLICABLE,
                None,
                t0,
                ["không phát hiện khuôn mặt đủ lớn — model chỉ áp dụng cho ảnh có mặt người"] + lim,
                {**evidence, "face_detected": False},
            )

        rect, det_conf = detection
        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim, evidence)

        crop = crop_face(doc.array, rect, INPUT_SIZE)
        tensor = preprocess_crop(crop)
        try:
            sigmoids = infer_sigmoids(tensor)
        except Exception as exc:  # noqa: BLE001 - inference boundary: hide backend details
            return self._done(
                Verdict.ERROR,
                None,
                t0,
                ["GenConViT-ED không trả được đầu ra hợp lệ"] + lim,
                {
                    **evidence,
                    "error_stage": "inference",
                    "error_type": type(exc).__name__,
                },
            )
        fake_frac, sig_fake, sig_real = aggregate_fake_frac(sigmoids)

        fake_thresh = float(cfg.value("deepfake_ml", "fake_thresh"))
        real_thresh = float(cfg.value("deepfake_ml", "real_thresh"))
        verdict = verdict_for_score(fake_frac, fake_thresh, real_thresh)

        evidence.update(
            {
                "model_available": True,
                "face_detected": True,
                "roi_rect": [int(v) for v in rect],
                "detector": "yunet_2023mar",
                "detector_confidence": round(det_conf, 4),
                "fake_frac": round(fake_frac, 4),
                "mean_sig_fake": round(sig_fake, 4),
                "mean_sig_real": round(sig_real, 4),
                "frames_aggregated": 1,
                "model_sha256_16": MODEL_SHA256[:16],
                "fake_thresh": fake_thresh,
                "real_thresh": real_thresh,
            }
        )

        mask = None
        if verdict in {Verdict.SUSPICIOUS_INDICATION, Verdict.INCONCLUSIVE}:
            mask = np.zeros(doc.array.shape[:2], dtype=np.uint8)
            x, y, w, h = rect
            mask[y : y + h, x : x + w] = 255

        if verdict is Verdict.SUSPICIOUS_INDICATION:
            lim.insert(
                0,
                f"classifier ghi nhận dấu hiệu deepfake mạnh trên vùng mặt "
                f"(fake_frac={fake_frac:.2f} ≥ {fake_thresh:.2f}) — cần đối chiếu chuyên gia",
            )
        elif verdict is Verdict.INCONCLUSIVE:
            lim.insert(
                0,
                f"score vùng xám (fake_frac={fake_frac:.2f} trong "
                f"({real_thresh:.2f}, {fake_thresh:.2f})) — không đủ cơ sở kết luận",
            )

        return self._done(
            verdict,
            round(fake_frac, 4),
            t0,
            lim,
            evidence,
            mask=mask,
            params_hash=cfg.snapshot_hash(),
        )

    def _done(
        self,
        verdict: Verdict,
        score: float | None,
        t0: float,
        limitations: list[str],
        evidence: dict | None = None,
        mask: np.ndarray | None = None,
        params_hash: str = "",
    ) -> AlgorithmResult:
        return AlgorithmResult(
            method=MethodName.DEEPFAKE_ML,
            verdict=verdict,
            score=score,
            score_definition=SCORE_DEF,
            mask=mask,
            evidence=evidence or {},
            limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
