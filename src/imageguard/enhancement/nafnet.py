"""Optional, view-only NAFNet inference through ONNX Runtime."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "deblurring_nafnet_2025may.onnx"
)
MODEL_BYTES = 91_736_251
MODEL_SHA256 = "07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587"
MIN_SIDE = 512
MAX_LONG_SIDE = 1024
_LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_session(
    path_string: str, modified_ns: int, file_size: int
) -> tuple[Any, str, bool]:
    del modified_ns  # The timestamp is part of the cache identity.
    path = Path(path_string)
    if file_size != MODEL_BYTES:
        raise ValueError("NAFNet weight has an unexpected size")
    model_sha256 = _sha256_file(path)
    if model_sha256 != MODEL_SHA256:
        raise ValueError("NAFNet weight SHA-256 does not match the pinned model")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "NAFNet requires the optional onnxruntime package; "
            "install the existing 'lama' extra."
        ) from exc

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    deterministic_compute = True
    try:
        options.use_deterministic_compute = True
    except (AttributeError, TypeError):
        deterministic_compute = False
        _LOGGER.warning(
            "Installed onnxruntime does not support use_deterministic_compute; "
            "continuing with one intra-op and inter-op thread."
        )

    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session, model_sha256, deterministic_compute


def _session() -> tuple[Any, str, bool]:
    path = Path(MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            "NAFNet weight is missing: deblurring_nafnet_2025may.onnx"
        )
    stat = path.stat()
    return _load_session(str(path), stat.st_mtime_ns, stat.st_size)


def _network_canvas(source: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Fit an image to the spatial range this ONNX export can run.

    A 256-pixel side collapses inside the graph (zero-sized Pad). A 512-pixel
    square completes. The canvas therefore keeps every side at least 512 and
    the long side at most 1024, using replicate padding when a downscale would
    otherwise leave the short side too small.
    """
    height, width = source.shape[:2]
    scale = 1.0
    long_side = max(height, width)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
    resized_width = max(1, int(np.floor(width * scale)))
    resized_height = max(1, int(np.floor(height * scale)))
    if (resized_height, resized_width) == (height, width):
        resized = source
    else:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(
            source,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
    pad_bottom = max(0, MIN_SIDE - resized_height)
    pad_right = max(0, MIN_SIDE - resized_width)
    if pad_bottom or pad_right:
        canvas = cv2.copyMakeBorder(
            resized,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
    else:
        canvas = resized
    return canvas, {
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
    }


def infer(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one deterministic CPU inference and return a BGR viewing image."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("NAFNet expects a three-channel BGR image")
    if source.dtype != np.uint8:
        raise ValueError("NAFNet expects uint8 image pixels")
    source = np.ascontiguousarray(source)
    height, width = source.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("NAFNet expects a non-empty image")

    network_image, fit = _network_canvas(source)
    network_height, network_width = network_image.shape[:2]

    rgb = cv2.cvtColor(network_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = (rgb * np.float32(0.00392)).transpose(2, 0, 1)[None, ...]
    session, model_sha256, deterministic_compute = _session()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    output_hwc = np.asarray(output)[0].transpose(1, 2, 0)
    if output_hwc.shape[:2] != (network_height, network_width):
        raise ValueError("NAFNet returned an unexpected spatial size")
    rgb_result = np.clip(output_hwc * 255.0, 0, 255).astype(np.uint8)
    bgr_result = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)
    bgr_result = bgr_result[: fit["resized_height"], : fit["resized_width"]]
    if (fit["resized_height"], fit["resized_width"]) != (height, width):
        bgr_result = cv2.resize(
            bgr_result,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    metadata = {
        "model_sha256": model_sha256,
        "model_bytes": MODEL_BYTES,
        "min_side": MIN_SIDE,
        "max_long_side": MAX_LONG_SIDE,
        "original_spatial_size": [height, width],
        "network_spatial_size": [network_height, network_width],
        "pad_bottom": fit["pad_bottom"],
        "pad_right": fit["pad_right"],
        "view_only": True,
        "deterministic_compute": deterministic_compute,
    }
    return bgr_result, metadata
