"""NAFNet inference, sidecar, and Enhance tab boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from imageguard.enhancement import nafnet
from imageguard.enhancement.derive import derive, write_sibling


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _valid_model() -> bool:
    path = Path(nafnet.MODEL_PATH)
    return (
        path.is_file()
        and path.stat().st_size == nafnet.MODEL_BYTES
        and nafnet._sha256_file(path) == nafnet.MODEL_SHA256
    )


class _FakeInput:
    name = "lq"


class _FakeSession:
    last_tensor: np.ndarray | None = None

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_tensor = inputs["lq"].copy()
        return [self.last_tensor]


def test_missing_model_raises_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.onnx"
    monkeypatch.setattr(nafnet, "MODEL_PATH", missing)
    image = np.zeros((32, 40, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="weight is missing"):
        derive(image, "nafnet", {})

    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_nafnet_inference_shape_determinism_and_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _valid_model():
        pytest.skip("pinned NAFNet weight is absent or does not match")

    session = _FakeSession()
    monkeypatch.setattr(
        nafnet,
        "_load_session",
        lambda *_args: (session, nafnet.MODEL_SHA256, True),
    )

    image = np.random.default_rng(20261106).integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )
    first, metadata = derive(image, "nafnet", {})
    second, repeated_metadata = derive(image, "nafnet", {})
    assert first.shape == image.shape
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert metadata["model_sha256"] == nafnet.MODEL_SHA256
    assert metadata["model_bytes"] == nafnet.MODEL_BYTES
    assert metadata["min_side"] == 512
    assert metadata["max_long_side"] == 1024
    assert metadata["original_spatial_size"] == [256, 256]
    assert metadata["network_spatial_size"] == [512, 512]
    assert metadata["pad_bottom"] == 256
    assert metadata["pad_right"] == 256
    assert metadata["view_only"] is True
    assert repeated_metadata["model_sha256"] == metadata["model_sha256"]
    canvas = cv2.copyMakeBorder(image, 0, 256, 0, 256, cv2.BORDER_REPLICATE)
    expected = (
        canvas[:, :, ::-1].astype(np.float32) * np.float32(0.00392)
    ).transpose(2, 0, 1)[None, ...]
    np.testing.assert_array_equal(session.last_tensor, expected)

    large = np.zeros((600, 1200, 3), dtype=np.uint8)
    resized, large_metadata = derive(large, "nafnet", {})
    assert resized.shape == large.shape
    assert large_metadata["original_spatial_size"] == [600, 1200]
    assert min(large_metadata["network_spatial_size"]) >= 512
    assert max(large_metadata["network_spatial_size"]) <= 1024

    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes(image))
    sidecar = write_sibling(
        source,
        tmp_path / "derived.png",
        first,
        "nafnet",
        {},
        hashlib.sha256(source.read_bytes()).hexdigest(),
        model_sha256=metadata["model_sha256"],
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["model_sha256"] == nafnet.MODEL_SHA256


def test_nafnet_study_marks_missing_runtime_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imageguard.evaluation import nafnet_arm

    monkeypatch.setattr(nafnet, "MODEL_PATH", tmp_path / "missing.onnx")
    payload = nafnet_arm.run_nafnet_arm(tmp_path / "out", n=1)

    assert payload["status"] == "NOT_RUN"
    assert "NAFNet weight is missing" in payload["reason"]
    report = (tmp_path / "out" / "nafnet_arm.md").read_text(encoding="utf-8")
    payload_json = (tmp_path / "out" / "nafnet_arm.json").read_text(encoding="utf-8")
    assert "NOT_RUN" in report
    assert str(tmp_path) not in report
    assert str(tmp_path) not in payload_json


def test_nafnet_arm_reuses_paired_protocol_with_stubbed_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imageguard.evaluation import enhance_study, nafnet_arm

    session = _FakeSession()
    model_metadata = {
        "model_sha256": nafnet.MODEL_SHA256,
        "model_bytes": nafnet.MODEL_BYTES,
        "max_long_side": 512,
        "original_spatial_size": [256, 256],
        "network_spatial_size": [256, 256],
        "view_only": True,
        "deterministic_compute": True,
    }
    monkeypatch.setattr(
        nafnet,
        "_load_session",
        lambda *_args: (session, nafnet.MODEL_SHA256, True),
    )
    monkeypatch.setattr(
        nafnet_arm, "_preflight", lambda: (0.01, model_metadata, None)
    )

    payload = nafnet_arm.run_nafnet_arm(tmp_path / "out", n=1)

    assert payload["status"] == "DONE"
    assert payload["n"] == 1
    assert payload["seed"] == 20261106
    assert payload["degradations"] == list(enhance_study.DEFAULT_DEGRADATIONS)
    assert payload["restorers"] == ["nafnet"]
    assert payload["methods"] == ["keypoint", "block", "ela", "noise"]
    assert {sample["role"] for sample in payload["samples"]} == {
        "manipulated_sample",
        "matched_original",
    }
    assert (tmp_path / "out" / "nafnet_arm.md").is_file()


def test_enhance_screen_nafnet_option_banner_and_source_boundary(
    qtbot,
) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QLabel

    from imageguard.ui import strings
    from imageguard.ui.screens.enhance import EnhanceScreen

    screen = EnhanceScreen()
    qtbot.addWidget(screen)
    index = screen.combo_op.findData("nafnet")
    assert index >= 0
    assert screen.combo_op.itemText(index) == "NAFNet (AI)"
    assert screen.ai_banner.objectName() == "enhance_ai_banner"
    assert screen.ai_banner.isHidden()
    assert not screen.params_box.isHidden()

    screen.combo_op.setCurrentIndex(index)
    assert not screen.ai_banner.isHidden()
    assert screen.params_box.isHidden()
    assert strings.ENHANCE_AI_BANNER in [
        label.text() for label in screen.ai_banner.findChildren(QLabel)
    ]
    assert screen._current_params() == {}
    screen.combo_op.setCurrentIndex(screen.combo_op.findData("unsharp"))
    assert screen.ai_banner.isHidden()
    assert not screen.params_box.isHidden()

    source = Path(__file__).resolve().parents[1] / "src/imageguard/ui/screens/enhance.py"
    assert "AnalysisService" not in source.read_text(encoding="utf-8")
