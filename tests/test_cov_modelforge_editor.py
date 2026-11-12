"""Coverage gaps: evaluation.modelforge.make_editor factory."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from imageguard.evaluation.modelforge import (
    MODEL_PATH,
    ClassicEditor,
    LaMaEditor,
    make_editor,
)


def _install_stub_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub onnxruntime so LaMaEditor can reach the model-path check without the [lama] extra."""

    def _session_options() -> Any:
        return types.SimpleNamespace(log_severity_level=0)

    def _inference_session(*_args: object, **_kwargs: object) -> object:
        return object()

    fake = types.ModuleType("onnxruntime")
    fake.SessionOptions = _session_options  # type: ignore[attr-defined]
    fake.InferenceSession = _inference_session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


def test_make_editor_returns_classic_for_non_lama_backends() -> None:
    for backend in ("classic", "", "telea", "unknown"):
        editor = make_editor(backend)
        assert isinstance(editor, ClassicEditor)
        assert not isinstance(editor, LaMaEditor)


def test_make_editor_lama_requires_model_file(monkeypatch: pytest.MonkeyPatch) -> None:
    if MODEL_PATH.is_file():
        pytest.importorskip("onnxruntime")
        editor = make_editor("lama")
        assert isinstance(editor, LaMaEditor)
        return

    _install_stub_onnxruntime(monkeypatch)
    with pytest.raises(FileNotFoundError, match="không thấy"):
        make_editor("lama")
