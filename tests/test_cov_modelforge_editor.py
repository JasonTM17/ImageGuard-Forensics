"""Coverage gaps: evaluation.modelforge.make_editor factory."""

from __future__ import annotations

import pytest

from imageguard.evaluation.modelforge import (
    MODEL_PATH,
    ClassicEditor,
    LaMaEditor,
    make_editor,
)


def test_make_editor_returns_classic_for_non_lama_backends() -> None:
    for backend in ("classic", "", "telea", "unknown"):
        editor = make_editor(backend)
        assert isinstance(editor, ClassicEditor)
        assert not isinstance(editor, LaMaEditor)


def test_make_editor_lama_requires_model_file() -> None:
    if MODEL_PATH.is_file():
        editor = make_editor("lama")
        assert isinstance(editor, LaMaEditor)
    else:
        with pytest.raises(FileNotFoundError, match="không thấy"):
            make_editor("lama")
