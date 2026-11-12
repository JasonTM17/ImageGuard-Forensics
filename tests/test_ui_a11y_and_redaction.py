"""Accessible names cho checkbox phương pháp và redact đường dẫn trong lỗi enhance."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from imageguard.ui import strings


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


_METHOD_CHECKBOXES = (
    ("chk_kp", "copy_move_keypoint"),
    ("chk_bl", "copy_move_block"),
    ("chk_ela", "ela"),
    ("chk_noise", "noise_inconsistency"),
    ("chk_nloc", "noise_local"),
    ("chk_freq", "frequency_analysis"),
    ("chk_spl", "splicing_cues"),
)


def test_analyze_method_checkboxes_distinct_accessible_names(qtbot, services):
    from imageguard.ui.main_window import MainWindow

    analysis, history = services
    w = MainWindow(analysis, history)
    qtbot.addWidget(w)
    a = w.analyze
    seen: set[str] = set()
    for attr, method_key in _METHOD_CHECKBOXES:
        chk = getattr(a, attr)
        label = strings.METHOD_LABEL[method_key]
        expected = strings.FMT_A11Y_ANALYZE_METHOD.format(
            group=strings.A11Y_ANALYZE_METHODS,
            label=label,
        )
        assert chk.accessibleName() == expected
        assert chk.accessibleName() != strings.A11Y_ANALYZE_METHODS
        seen.add(chk.accessibleName())
    assert len(seen) == len(_METHOD_CHECKBOXES)


class _Exc(Exception):
    pass


@pytest.mark.parametrize(
    ("message", "expected_tail"),
    [
        (
            r"C:\Users\Jane Doe\Project Dir\file.png",
            "file.png",
        ),
        (
            'Permission denied: "C:\\Users\\Jane Doe\\secret.png"',
            'Permission denied: secret.png',
        ),
        (
            "cannot read '/home/jane/My Documents/scan.tiff'",
            "cannot read scan.tiff",
        ),
        (
            r"Open \\server\share\Team Dir\image.jpeg failed",
            "Open image.jpeg failed",
        ),
        (
            "Kernel size must be odd",
            "Kernel size must be odd",
        ),
        (
            "Division by zero",
            "Division by zero",
        ),
    ],
)
def test_sanitize_exception_text_paths(message, expected_tail):
    from imageguard.ui.screens.enhance import _sanitize_exception_text

    out = _sanitize_exception_text(_Exc(message))
    assert out == f"_Exc: {expected_tail}"
    assert "Jane Doe" not in out
    assert "Project Dir" not in out
    assert "\\\\server" not in out


def test_redact_paths_in_message_rejects_none():
    from imageguard.ui.screens.enhance import _redact_paths_in_message

    with pytest.raises(ValueError, match="message must be a string"):
        _redact_paths_in_message(None)  # type: ignore[arg-type]


def test_sanitize_exception_empty_uses_generic():
    from imageguard.ui.screens.enhance import _sanitize_exception_text

    class _EmptyStrExc(Exception):
        def __str__(self) -> str:
            return "   "

    out = _sanitize_exception_text(_EmptyStrExc())
    assert out.endswith(strings.MSG_EXCEPTION_PATH_REDACTED)


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\Jane Doe\Project Dir\file.png",
        "/home/jane/My Documents/nested/file.png",
    ],
)
def test_redact_paths_basename_only(path):
    from imageguard.ui.screens.enhance import _redact_paths_in_message

    redacted = _redact_paths_in_message(f"failed opening {path}")
    assert redacted == "failed opening file.png"
    assert "Jane" not in redacted
    assert "Documents" not in redacted
