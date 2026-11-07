"""Export confirmation names both files actually written."""

import cv2

from imageguard.ui.screens.enhance import EnhanceScreen


def test_export_confirmation_names_png_and_sidecar(qtbot, valid_png, tmp_path, monkeypatch):
    from imageguard.ui.screens import enhance

    screen = EnhanceScreen()
    qtbot.addWidget(screen)
    screen.load_path(valid_png)
    screen._derived = screen._doc.array.copy()
    screen._run_kind = "unsharp"
    screen._run_params = {}
    dest = tmp_path / "exported.png"
    monkeypatch.setattr(
        enhance.QFileDialog, "getSaveFileName", lambda *a, **k: (str(dest), "PNG")
    )
    messages = []
    monkeypatch.setattr(
        enhance.QMessageBox, "information", lambda parent, title, text: messages.append(text)
    )
    screen._export()
    assert dest.is_file()
    assert cv2.imread(str(dest)) is not None
    assert dest.with_suffix(".json").is_file()
    assert len(messages) == 1
    assert dest.name in messages[0]
    assert dest.with_suffix(".json").name in messages[0]
