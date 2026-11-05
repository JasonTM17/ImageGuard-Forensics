"""UI app-args tests — app.main(initial_image=...) và sys.argv.

main() gọi app.exec() — patch QApplication để exec() trả 0, không start
event loop thật. MainWindow được bắt lại qua factory để kiểm tra nhánh
analyze đã nạp ảnh hay chưa.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


@pytest.fixture()
def services(tmp_path):
    from imageguard.ui.app import build_services

    return build_services(db_path=tmp_path / "t.db",
                          artifacts_root=tmp_path / "art")


class _FakeApp:
    """QApplication stand-in — exec() trả 0 để main() trả về ngay."""

    def __init__(self, argv):
        self.argv = argv

    @staticmethod
    def instance():
        return None

    def exec(self):
        return 0


@pytest.fixture()
def main_env(monkeypatch, services):
    """Patch QApplication/build_services + bắt MainWindow mà main() tạo."""
    import imageguard.ui.app as app_mod

    captured = {}
    real_window = app_mod.MainWindow

    def _capture(analysis, history):
        win = real_window(analysis, history)
        captured["window"] = win
        return win

    monkeypatch.setattr(app_mod, "QApplication", _FakeApp)
    monkeypatch.setattr(app_mod, "build_services", lambda: services)
    monkeypatch.setattr(app_mod, "MainWindow", _capture)
    return captured


def _run_main(monkeypatch, argv, initial_image):
    import imageguard.ui.app as app_mod

    monkeypatch.setattr(sys, "argv", list(argv))
    return app_mod.main(initial_image=initial_image)


def test_main_loads_initial_image_existing_file(qtbot, main_env, valid_png, monkeypatch):
    rc = _run_main(monkeypatch, ["prog"], valid_png)

    assert rc == 0
    win = main_env["window"]
    qtbot.addWidget(win)
    assert win.analyze._path == valid_png
    assert win.analyze.btn_run.isEnabled()
    assert win.analyze.canvas._base is not None


def test_main_ignores_initial_image_missing_file(qtbot, main_env, tmp_path, monkeypatch):
    rc = _run_main(monkeypatch, ["prog"], tmp_path / "missing.png")

    assert rc == 0
    assert main_env["window"].analyze._path is None


def test_main_argv_loads_existing_file(qtbot, main_env, valid_png, monkeypatch):
    rc = _run_main(monkeypatch, ["prog", str(valid_png)], None)

    assert rc == 0
    assert main_env["window"].analyze._path == valid_png


def test_main_argv_skips_flag_then_loads_file(qtbot, main_env, valid_png, monkeypatch):
    rc = _run_main(monkeypatch, ["prog", "--flag", str(valid_png)], None)

    assert rc == 0
    assert main_env["window"].analyze._path == valid_png


def test_main_argv_only_flag_loads_nothing(qtbot, main_env, monkeypatch):
    rc = _run_main(monkeypatch, ["prog", "--help"], None)

    assert rc == 0
    assert main_env["window"].analyze._path is None


def test_main_argv_missing_path_not_loaded(qtbot, main_env, tmp_path, monkeypatch):
    rc = _run_main(monkeypatch, ["prog", str(tmp_path / "missing.png")], None)

    assert rc == 0
    assert main_env["window"].analyze._path is None
