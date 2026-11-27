"""Selection, focus and current index are independent in Qt accessibility."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from imageguard.services.analysis_service import AnalysisRequest
from imageguard.ui.app import build_services
from imageguard.ui.screens.history import HistoryScreen


@pytest.fixture()
def history_screen(qtbot, tmp_path, valid_png):
    analysis, history = build_services(db_path=tmp_path / "history.db", artifacts_root=tmp_path / "artifacts")
    for _ in range(2):
        analysis.analyze(AnalysisRequest(valid_png, methods=["ela"], include_exif=False))
    screen = HistoryScreen(history)
    qtbot.addWidget(screen)
    screen.resize(1000, 600)
    screen.show()
    screen.refresh()
    assert screen.table.rowCount() == 2
    assert screen.table.currentRow() == -1
    return screen


def _assert_actions(screen, enabled):
    for button in (screen.btn_reopen, screen.btn_export, screen.btn_delete):
        assert button.isEnabled() is enabled


@pytest.mark.parametrize("whole_row", [False, True])
def test_accessible_selection_then_pointer_click_enables_actions(
    qtbot, history_screen, whole_row,
):
    screen = history_screen
    index = screen.table.model().index(0, 1)
    flags = QItemSelectionModel.SelectionFlag.ClearAndSelect
    if whole_row:
        flags |= QItemSelectionModel.SelectionFlag.Rows
    screen.table.selectionModel().select(index, flags)
    assert screen.table.currentRow() == -1
    expected_id = screen.table.item(0, 4).text()
    assert screen._selected_id() == expected_id
    _assert_actions(screen, True)

    qtbot.mouseClick(screen.table.viewport(), Qt.MouseButton.LeftButton,
                     pos=screen.table.visualRect(index).center())
    assert screen.table.currentRow() == 0
    assert screen._selected_id() == expected_id
    _assert_actions(screen, True)


@pytest.mark.parametrize("input_method", ["pointer", "keyboard"])
def test_pointer_and_keyboard_selection_then_clear(history_screen, qtbot, input_method):
    screen = history_screen
    table = screen.table
    if input_method == "pointer":
        index = table.model().index(0, 1)
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualRect(index).center())
    else:
        table.setFocus()
        qtbot.keyClick(table, Qt.Key.Key_Down)
    assert table.currentRow() >= 0
    assert screen._selected_id() == table.item(table.currentRow(), 4).text()
    _assert_actions(screen, True)

    table.clearSelection()
    assert table.currentRow() >= 0  # focus remains; it must not authorize an action
    assert screen._selected_id() is None
    _assert_actions(screen, False)


def test_actions_export_selected_record_even_when_current_index_differs(
    history_screen, tmp_path, monkeypatch,
):
    screen = history_screen
    selection = screen.table.selectionModel()
    index = screen.table.model().index(0, 1)
    selection.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
    selection.setCurrentIndex(screen.table.model().index(1, 1), QItemSelectionModel.SelectionFlag.NoUpdate)
    assert screen.table.currentRow() == 1
    selected_id = screen.table.item(0, 4).text()
    other_id = screen.table.item(1, 4).text()
    destination = tmp_path / "selected_report.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(destination), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: QMessageBox.StandardButton.Ok)
    screen.btn_export.click()
    exported = destination.read_text(encoding="utf-8")
    assert selected_id in exported
    assert other_id not in exported


def test_multiple_programmatic_rows_do_not_choose_an_arbitrary_report(history_screen):
    screen = history_screen
    selection = screen.table.selectionModel()
    for row in range(2):
        selection.select(screen.table.model().index(row, 1),
                         QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert {index.row() for index in screen.table.selectedIndexes()} == {0, 1}
    assert screen._selected_id() is None
    _assert_actions(screen, False)


def test_nonempty_history_has_no_stale_empty_accessibility_text(history_screen):
    assert history_screen.lbl_empty.isHidden()
    assert not history_screen.lbl_empty.text()
