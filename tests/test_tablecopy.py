"""Copying a table copies the whole table, as CSV — the shared grid/copy layer in
``smile_msi.gui.common`` plus the application-wide ⌘C / Ctrl+C over any item view."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from smile_msi.gui import common  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _clear_clipboard(app):
    QtGui.QGuiApplication.clipboard().clear()


def _table(headers=("m/z", "lipid"), rows=(("744.5542", "PE 36:2"), ("760.5851", "PC 34:1"))):
    t = QtWidgets.QTableWidget()
    common.fill_table(t, list(headers), [list(r) for r in rows])
    return t


def _clip():
    return QtGui.QGuiApplication.clipboard().text()


# ----- the grid readers ---------------------------------------------------- #
def test_copy_table_copies_every_row_with_a_header(app):
    t = _table()
    assert common.copy_table(t) == (2, 2)
    assert _clip() == "m/z,lipid\n744.5542,PE 36:2\n760.5851,PC 34:1\n"


def test_one_selected_cell_still_copies_the_whole_table(app):
    """The reported bug: clicking a cell and pressing ⌘C used to yield that one number."""
    t = _table()
    t.setCurrentCell(1, 0)
    assert len(t.selectedIndexes()) == 1
    common.copy_table(t)
    assert _clip().splitlines() == ["m/z,lipid", "744.5542,PE 36:2", "760.5851,PC 34:1"]


def test_two_selected_rows_copy_just_those_rows(app):
    t = _table(rows=[(str(i), f"L{i}") for i in range(5)])
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    for r in (1, 3):
        t.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(r, 0, r, 1), True)
    assert common.copy_table(t) == (2, 2)
    assert _clip().splitlines() == ["m/z,lipid", "1,L1", "3,L3"]


def test_hidden_rows_and_columns_stay_out_of_the_copy(app):
    t = _table(headers=("m/z", "lipid", "note"),
               rows=[("1", "a", "x"), ("2", "b", "y"), ("3", "c", "z")])
    t.setRowHidden(1, True)
    t.setColumnHidden(2, True)
    assert common.copy_table(t) == (2, 2)
    assert _clip().splitlines() == ["m/z,lipid", "1,a", "3,c"]


def test_a_checkbox_only_cell_copies_its_tick_state(app):
    """The segmentation '✓' column and the standards 'Use' column hold no text at all."""
    t = QtWidgets.QTableWidget(2, 2)
    t.setHorizontalHeaderLabels(["use", "name"])
    for r, state in enumerate((QtCore.Qt.Checked, QtCore.Qt.Unchecked)):
        chk = QtWidgets.QTableWidgetItem()
        chk.setCheckState(state)
        t.setItem(r, 0, chk)
        t.setItem(r, 1, QtWidgets.QTableWidgetItem(f"row{r}"))
    common.copy_table(t)
    assert _clip().splitlines() == ["use,name", "yes,row0", "no,row1"]


def test_a_value_with_a_comma_is_quoted(app):
    t = _table(headers=("lipid",), rows=[("PE 16:0_18:1, [M-H]-",)])
    common.copy_table(t)
    assert _clip() == 'lipid\n"PE 16:0_18:1, [M-H]-"\n'


def test_copy_for_excel_is_tab_separated(app):
    assert common.copy_table(_table(), sep="\t") == (2, 2)
    assert _clip() == "m/z\tlipid\n744.5542\tPE 36:2\n760.5851\tPC 34:1\n"


def test_the_clipboard_is_written_with_setText_not_a_python_QMimeData(app):
    """A QMimeData built in Python outlives the interpreter and segfaults PySide's static
    teardown, so every process that copied a table crashed on quit. Keep it setText."""
    import inspect
    assert "setMimeData(" not in inspect.getsource(common.copy_grid)


def test_a_tree_copies_every_visible_row(app):
    tree = QtWidgets.QTreeWidget()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["sample", "group"])
    parent = QtWidgets.QTreeWidgetItem(tree, ["Treated", ""])
    QtWidgets.QTreeWidgetItem(parent, ["S01", "Trt"])
    QtWidgets.QTreeWidgetItem(parent, ["S02", "Trt"])
    tree.expandAll()
    assert common.copy_table(tree) == (3, 2)
    assert _clip().splitlines() == ["sample,group", "Treated,", "S01,Trt", "S02,Trt"]


def test_a_list_copies_its_visible_rows(app):
    lw = QtWidgets.QListWidget()
    lw.addItems(["endoneurium", "perineurium", "epineurium"])
    lw.item(1).setHidden(True)
    assert common.copy_table(lw) == (2, 1)
    assert _clip().splitlines() == ["item", "endoneurium", "epineurium"]


def test_an_empty_table_copies_nothing(app):
    t = QtWidgets.QTableWidget()
    common.fill_table(t, ["m/z"], [])
    assert common.copy_table(t) is None
    assert _clip() == ""


# ----- the application-wide shortcut --------------------------------------- #
def _ctrl_c(widget):
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier)
    QtWidgets.QApplication.sendEvent(widget, ev)
    return ev


def test_ctrl_c_over_a_table_copies_the_whole_table(app):
    common.install_copy_shortcut(app)
    t = _table()
    ev = _ctrl_c(t)
    assert ev.isAccepted()
    assert _clip().splitlines()[0] == "m/z,lipid"


def test_ctrl_c_on_the_viewport_reaches_the_table(app):
    common.install_copy_shortcut(app)
    t = _table()
    _ctrl_c(t.viewport())
    assert _clip().splitlines()[0] == "m/z,lipid"


def test_ctrl_c_in_a_text_field_is_left_to_the_text_field(app):
    common.install_copy_shortcut(app)
    dlg = QtWidgets.QDialog()
    edit = QtWidgets.QLineEdit(dlg)
    QtWidgets.QTableWidget(2, 2, dlg)          # a table in the same window must not win
    _ctrl_c(edit)
    assert _clip() == ""


def test_install_copy_shortcut_is_idempotent(app):
    common.install_copy_shortcut(app)
    assert common.install_copy_shortcut(app) is None


# ----- the right-click affordance ------------------------------------------ #
def test_filling_a_table_gives_it_a_copy_menu(app):
    t = _table()
    assert t.contextMenuPolicy() == QtCore.Qt.CustomContextMenu
    assert t.property("_smile_table_menu") is True


def test_install_table_export_leaves_a_view_with_its_own_menu_alone(app):
    t = QtWidgets.QTableWidget(1, 1)
    t.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)   # a bespoke menu, as segment.py has
    assert common.install_table_export(t) is None
    assert t.property("_smile_table_menu") is None


def test_add_copy_actions_disables_copy_selected_with_no_selection(app):
    t = _table()
    menu = QtWidgets.QMenu()
    common.add_copy_actions(menu, t)
    labels = [a.text() for a in menu.actions()]
    assert labels == ["Copy table (CSV)", "Copy selected rows",
                      "Copy for Excel (tab-separated)"]
    assert menu.actions()[0].isEnabled() and not menu.actions()[1].isEnabled()


# ----- export from a dialog ------------------------------------------------- #
def test_export_reports_on_the_main_window_even_when_launched_from_a_dialog(app, tmp_path,
                                                                           monkeypatch):
    """resultviews / scriptconsole pass a plain widget as the export parent; export_rows used
    to write the file and then raise AttributeError looking for statusBar()."""
    win = QtWidgets.QMainWindow()
    dlg = QtWidgets.QDialog(win)
    t = _table()
    t.setParent(dlg)
    out = tmp_path / "grid.csv"
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_save_file_name",
                        lambda *a, **k: (str(out), "CSV (*.csv)"))
    assert common.export_table(dlg, t, stem="grid") == str(out)
    assert out.read_text(encoding="utf-8-sig").splitlines()[0] == "m/z,lipid"
    assert "grid.csv" in win.statusBar().currentMessage()
