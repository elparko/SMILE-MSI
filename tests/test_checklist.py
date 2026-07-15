"""The ``common.CheckList`` contract — the app's one multi-select primitive.

Fourteen screens read their selection back off this widget, so its promises are load-bearing:
bulk actions act on the *filtered* subset, ticks are keyed (they survive a re-filter and a
re-populate), disabled rows are untouchable, and row order is preserved for the callers whose
matrices and filenames inherit it. Builds no MainWindow — these run in milliseconds.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from smile_msi.gui.common import (CheckList, check_list_bar, check_list_items,  # noqa: E402
                                  check_table_bar, check_table_items, check_tree_bar,
                                  check_tree_leaves, install_space_toggle, move_selected_rows)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _entries(n):
    return [(i, f"Item {i}") for i in range(n)]


def _buttons(cl):
    return {b.text(): b for b in cl.findChildren(QtWidgets.QPushButton)}


def test_bulk_actions_and_order(app):
    cl = CheckList(_entries(12), noun="item")
    b = _buttons(cl)
    b["All"].click()
    assert cl.checked_keys() == set(range(12))
    b["None"].click()
    assert cl.checked_keys() == set()
    cl.set_checked({1, 3})
    b["Invert"].click()
    assert cl.checked_keys() == {0, 2, 4, 5, 6, 7, 8, 9, 10, 11}
    # a set would scramble the callers that inherit the list's order
    cl.set_checked({7, 2, 9})
    assert cl.checked_in_order() == [2, 7, 9]
    assert cl.count() == 12                     # every row, not the filtered subset


def test_bulk_actions_respect_the_filter(app):
    """Type three letters, hit All, and you have ticked exactly that subset — and rows the
    filter hides keep the state they had."""
    cl = CheckList(_entries(12), noun="item")
    cl.set_checked({0})
    cl._filter.setText("Item 1")                # Item 1, Item 10 … Item 11
    shown = cl._list.count()
    assert 0 < shown < 12
    _buttons(cl)["All"].click()
    assert len(cl.checked_keys()) == shown + 1  # the shown rows, plus hidden-but-ticked Item 0
    assert 0 in cl.checked_keys()
    cl._filter.clear()
    assert len(cl.checked_keys()) == shown + 1  # ticks are keyed → they survive un-filtering


def test_extra_action_predicate_survives_clicked_bool(app):
    """A domain shortcut is wired to ``clicked(bool)``. PySide6 reads ``lambda p=pred:`` as
    arity-1 and feeds the checked flag straight into ``p``, so the predicate arrives as
    ``False`` and the button silently raises inside the slot. Click it for real."""
    cl = CheckList(_entries(12), noun="item",
                   extra_actions=[("Even", "even keys", lambda k: k % 2 == 0)])
    _buttons(cl)["Even"].click()                # not cl.set_where(...) — the click is the bug
    assert cl.checked_keys() == {0, 2, 4, 6, 8, 10}
    # and it is a *set*, not a toggle: it unticks the rows that fail the predicate
    cl.set_checked({1, 3})
    _buttons(cl)["Even"].click()
    assert cl.checked_keys() == {0, 2, 4, 6, 8, 10}


def test_extra_action_respects_the_filter(app):
    cl = CheckList(_entries(12), noun="item",
                   extra_actions=[("Even", "even keys", lambda k: k % 2 == 0)])
    cl._filter.setText("Item 1")                # 1, 10, 11
    _buttons(cl)["Even"].click()
    assert cl.checked_keys() == {10}            # only the shown even row


def test_disabled_rows_are_untouchable(app):
    """An already-added region is shown greyed, and no path may tick it — not a bulk action,
    not set_checked, not the Space run-toggle."""
    cl = CheckList([(0, "off", "", False), (1, "on", "", True)], noun="item")
    _buttons(cl)["All"].click()
    assert cl.checked_keys() == {1}
    cl.set_checked({0, 1})
    assert cl.checked_keys() == {1}
    _buttons(cl)["Invert"].click()
    assert cl.checked_keys() == set()


def test_check_new_only_ticks_unseen_keys(app):
    """A newly added sample joins the run by default; a row the user deliberately unticked
    must not come back when the roster is rebuilt."""
    cl = CheckList([(0, "a"), (1, "b")], noun="sample")
    cl.set_all(True)
    cl.set_checked({0})                                  # user unticks b
    cl.set_entries([(0, "a"), (1, "b"), (2, "c")], check_new=True)
    assert cl.checked_keys() == {0, 2}
    cl.set_entries([(0, "a"), (1, "b"), (2, "c")], check_new=True)
    assert cl.checked_keys() == {0, 2}                   # idempotent — c is now 'seen'


def test_set_entries_drops_vanished_keys(app):
    cl = CheckList(_entries(4), noun="item")
    cl.set_all(True)
    cl.set_entries([(0, "a"), (3, "d")])
    assert cl.checked_keys() == {0, 3}


def test_space_toggles_the_selected_run(app):
    """Click the first row, shift-click the last, press Space. Repopulating destroys every
    QListWidgetItem, so the run is re-selected by key — holding the pointers raises
    'Internal C++ object already deleted'."""
    cl = CheckList(_entries(6), noun="item")
    for i in (1, 2, 3):
        cl._list.item(i).setSelected(True)
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier)
    assert cl.eventFilter(cl._list, ev) is True
    assert cl.checked_keys() == {1, 2, 3}
    assert {it.data(QtCore.Qt.UserRole) for it in cl._list.selectedItems()} == {1, 2, 3}
    cl.eventFilter(cl._list, ev)                         # a second press unticks the run
    assert cl.checked_keys() == set()


def test_space_run_follows_the_first_row(app):
    """One press can't half-toggle a mixed run: it follows the first row's new state."""
    cl = CheckList(_entries(6), noun="item")
    cl.set_checked({2})
    for i in (1, 2, 3):
        cl._list.item(i).setSelected(True)
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier)
    cl.eventFilter(cl._list, ev)                         # first row (1) was off → tick all three
    assert cl.checked_keys() == {1, 2, 3}


def test_filter_chrome_hides_on_short_lists(app):
    """On a three-region dialog the filter box and the Space hint are louder than the list."""
    short = CheckList(_entries(3), noun="region")
    assert not short._filter.isVisibleTo(short) and not short._hint.isVisibleTo(short)
    long = CheckList(_entries(30), noun="region")
    assert long._filter.isVisibleTo(long) and long._hint.isVisibleTo(long)
    # growing past the threshold reveals it; shrinking back clears a stale needle
    short.set_entries(_entries(30))
    assert short._filter.isVisibleTo(short)
    short._filter.setText("Item 29")
    short.set_entries(_entries(3))
    assert not short._filter.isVisibleTo(short)
    assert short._filter.text() == ""                    # else it would silently hide rows
    assert short._list.count() == 3


def test_icon_provider_is_cosmetic(app):
    """A swatch that raises must never cost you the row."""
    def boom(_k):
        raise RuntimeError("no colour")

    cl = CheckList(_entries(3), noun="region", icon_for=boom)
    assert cl._list.count() == 3
    cl.set_icon_for(lambda k: QtGui.QIcon())
    assert cl._list.count() == 3


def test_changed_fires_once_per_bulk_action(app):
    cl = CheckList(_entries(5), noun="item")
    seen = []
    cl.changed.connect(lambda: seen.append(1))
    _buttons(cl)["All"].click()
    assert len(seen) == 1                                # not one signal per row
    cl._list.item(0).setCheckState(QtCore.Qt.Unchecked)  # a manual tick still reports
    assert len(seen) == 2
    assert cl.checked_keys() == {1, 2, 3, 4}


# ---- check_tree_bar: the same bargain, for trees that group rows under headers ---- #

def _tree(app, n_heads=2, n_leaves=3):
    """A sample-header tree shaped like scope.py's region picker: bold non-checkable
    headers, checkable leaves beneath them."""
    t = QtWidgets.QTreeWidget()
    t.setHeaderHidden(True)
    for h in range(n_heads):
        head = QtWidgets.QTreeWidgetItem([f"Sample {h}"])
        head.setFlags(QtCore.Qt.ItemIsEnabled)           # a header: not checkable
        t.addTopLevelItem(head)
        for i in range(n_leaves):
            leaf = QtWidgets.QTreeWidgetItem([f"Region {h}{i}"])
            leaf.setFlags(leaf.flags() | QtCore.Qt.ItemIsUserCheckable)
            leaf.setCheckState(0, QtCore.Qt.Unchecked)
            head.addChild(leaf)
        head.setExpanded(True)
    return t


def _ticked(t):
    return [it.text(0) for it in check_tree_leaves(t, visible_only=False)
            if it.checkState(0) == QtCore.Qt.Checked]


def test_tree_bar_bulk_actions_skip_headers(app):
    t = _tree(app)
    bar = check_tree_bar(t, "region")
    b = _buttons(bar)
    assert len(check_tree_leaves(t)) == 6                # 6 leaves, 0 headers
    b["All"].click()
    assert len(_ticked(t)) == 6
    assert t.topLevelItem(0).checkState(0) != QtCore.Qt.Checked   # header untouched
    b["None"].click()
    assert _ticked(t) == []
    b["Invert"].click()
    assert len(_ticked(t)) == 6


def test_tree_bar_respects_hidden_rows(app):
    """Bulk actions act on the *shown* leaves — that is what makes them compose with the
    filter box above them — and leave the hidden ones as they were."""
    t = _tree(app)
    bar = check_tree_bar(t, "region")
    t.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Checked)
    for i in range(3):                                   # hide all of Sample 0's leaves
        t.topLevelItem(0).child(i).setHidden(True)
    _buttons(bar)["All"].click()
    assert sorted(_ticked(t)) == ["Region 00", "Region 10", "Region 11", "Region 12"]
    _buttons(bar)["None"].click()
    assert _ticked(t) == ["Region 00"]                   # the hidden, ticked leaf survives


def test_tree_bar_skips_disabled_and_placeholder_rows(app):
    """scope.py's empty state is a NoItemFlags placeholder; it must never be ticked."""
    t = QtWidgets.QTreeWidget()
    ph = QtWidgets.QTreeWidgetItem(["(no regions yet)"])
    ph.setFlags(QtCore.Qt.NoItemFlags)
    t.addTopLevelItem(ph)
    bar = check_tree_bar(t, "region")
    _buttons(bar)["All"].click()
    assert check_tree_leaves(t) == [] and _ticked(t) == []
    assert bar.findChild(QtWidgets.QLabel).text() == "0 of 0 selected"


def test_tree_bar_count_tracks_manual_ticks_and_refresh(app):
    t = _tree(app, n_heads=1, n_leaves=4)
    bar = check_tree_bar(t, "region")
    count = bar.findChild(QtWidgets.QLabel)
    assert count.text() == "0 of 4 selected"
    t.topLevelItem(0).child(0).setCheckState(0, QtCore.Qt.Checked)   # itemChanged → recount
    assert count.text() == "1 of 4 selected"
    _buttons(bar)["All"].click()
    assert count.text() == "4 of 4 selected"
    # a bar built over an empty tree must be able to catch up after populate
    t.topLevelItem(0).addChild(QtWidgets.QTreeWidgetItem(["late"]))
    t.topLevelItem(0).child(4).setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    bar.refresh_count()
    assert count.text() == "4 of 5 selected"


# ---- check_list_bar / move_selected_rows: the lists that carry per-item state ---- #

def _plain_list(n=5, checked=False):
    """A bare QListWidget of checkable rows — figeditor's shape, which can't be a CheckList
    because its repopulate would destroy the per-item label override + original-index roles."""
    lw = QtWidgets.QListWidget()
    lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    for i in range(n):
        it = QtWidgets.QListWidgetItem(f"row{i}")
        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        it.setData(QtCore.Qt.UserRole, i)          # the per-item state a rebuild would lose
        lw.addItem(it)
    return lw


def _order(lw):
    return [lw.item(i).text() for i in range(lw.count())]


def _list_ticked(lw):
    return [it.text() for it in check_list_items(lw, visible_only=False)
            if it.checkState() == QtCore.Qt.Checked]


def test_list_bar_bulk_actions_and_count(app):
    lw = _plain_list()
    fired = []
    bar = check_list_bar(lw, "ion", on_change=lambda: fired.append(1))
    count = bar.findChild(QtWidgets.QLabel)
    assert count.text() == "0 of 5 selected"
    b = _buttons(bar)
    b["All"].click()
    assert len(_list_ticked(lw)) == 5 and count.text() == "5 of 5 selected"
    b["None"].click()
    assert _list_ticked(lw) == []
    b["Invert"].click()
    assert len(_list_ticked(lw)) == 5
    assert len(fired) == 3                          # one callback per bulk action, not per row


def test_list_bar_respects_hidden_rows(app):
    lw = _plain_list()
    bar = check_list_bar(lw, "ion")
    lw.item(0).setCheckState(QtCore.Qt.Checked)
    for i in (0, 1):
        lw.item(i).setHidden(True)
    _buttons(bar)["All"].click()
    assert _list_ticked(lw) == ["row0", "row2", "row3", "row4"]
    _buttons(bar)["None"].click()
    assert _list_ticked(lw) == ["row0"]              # the hidden, ticked row survives


def test_reorder_buttons_appear_only_when_asked(app):
    assert check_list_bar(_plain_list(), "ion").findChildren(QtWidgets.QToolButton) == []
    bar = check_list_bar(_plain_list(), "ion", reorder=True)
    assert len(bar.findChildren(QtWidgets.QToolButton)) == 2


def test_move_selected_rows_keeps_state_and_selection(app):
    """takeItem/insertItem moves the *item*, so its check state and data roles ride along —
    that is the whole reason this is a move and not a rebuild."""
    lw = _plain_list()
    lw.item(3).setCheckState(QtCore.Qt.Checked)
    lw.item(3).setSelected(True)
    move_selected_rows(lw, -1)
    assert _order(lw) == ["row0", "row1", "row3", "row2", "row4"]
    assert lw.item(2).checkState() == QtCore.Qt.Checked
    assert lw.item(2).data(QtCore.Qt.UserRole) == 3
    assert [i.text() for i in lw.selectedItems()] == ["row3"]


def test_move_selected_rows_multi_row_keeps_whole_run_selected(app):
    """setCurrentItem defaults to ClearAndSelect, which would drop every row of a multi-row
    move but the cursor's."""
    lw = _plain_list()
    lw.item(0).setSelected(True)
    lw.item(1).setSelected(True)
    move_selected_rows(lw, +1)
    assert _order(lw) == ["row2", "row0", "row1", "row3", "row4"]
    assert sorted(i.text() for i in lw.selectedItems()) == ["row0", "row1"]


def test_move_selected_rows_stops_at_the_wall(app):
    lw = _plain_list()
    lw.item(0).setSelected(True)
    move_selected_rows(lw, -1)
    assert _order(lw) == ["row0", "row1", "row2", "row3", "row4"]
    lw.clearSelection()
    lw.item(4).setSelected(True)
    move_selected_rows(lw, +1)
    assert _order(lw) == ["row0", "row1", "row2", "row3", "row4"]
    lw.clearSelection()
    move_selected_rows(lw, -1)                       # nothing selected → no-op, no crash
    assert _order(lw) == ["row0", "row1", "row2", "row3", "row4"]


def test_move_fires_on_change_once(app):
    lw = _plain_list()
    fired = []
    check_list_bar(lw, "ion", on_change=lambda: fired.append(1), reorder=True)
    lw.item(1).setSelected(True)
    move_selected_rows(lw, -1, lambda: fired.append(1))
    assert len(fired) == 1                           # the move itself must not churn itemChanged


def test_install_space_toggle_follows_the_first_row(app):
    lw = _plain_list()
    install_space_toggle(lw)
    lw.item(2).setCheckState(QtCore.Qt.Checked)
    for i in (1, 2, 3):
        lw.item(i).setSelected(True)
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(lw, ev)         # first row (1) was off → tick the run
    assert _list_ticked(lw) == ["row1", "row2", "row3"]
    QtWidgets.QApplication.sendEvent(lw, ev)
    assert _list_ticked(lw) == []


def test_install_space_toggle_ignores_disabled_rows(app):
    lw = _plain_list()
    lw.item(1).setFlags(lw.item(1).flags() & ~QtCore.Qt.ItemIsEnabled)
    install_space_toggle(lw)
    for i in (1, 2):
        lw.item(i).setSelected(True)
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(lw, ev)
    assert _list_ticked(lw) == ["row2"]


# ---- check_table_bar: a tick column in a QTableWidget ---- #

def _table(rows=4, tickable=True):
    """The segmentation cluster table's shape: column 0 is a tick box, the rest is data."""
    t = QtWidgets.QTableWidget(rows, 2)
    for r in range(rows):
        chk = QtWidgets.QTableWidgetItem()
        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if tickable:
            flags |= QtCore.Qt.ItemIsUserCheckable
        chk.setFlags(flags)
        chk.setCheckState(QtCore.Qt.Unchecked)
        chk.setData(QtCore.Qt.UserRole, r)
        t.setItem(r, 0, chk)
        t.setItem(r, 1, QtWidgets.QTableWidgetItem(f"cluster {r}"))
    return t


def _table_ticked(t):
    return [it.data(QtCore.Qt.UserRole) for it in check_table_items(t, 0, visible_only=False)
            if it.checkState() == QtCore.Qt.Checked]


def test_table_bar_bulk_actions(app):
    t = _table()
    fired = []
    bar = check_table_bar(t, 0, "segment", on_change=lambda: fired.append(1))
    count = bar.findChild(QtWidgets.QLabel)
    assert count.text() == "0 of 4 selected"
    b = _buttons(bar)
    b["All"].click()
    assert _table_ticked(t) == [0, 1, 2, 3] and count.text() == "4 of 4 selected"
    b["None"].click()
    assert _table_ticked(t) == []
    b["Invert"].click()
    assert _table_ticked(t) == [0, 1, 2, 3]
    assert len(fired) == 3


def test_table_bar_skips_hidden_rows_and_non_tick_columns(app):
    t = _table()
    bar = check_table_bar(t, 0, "segment")
    t.item(0, 0).setCheckState(QtCore.Qt.Checked)
    t.setRowHidden(0, True)
    _buttons(bar)["All"].click()
    assert _table_ticked(t) == [0, 1, 2, 3]       # row 0 was already ticked, and untouched
    _buttons(bar)["None"].click()
    assert _table_ticked(t) == [0]                # the hidden, ticked row survives


def test_table_bar_ignores_non_checkable_and_missing_cells(app):
    t = _table(tickable=False)
    bar = check_table_bar(t, 0, "segment")
    _buttons(bar)["All"].click()
    assert _table_ticked(t) == []
    assert bar.findChild(QtWidgets.QLabel).text() == "0 of 0 selected"
    t2 = _table(rows=2)
    t2.setItem(1, 0, None)                        # a row built without its tick cell yet
    bar2 = check_table_bar(t2, 0, "segment")
    _buttons(bar2)["All"].click()                 # must not raise on the None cell
    assert _table_ticked(t2) == [0]


def test_table_bar_refresh_count_after_a_blocked_rebuild(app):
    """_fill_seg_table rebuilds rows with the table's signals blocked, so itemChanged never
    fires and the count would go stale."""
    t = _table(rows=3)
    bar = check_table_bar(t, 0, "segment")
    _buttons(bar)["All"].click()
    t.blockSignals(True)
    t.setRowCount(0)
    t.blockSignals(False)
    bar.refresh_count()
    assert bar.findChild(QtWidgets.QLabel).text() == "0 of 0 selected"
