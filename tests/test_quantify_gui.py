"""GUI smoke test for the Quantification (calibration) dialog (plan 03, headless)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from smile_msi import quantify  # noqa: E402
from smile_msi.gui import quantify as qd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ds = None
        self.peaks = [{"mz": 700.0}, {"mz": 750.0}]
        self.regions = [{"name": "std-low", "mask": np.array([True, False, True])},
                        {"name": "std-high", "mask": np.array([False, True, True])}]
        self._calibration_models = []
        self._dirty = False

    def _region_pixel_mask(self, r, _seen=None):
        return r.get("mask")

    def _mark_dirty(self):
        self._dirty = True


def test_quantify_dialog_populates_and_collects_levels(app):
    main = _FakeMain()
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()
    assert dlg.analyte.count() == 2                 # two peaks
    assert dlg.is_combo.count() == 3                # (none) + two peaks

    dlg.table.item(0, 0).setCheckState(QtCore.Qt.Checked)
    dlg.table.item(0, 2).setText("1.0")
    dlg.table.item(1, 0).setCheckState(QtCore.Qt.Checked)
    dlg.table.item(1, 2).setText("5.0")

    levels = dlg._levels()
    assert len(levels) == 2
    assert isinstance(levels[0], quantify.CalLevel)
    assert levels[0].region_name == "std-low"
    assert levels[0].concentration == 1.0
    assert levels[1].concentration == 5.0


def test_quantify_dialog_fit_without_dataset_is_graceful(app):
    main = _FakeMain()
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()
    dlg._fit()                                      # ds is None → status message, no crash
    assert dlg._model is None


def test_quantify_dialog_needs_two_levels(app):
    main = _FakeMain()
    main.ds = object()                              # non-None so we pass the ds gate
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()
    dlg.table.item(0, 0).setCheckState(QtCore.Qt.Checked)
    dlg.table.item(0, 2).setText("1.0")
    dlg._fit()                                      # only one level
    assert "at least two" in dlg.report.toPlainText().lower()


def test_analyte_combo_is_filterable_and_labelled(app):
    """The working set can run to hundreds of ions, so both m/z dropdowns are type-to-filter.
    Each entry is labelled 'm/z · lipid' (so the completer matches a lipid name too) and
    carries the full-precision m/z in its data role."""
    main = _FakeMain()
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()
    for combo in (dlg.analyte, dlg.is_combo):
        assert combo.isEditable()
        assert combo.completer().completionMode() == QtWidgets.QCompleter.PopupCompletion
        assert combo.completer().filterMode() == QtCore.Qt.MatchContains
    assert dlg.analyte.itemData(0) == 700.0
    assert dlg.is_combo.itemText(0) == "(none)" and dlg.is_combo.itemData(0) is None


def test_combo_mz_resolves_from_the_text_not_a_stale_index(app):
    """Typing over an editable combo leaves ``currentIndex()`` parked on the previously chosen
    row, so ``currentData()`` still returns the *old* m/z. Reading that would quantify the old
    ion while the box displayed the new one — wrong, and silently so."""
    main = _FakeMain()
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()

    dlg.analyte.setCurrentIndex(0)
    assert dlg._combo_mz(dlg.analyte) == 700.0

    dlg.analyte.setEditText("not a number")
    assert dlg.analyte.currentData() == 700.0            # Qt keeps the stale data role
    assert dlg._combo_mz(dlg.analyte) is None            # …but we resolve from the text

    dlg.analyte.setCurrentIndex(0)                       # parked on 700, type 750's label
    dlg.analyte.setEditText(dlg.analyte.itemText(1))
    assert dlg._combo_mz(dlg.analyte) == 750.0

    dlg.is_combo.setCurrentIndex(0)
    assert dlg._combo_mz(dlg.is_combo) is None           # '(none)' → no internal standard, no crash


def test_fit_reports_when_no_analyte_is_chosen(app):
    """An empty analyte box used to reach ``float(currentText())`` and raise. Get past the
    dataset and two-levels guards first, so the analyte check is the one under test."""
    main = _FakeMain()
    main.ds = object()                                    # past the "Load a dataset first" guard
    dlg = qd.QuantifyDialog(main)
    dlg.load_from_main()
    for row, conc in ((0, "1.0"), (1, "5.0")):            # past the "two levels" guard
        dlg.table.item(row, 0).setCheckState(QtCore.Qt.Checked)
        dlg.table.item(row, 2).setText(conc)
    assert len(dlg._levels()) == 2

    dlg.analyte.setEditText("")
    dlg._fit()                                            # must not raise on float("")
    assert "analyte" in dlg.report.toPlainText().lower()
