"""GUI smoke test for the landmark registration dialog (plan 05, headless)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.gui import registerdialog as rd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    return tmp_path


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.applied = None
        self.steps = []

    def _apply_optical_coeffs(self, coeffs):
        self.applied = dict(coeffs)

    def record_step(self, kind, *, label="", params=None, regions=None):
        self.steps.append((kind, params))


def _fill(dlg, pts):
    while dlg.table.rowCount() < len(pts):
        dlg._add_row()
    for r, (a, b, c, d) in enumerate(pts):
        dlg.table.item(r, 0).setText(str(a))
        dlg.table.item(r, 1).setText(str(b))
        dlg.table.item(r, 2).setText(str(c))
        dlg.table.item(r, 3).setText(str(d))


def test_landmark_dialog_identity(app, home):
    main = _FakeMain()
    dlg = rd.LandmarkDialog(main)
    # photo == ion at four corners → identity transform, ~0 RMSE
    _fill(dlg, [(0, 0, 0, 0), (10, 0, 10, 0), (0, 10, 0, 10), (10, 10, 10, 10)])
    dlg._estimate()
    assert dlg.result is not None
    assert dlg.result.rmse < 1e-6
    assert main.applied is not None                       # coeffs pushed to the optical controls
    assert main.steps and main.steps[0][0] == "registration"
    assert "RMSE" in dlg.status.text()


def test_landmark_dialog_recovers_translation(app, home):
    main = _FakeMain()
    dlg = rd.LandmarkDialog(main)
    # ion = photo + (5, 3) → a pure translation
    _fill(dlg, [(0, 0, 5, 3), (10, 0, 15, 3), (0, 10, 5, 13), (8, 8, 13, 11)])
    dlg._estimate()
    assert dlg.result is not None and dlg.result.rmse < 1e-6
    assert main.applied is not None


def test_landmark_dialog_needs_enough_pairs(app, home):
    main = _FakeMain()
    dlg = rd.LandmarkDialog(main)
    _fill(dlg, [(0, 0, 0, 0)])                             # only one pair
    dlg._estimate()
    assert dlg.result is None
    assert "at least" in dlg.status.text().lower()
