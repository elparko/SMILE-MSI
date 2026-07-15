"""GUI smoke test for the Acquisition metadata & reporting dialog (plan 02, headless)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import provenance, standards  # noqa: E402
from smile_msi.gui import reportingdialog as rd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._acquisition_meta = {}
        self.prov = provenance.Provenance(title="t", started="2026-06-24T00:00:00+00:00")
        self.ds = None
        self._dirty = False
        self.exported = False

    def _mark_dirty(self):
        self._dirty = True

    def export_imzml(self):
        self.exported = True


def test_reporting_dialog_collect_save_validate(app):
    main = _FakeMain()
    dlg = rd.ReportingDialog(main)
    dlg.load_from_main()                       # seeds from the active profile, no error
    dlg._fields["organism"].setText("Mus musculus")
    dlg._fields["tissue"].setText("sciatic nerve")
    dlg._fields["matrix"].setText("DHB")
    dlg._fields["section_thickness_um"].setText("10")
    dlg._fields["laser_spot_um"].setText("20")

    meta = dlg._collect()
    assert isinstance(meta, standards.AcquisitionMeta)
    assert meta.organism == "Mus musculus"
    assert meta.section_thickness_um == 10.0

    dlg._save()
    assert main._dirty is True
    assert main._acquisition_meta["organism"] == "Mus musculus"

    dlg._validate()                            # populates the MIAMSIE checklist view, no error
    assert dlg.status.text().startswith("MIAMSIE")
    assert dlg.report.toPlainText()            # non-empty checklist

    dlg._export()                              # no dataset → graceful, doesn't export
    assert main.exported is False


def test_reporting_dialog_bad_number_is_none(app):
    main = _FakeMain()
    dlg = rd.ReportingDialog(main)
    dlg._fields["section_thickness_um"].setText("not-a-number")
    assert dlg._collect().section_thickness_um is None   # never raises
