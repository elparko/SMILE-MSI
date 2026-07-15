"""GUI smoke test for the Cohort-tab batch-assignment menu (plan 01 GUI, headless)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import cohort  # noqa: E402
from smile_msi.gui.cohortview import CohortMixin  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeTab(CohortMixin, QtWidgets.QMainWindow):
    """Real CohortMixin methods on a bare window, with cohort persistence stubbed."""
    def __init__(self, samples):
        super().__init__()
        self.cohort = cohort.Cohort(name="t", samples=samples)
        self.saved = False

    def _save_cohort(self):
        self.saved = True


def test_batch_by_folder_assigns_and_saves(app):
    samples = [cohort.SampleRef(name="a", session_path="/a.json", source="/run1/a.imzML"),
               cohort.SampleRef(name="b", session_path="/b.json", source="/run2/b.imzML")]
    tab = _FakeTab(samples)
    tab._cohort_batch_by_folder()
    assert tab.saved
    assert set(tab.cohort.batches()) == {"run1", "run2"}        # batch = containing folder
    assert tab.cohort.groups() == []                            # group axis untouched


def test_batch_by_folder_empty_cohort_is_graceful(app):
    tab = _FakeTab([])
    tab._cohort_batch_by_folder()                              # no samples → no crash
    assert tab.cohort.batches() == []
