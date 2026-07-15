"""AnalysisDialog provenance logging (plan 24).

Guards that running an analysis through the generic dialog records all three provenance
channels the dedicated tabs used to — the re-openable History run record, the session audit
trail (``win.prov`` via ``record_step``), and a report item carrying its source context —
so a future refactor can't silently drop provenance again.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.gui import main as M
from smile_msi.gui.analysisdialog import AnalysisDialog
from smile_msi import registry


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sync_run(fn, *a, on_done=None, want_progress=False, run_id=None,
              on_cancelled=None, on_error=None, **k):
    res = fn(*a)
    if on_done:
        on_done(res)


@pytest.fixture
def win(app, tmp_path):
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "prov.imzML"                       # non-synthetic → run store persists
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w._session_path = str(tmp_path / "prov.smile.json")
    w.set_active_mz(float(w.peaks[0]["mz"]))
    w._run = _sync_run
    return w


def test_analysis_run_logs_history_audit_and_report(win):
    prov_before = len(win.prov.steps)
    store = win.run_store
    assert store is not None
    runs_before = len(store.list_runs())

    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["dgmm"]))
    dlg._run()

    # works
    assert dlg.result is not None and dlg._result_widget is not None

    # 1) History run record — done, with dataset identity + params
    runs = store.list_runs()
    assert len(runs) == runs_before + 1
    r = runs[0]
    assert r.step_id == "dgmm" and r.status == "done"
    assert r.dataset == "prov.imzML" and r.dataset_fingerprint and isinstance(r.params, dict)

    # 2) session audit trail (methods provenance) grew by exactly one step
    assert len(win.prov.steps) == prov_before + 1
    step = win.prov.steps[-1]
    assert step["step"] == "segmentation"          # dgmm maps to the segmentation methods kind
    assert "k" in step["params"]

    # 3) report logging carries the source context + a run_id backlink
    win.report_items = []
    dlg2 = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["dgmm"]))
    dlg2._run()
    dlg2._add_to_report()
    # dgmm has no table → nothing added; use a table analysis for the report assertion
    win.set_active_mz(win.active_mz)
    # tag two groups for an A/B table analysis
    rows, cols = win.ds._pixel_rows_cols()
    left = cols < win.ds.width / 2
    win._new_region(name="A", mask=left, select=False)
    win._new_region(name="B", mask=~left, select=False)
    for rg in win.regions:
        rg["group"] = {"A": "A", "B": "B"}.get(rg["name"], rg.get("group"))
    dlg3 = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["roi_comparison"]))
    dlg3._run()
    dlg3._add_to_report()
    assert win.report_items, "a table analysis should log a report item"
    src = win.report_items[-1].get("source") or {}
    assert src.get("dataset") and "run_id" in src


def test_session_embeds_audit_trail_and_run_index(win):
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["dgmm"]))
    dlg._run()
    state = win._session_state()
    assert state["provenance"]["steps"], "session must embed the methods audit trail"
    assert state["analysis_runs"], "session must embed the run-history index"
