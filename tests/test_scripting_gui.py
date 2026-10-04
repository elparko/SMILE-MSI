"""Script Console GUI tests — drive the console headless (offscreen Qt).

Covers building the API from live regions/groups, the synchronous run + result rendering
(log / table / image tabs), the error path, apply-to-app (features + segmentation), the AI
guide, and workflow preset save/load.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import scripting  # noqa: E402
from smile_msi.gui import main as M  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sync_run(fn, *a, on_done=None, want_progress=False, busy="", modal=False, title=None, **k):
    res = fn(*a, **k)
    if on_done:
        on_done(res)
    return None


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))    # isolate the workflow store
    w = M.MainWindow()
    w._run = _sync_run
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    rows, cols = ds._pixel_rows_cols()
    left = cols < ds.width / 2
    w.regions = [
        {"name": "L", "color": "#DD8452", "segments": set(), "mask": left,
         "parent": None, "visible": True, "group": "Group A"},
        {"name": "R", "color": "#55A868", "segments": set(), "mask": ~left,
         "parent": None, "visible": True, "group": "Group B"},
    ]
    return w


def _console(win):
    win._open_script_console()
    return win._script_console


def test_build_api_from_regions_and_groups(win):
    con = _console(win)
    api = con._build_api()
    assert set(api.region_names()) == {"L", "R"}
    assert set(api.group_names()) == {"Group A", "Group B"}
    assert api.ppm == win.ppm


def test_run_renders_log_table_image(win):
    con = _console(win)
    con.editor.setPlainText(
        "peaks = find_peaks(snr=4, max_peaks=40)\n"
        "log('have', len(peaks))\n"
        "res = compare('Group A', 'Group B')\n"
        "table(res.head(6), 'A vs B')\n"
        "image(float(peaks[0]['mz']), 'first ion')\n"
    )
    con._run()
    titles = [con.out_tabs.tabText(i) for i in range(con.out_tabs.count())]
    assert titles == ["Log", "A vs B", "first ion"]
    assert "have 40" in con.log_view.toPlainText()
    assert con._apply_btn.isEnabled()
    assert con._last_result.ok


def test_run_error_shows_in_log(win):
    con = _console(win)
    con.editor.setPlainText("this is not python !!")
    con._run()
    assert not con._last_result.ok
    assert "SyntaxError" in con.log_view.toPlainText()
    assert con.out_tabs.currentIndex() == 0          # jumped to the Log tab


def test_apply_features_and_segmentation(win):
    con = _console(win)
    con.editor.setPlainText("find_peaks(snr=4, max_peaks=30)\nsegment(n_clusters=3)\n")
    con._run()
    con._apply_features()
    assert len(win.peaks) == 30
    con._apply_segmentation()
    assert win.seg is not None and win.seg.n_clusters == 3


def test_ai_guide_builds(win):
    con = _console(win)
    doc = scripting.capabilities_doc(con._build_api())
    assert "find_peaks(" in doc and "runtime context" in doc


def test_workflow_save_and_reload(win):
    con = _console(win)
    con.editor.setPlainText("find_peaks(snr=5)\n")
    con._wf_name, con._wf_desc = "Quick peaks", "demo"
    path = con._current_workflow().save()
    assert os.path.exists(path)
    con.editor.setPlainText("# cleared")
    con._apply_loaded(scripting.Workflow.load(path))
    assert con.editor.toPlainText() == "find_peaks(snr=5)\n"
    assert "Quick peaks" in [w["name"] for w in scripting.list_workflows()]


def test_insert_example(win):
    con = _console(win)
    first = next(iter(scripting.EXAMPLES.values()))
    con._insert_example(first)
    assert con.editor.toPlainText() == first


# --------------------------------------------------------------------------- #
# robustness — closing mid-run, render/guide error recovery, zero-output status
# --------------------------------------------------------------------------- #
def test_close_while_running_cancels_and_restores_buttons(win):
    con = _console(win)
    con._running = True
    con._set_running(True)
    win._cancel = False
    assert not con.b_run.isEnabled() and con.b_cancel.isEnabled() and con.editor.isReadOnly()

    con.close()                                       # window-close mid-run

    assert win._cancel is True                        # the worker was signalled to stop
    assert con._running is False
    assert con.b_run.isEnabled() and not con.b_cancel.isEnabled()   # Run/Cancel restored
    assert not con.editor.isReadOnly()


def test_escape_key_closes_and_cancels(win):
    from PySide6 import QtCore, QtGui
    con = _console(win)
    con._running = True
    con._set_running(True)
    win._cancel = False
    con.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key_Escape,
                                      QtCore.Qt.KeyboardModifier.NoModifier))
    assert win._cancel is True and con._running is False
    assert con.b_run.isEnabled()


def test_render_survives_bad_table_and_image(win):
    from types import SimpleNamespace
    con = _console(win)
    # a result whose table/image objects blow up the per-tab widget builders
    bad = SimpleNamespace(logs=["ran"], stdout="", values={}, error="",
                          tables=[("Bad table", object())],          # no .columns → raises
                          images=[("Bad image", "not-an-array", {})])  # asarray(float) → raises
    con._render(bad)                                  # must not propagate
    titles = [con.out_tabs.tabText(i) for i in range(con.out_tabs.count())]
    assert "Bad table" in titles and "Bad image" in titles   # fallback notes still tabbed


def test_guide_survives_capabilities_error(win, monkeypatch):
    con = _console(win)

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(scripting, "capabilities_doc", boom)

    shown = {}

    def fake_exec(self):                              # don't block on the modal guide dialog
        views = self.findChildren(QtWidgets.QPlainTextEdit)
        shown["text"] = views[0].toPlainText() if views else ""
        return 0
    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)

    con._show_guide()                                 # must not raise
    assert "Could not build the scripting guide" in shown.get("text", "")


def test_zero_output_workflow_status(win):
    con = _console(win)
    con.editor.setPlainText("x = 1 + 1\n")            # valid, but surfaces nothing
    con._run()
    assert con._last_result.ok
    assert "no output" in con._status.text().lower()


def test_apply_regions_adds_named_regions_and_undoes(win):
    con = _console(win)
    con.editor.setPlainText(
        "endo = threshold_mask(888.6236, 60)\n"
        "peri = ring(endo, width_px=2, mode='outer')\n"
        "add_region('endo', endo); add_region('peri', peri, color='#ff8800')\n"
        "add_region('epi', invert(endo | peri))\n"
    )
    con._run()
    assert con._last_result.ok and con._apply_btn.isEnabled()
    before = [rg["name"] for rg in win.regions]
    con._apply_regions()
    names = [rg["name"] for rg in win.regions]
    assert names == before + ["endo", "peri", "epi"]
    by = {rg["name"]: rg for rg in win.regions}
    assert by["peri"]["color"] == "#ff8800"
    assert int(by["endo"]["mask"].sum()) + int(by["peri"]["mask"].sum()) \
        + int(by["epi"]["mask"].sum()) == win.ds.n_pixels
    assert "regions: L, R, endo, peri, epi" in con._ctx_note.text()
    win._undo()
    assert [rg["name"] for rg in win.regions] == before
