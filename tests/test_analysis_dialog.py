"""The Analyze surface (plan 24) — one fast, comprehensive pass.

Everything downstream analysis-wise now runs through the generic AnalysisDialog launched from
the Analyze gallery. This module builds ONE MainWindow (module-scoped) and drives the whole
surface against it: every converted analysis runs + renders + logs (History run record +
methods audit trail), the gallery scope routing, the dockable-tab open/close, re-open from
History without recompute, and the export/apply actions. Keeping a single window (the
expensive part) makes the full file run in a few seconds instead of rebuilding per test.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.gui import main as M, filedialogs
from smile_msi.gui.analysisdialog import AnalysisDialog
from smile_msi.gui.featureexport import FeatureExportDialog
from smile_msi.gui.gallery import CONVERTED, _COHORT_SCREENS
from smile_msi import registry, spatial

# converted steps that need a prior stats table / per-step pair config the gallery state
# doesn't carry — exercised elsewhere, skipped in the blanket run-everything sweep.
_SKIP = {"filter_auc", "marker_panel"}
_CONVERTED = sorted(CONVERTED - _SKIP)


def _sync_run(fn, *a, on_done=None, want_progress=False, run_id=None,
              on_cancelled=None, on_error=None, **k):
    res = fn(*a)
    if on_done:
        on_done(res)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def win(app, tmp_path_factory):
    """One primed window with regions, groups, a segmentation and a run store — shared across
    the module so we pay the build cost once."""
    tmp = tmp_path_factory.mktemp("analysis")
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "analysis.imzML"                   # non-synthetic → run store persists
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w._session_path = str(tmp / "analysis.smile.json")
    w.set_active_mz(float(w.peaks[0]["mz"]))
    rows, cols = ds._pixel_rows_cols()
    thirds = np.digitize(cols, [ds.width / 3, 2 * ds.width / 3])
    for i, nm in enumerate(["L", "M", "R"]):
        w._new_region(name=nm, mask=(thirds == i), select=False)
    for rg in w.regions:
        rg["group"] = {"L": "A", "M": "C", "R": "B"}[rg["name"]]
    w.seg = spatial.segment(ds, [p["mz"] for p in w.peaks], n_clusters=3)
    w._run = _sync_run
    return w


# --------------------------------------------------------------------------- #
# every converted analysis: runs, renders, and logs all provenance channels
# --------------------------------------------------------------------------- #
# cheaper params for the heavy engines — the sweep checks "runs + logs", not full-fidelity
# compute (which the engine tests cover). Keeps the whole file well under a minute.
_CHEAP = {
    "shap_biomarkers": {"n_estimators": 40, "max_pixels": 2000},
    "classify_cv": {"n_folds": 3},
}


@pytest.mark.parametrize("step_id", _CONVERTED)
def test_converted_analysis_runs_renders_and_logs(win, step_id):
    sd = registry.REGISTRY[step_id]
    store = win.run_store
    prov_before = len(win.prov.steps)
    runs_before = len(store.list_runs())

    dlg = win.open_analysis(AnalysisDialog(win, sd))
    assert win.tabs.indexOf(dlg) >= 0                       # connects: docked as a tab
    if step_id in _CHEAP:
        dlg.form.set_values({**dlg.form.values(), **_CHEAP[step_id]})
    dlg._run()

    # works
    assert dlg.result is not None and dlg._result_widget is not None

    # logs (1): a done History run record with dataset identity + params
    runs = store.list_runs()
    assert len(runs) == runs_before + 1
    done = [r for r in runs if r.step_id == step_id and r.status == "done"]
    assert done and done[0].dataset_fingerprint and isinstance(done[0].params, dict)

    # logs (2): the session methods audit trail grew by one step
    assert len(win.prov.steps) == prov_before + 1

    win._close_analysis(dlg)                               # closing keeps the run in History
    assert win.tabs.indexOf(dlg) < 0


# --------------------------------------------------------------------------- #
# dockable tabs, gallery routing, re-open, export/apply
# --------------------------------------------------------------------------- #
def test_open_two_and_switch_and_close(win):
    n0 = win.tabs.count()
    d1 = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["dgmm"]))
    d2 = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["pca"]))
    assert win.tabs.count() == n0 + 2 and win.tab_strip.count() == win.tabs.count()
    win.tabs.setCurrentIndex(win.tabs.indexOf(d1))
    assert win.tabs.currentWidget() is d1
    win._close_analysis(d1)
    win._close_analysis(d2)
    assert win.tabs.count() == n0


def test_cohort_scope_routes_to_the_cohort_screen(win):
    win._set_gallery_scope("cohort")
    seen = []
    orig = win.reveal_view
    win.reveal_view = lambda label: seen.append(label)
    try:
        win._launch_card(registry.REGISTRY["cohort_embedding"])
        win._launch_card(registry.REGISTRY["roi_comparison"])   # slide step w/ cohort target
    finally:
        win.reveal_view = orig
    win._set_gallery_scope("slide")
    assert seen == ["Cohort UMAP", "Cohort"]
    assert set(_COHORT_SCREENS)                                  # sanity: labels defined


def test_reopen_from_history_does_not_recompute(win):
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["dgmm"]))
    dlg._run()
    run = win.run_store.list_runs()[0]
    win._close_analysis(dlg)

    sd = registry.REGISTRY["dgmm"]
    calls = {"n": 0}
    orig = sd.run
    object.__setattr__(sd, "run", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k))
    try:
        reopened = AnalysisDialog.from_run(win, run, win.run_store)
    finally:
        object.__setattr__(sd, "run", orig)
    assert calls["n"] == 0 and reopened.b_run.text() == "Re-run"
    win._close_analysis(reopened)


def test_export_actions_present_on_a_result(win):
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["roi_comparison"]))
    dlg._run()
    for b in (dlg.b_copy, dlg.b_image, dlg.b_report, dlg.b_hub):
        assert b.isEnabled()
    dlg._copy_table()
    assert "mz" in QtWidgets.QApplication.clipboard().text()
    win._close_analysis(dlg)


def test_classify_map_saves_prediction_as_regions(win):
    n0 = len(win.regions)
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["classify_map"]))
    dlg._run()
    regs = registry.REGISTRY["classify_map"].regions_out(dlg.result, win.ds)
    assert regs, "classify_map should offer one region per predicted class"
    dlg._apply_new_regions(regs)
    assert len(win.regions) > n0
    win._close_analysis(dlg)


def test_region_correlation_offers_matched_region_merges(win):
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY["region_correlation"]))
    dlg._run()
    regs = registry.REGISTRY["region_correlation"].regions_out(dlg.result, win.ds)
    # the write-back merges each region with its closest match into a '{a} + {b}' region
    assert isinstance(regs, dict)
    assert all(" + " in name for name in regs)
    win._close_analysis(dlg)


# --------------------------------------------------------------------------- #
# exporting the feature list with every analysis's columns joined on
# --------------------------------------------------------------------------- #
def _run_and_close(win, step_id):
    dlg = win.open_analysis(AnalysisDialog(win, registry.REGISTRY[step_id]))
    dlg._run()
    win._close_analysis(dlg)
    return dlg


def test_a_dict_result_holding_frames_persists_as_its_table(win):
    """``filter_auc`` returns a dict of DataFrames. ``json.dump`` would stringify them, so the
    run must persist its flat table instead — otherwise it reopens (and exports) as garbage."""
    dlg = AnalysisDialog(win, registry.REGISTRY["filter_auc"])
    full = pd.DataFrame({"mz": [700.5, 750.25], "AUC": [0.9, 0.2], "kept": [True, False]})
    payload = dlg._persist_payload({"full": full, "trimmed": full.iloc[:1],
                                    "peaks": [{"mz": 700.5}], "col": "AUC", "cut": 0.7,
                                    "direction": "either side"})
    assert isinstance(payload, pd.DataFrame) and list(payload.columns) == ["mz", "AUC", "kept"]


def test_an_image_dict_result_still_persists_as_json(win):
    """The other half of the same guard: a dict of arrays really does round-trip, so DGMM's
    ion segmentation must keep persisting as its dict and re-rendering as the image."""
    dlg = _run_and_close(win, "dgmm")
    assert isinstance(dlg._persist_payload(dlg.result), dict)


def test_feature_export_carries_every_per_feature_analysis(win, tmp_path, monkeypatch):
    for step in ("roi_comparison", "pca", "class_comparison"):
        _run_and_close(win, step)

    atts = win._analysis_attachments()
    of = lambda p: [a for a in atts if a.prefix.startswith(p)]      # noqa: E731
    assert any(a.usable for a in of("roi_comparison"))              # per-ion stats attach
    assert any(a.usable for a in of("pca"))                         # loadings pivot per component
    # keyed by lipid class, not m/z: present in the picker, greyed, with a reason
    cls = of("class_comparison")
    assert cls and not any(a.usable for a in cls)
    assert all("m/z" in a.reason for a in cls)

    base = win._centroid_marker_df()
    dlg = FeatureExportDialog(win, base, atts, default_name="fl")
    assert dlg._checks.checked_keys() == {a.key for a in atts if a.usable}   # carry everything

    out = str(tmp_path / "fl.csv")
    monkeypatch.setattr(filedialogs, "get_save_file_name", lambda *a, **k: (out, ""))
    dlg._export()

    assert dlg.path == out
    df = pd.read_csv(out)
    assert len(df) == len(base)                                     # no feature is ever dropped
    assert list(df.columns)[:len(base.columns)] == list(base.columns)
    assert any(c.endswith(".AUC") for c in df.columns)
    assert any(c.endswith(".loading") for c in df.columns)
    assert not any(c.startswith("class_comparison") for c in df.columns)


def test_an_ab_run_is_titled_by_its_contrast(win):
    """Two A-vs-B runs record identical ``inputs`` (every group on the slide), so only the title
    can tell them apart — in History, in the export picker, and in the exported column names."""
    dlg = _run_and_close(win, "roi_comparison")
    assert dlg.run is not None
    assert dlg.run.title.startswith("Region comparison (A vs B) — ")
    assert " vs " in dlg.run.title.split(" — ", 1)[1]
    # and the title never poisons the staleness digest, which still sees only the scope
    assert "a_label" not in dlg.run.inputs


def test_feature_export_can_carry_the_annotation_alone(win, tmp_path, monkeypatch):
    """The other end of the range the user asked for: tick nothing, get just the list."""
    _run_and_close(win, "roi_comparison")
    base = win._centroid_marker_df()
    dlg = FeatureExportDialog(win, base, win._analysis_attachments(), default_name="fl")
    dlg._checks.set_checked([])

    out = str(tmp_path / "bare.csv")
    monkeypatch.setattr(filedialogs, "get_save_file_name", lambda *a, **k: (out, ""))
    dlg._export()
    assert list(pd.read_csv(out).columns) == list(base.columns)
