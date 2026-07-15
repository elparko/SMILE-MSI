"""Per-analysis ScopeBar: every test states + lets you choose the *named feature set*
(the working set, a region's own list, or a saved ★ list) it runs on — and, for the
per-region tests, the groups — instead of silently grabbing the side-panel selection.
Driven headless like test_gui.py."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.gui import main as M  # noqa: E402
from smile_msi.gui.scope import ScopeBar  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def win(app):
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    return w


def _combo_pick(bar, kind, name):
    """Drive the bar's feature dropdown the way a user click would (rebuild + activate)."""
    bar.refresh()
    combo = bar.feat_combo
    idx = next(i for i in range(combo.count()) if combo.itemData(i) == (kind, name))
    bar._on_feat_combo(idx)
    return idx


def _dialog_scope(win, step_id):
    """A live per-analysis ScopeBar the way the app now builds it. Post-plan24 the single-slide
    analyses (co-loc / components / discriminating / region-comparison …) moved off the tab strip
    into an :class:`AnalysisDialog`, so their 'Data in this analysis' bar is built per *open
    analysis* rather than as an eager ``win.<name>_scope`` attribute — but it still self-registers
    on ``win._scope_bars`` (the chokepoint ``_refresh_scope_bars`` drives), so the contract these
    tests assert is unchanged; only where the bar lives is. Mirrors ``_analysis_scope_bar`` in
    test_gui.py."""
    import smile_msi.registry as registry
    from smile_msi.gui.analysisdialog import AnalysisDialog
    return AnalysisDialog(win, registry.REGISTRY[step_id]).scope


def test_every_analysis_has_a_scope_bar(win):
    """Each analysis embeds a ScopeBar, and all of them default to the window's visible feature
    set (so leaving the bar untouched changes nothing). Post-plan24 the single-slide analyses
    build theirs on demand in an AnalysisDialog; Segmentation keeps its eager bar and the Ion
    montage grid builds one when its dialog opens — all self-register on ``win._scope_bars``."""
    vis = sorted(win._visible_mzs())

    # eager (Segmentation tab) + the Ion-montage dialog's bar, built via its real open path
    win._open_montage_dialog()
    for bar in (win.seg_scope, win.montage_scope):
        assert isinstance(bar, ScopeBar)
        assert bar in win._scope_bars
        assert sorted(win._scope_mzs(bar)) == vis            # default == visible features

    # every ScopeBar-backed single-slide analysis now opens as an AnalysisDialog (the retired
    # coloc_scope / comp_scope / stats_scope / cmp_scope tabs): co-localization, components,
    # per-region stats, and region comparison each embed a defaulting bar.
    for step_id in ("colocalize", "coloc_modules", "region_correlation",   # ← old coloc_scope
                    "pca", "nmf",                                          # ← old comp_scope
                    "discriminating_features", "multigroup_features",       # ← old stats_scope
                    "roi_localization", "roi_comparison", "class_comparison"):  # ← old cmp_scope
        bar = _dialog_scope(win, step_id)
        assert isinstance(bar, ScopeBar), step_id
        assert bar in win._scope_bars
        assert sorted(win._scope_mzs(bar)) == vis            # default == visible features


def test_feature_set_dropdown_lists_working_set_and_saved_lists(win):
    """The dropdown offers the working set first, then any saved ★ lists — picking a
    list runs the analysis on its ions without changing the window-wide active set."""
    bar = _dialog_scope(win, "colocalize")               # was the retired win.coloc_scope tab bar
    bar._feature_choice = None
    vis = sorted(win._visible_mzs())
    name = win._save_feature_list_to_library("Panel A",
                                             [{"mz": m, "lipid": ""} for m in vis[:5]])
    try:
        _combo_pick(bar, "list", name)
        assert bar._feature_choice == ("list", name)
        got = sorted(round(m, 4) for m in win._scope_mzs(bar))
        assert got == sorted(round(m, 4) for m in vis[:5])    # analysis runs on the list
        assert f"★ {name}" in bar._feature_text()
        assert win._active_feature_scope == "All slide"       # active set NOT switched

        # back to the working set via the default entry
        _combo_pick(bar, "default", None)
        assert bar._feature_choice is None
        assert sorted(win._scope_mzs(bar)) == vis
    finally:
        win._feature_lists.pop(name, None)
        bar._feature_choice = None
        bar.refresh()


def test_choice_falls_back_when_list_deleted(win):
    """A chosen list that's since deleted falls back to the working set rather than
    running on an empty input, and the dead choice is cleared."""
    bar = _dialog_scope(win, "pca")                      # was the retired win.comp_scope tab bar
    vis = sorted(win._visible_mzs())
    name = win._save_feature_list_to_library("Temp", [{"mz": vis[0], "lipid": ""}])
    bar._feature_choice = ("list", name)
    assert sorted(round(m, 4) for m in bar.feature_mzs()) == [round(vis[0], 4)]

    win._feature_lists.pop(name, None)                        # delete it out from under the bar
    assert sorted(bar.feature_mzs()) == vis                   # fell back to working set
    assert bar._feature_choice is None
    bar.refresh()


def test_rebuild_preserves_choice(win):
    """Refreshing the dropdown (a region/feature change elsewhere) keeps the chosen set
    selected rather than snapping back to the working set."""
    bar = win.seg_scope
    vis = sorted(win._visible_mzs())
    name = win._save_feature_list_to_library("Keep", [{"mz": m, "lipid": ""} for m in vis[:4]])
    try:
        _combo_pick(bar, "list", name)
        bar.refresh()                                         # what a sync hook fires
        assert bar.feat_combo.currentData() == ("list", name)
        assert bar._feature_choice == ("list", name)
    finally:
        win._feature_lists.pop(name, None)
        bar._feature_choice = None
        bar.refresh()


@pytest.mark.xfail(reason="plan24 refactor (OSS_READINESS_AUDIT blocker #5): do_components() and "
                          "comp_scope were retired — Components runs via AnalysisDialog + registry, "
                          "whose run reads the gallery's feature state (all visible peaks), not a "
                          "per-dialog scope bar, so this old scope-bar→worker contract no longer "
                          "holds. Scope-bar feature selection is covered by "
                          "test_feature_set_dropdown_lists_working_set_and_saved_lists.",
                   strict=False)
def test_do_components_runs_on_the_chosen_list(win, monkeypatch):
    """The analysis actually consumes the bar's chosen set: do_components hands that
    list's ions (not the full working set) to the worker."""
    captured = {}

    def fake_run(fn, *args, **kwargs):
        captured["mzs"] = args[1]                             # run_components(ds, mzs, ...)
    monkeypatch.setattr(win, "_run", fake_run)

    vis = sorted(win._visible_mzs())
    name = win._save_feature_list_to_library("Panel3", [{"mz": m, "lipid": ""} for m in vis[:3]])
    try:
        _combo_pick(win.comp_scope, "list", name)
        win.do_components()
        assert sorted(round(m, 4) for m in captured["mzs"]) == sorted(round(m, 4) for m in vis[:3])
    finally:
        win._feature_lists.pop(name, None)
        win.comp_scope._feature_choice = None
        win.comp_scope.refresh()


def test_scope_grouping_uses_chosen_groups(win):
    """_scope_grouping builds the per-region grouping from a ScopeBar's chosen groups,
    and falls back to the window default when fewer than two are chosen."""
    win.regions = []
    win.seg = None
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    names = [r["name"] for r in win.regions]
    assert len(names) == 2

    bar = ScopeBar(win, feature=False, groups=True)
    bar._group_override = list(names)
    labels, gnames = win._scope_grouping(bar)
    assert labels is not None
    assert set(gnames) == set(names)
    assert set(np.unique(labels)) <= {-1, 0, 1}
    assert "Across groups: " + names[0] in bar._group_text()

    bar._group_override = [names[0]]                          # < 2 → default grouping
    assert bar.group_regions() is None
    labels2, _ = win._scope_grouping(bar)
    assert labels2 is not None

    win._scope_bars.remove(bar)
    win.regions = []


@pytest.mark.xfail(reason="plan24 refactor (OSS_READINESS_AUDIT blocker #5): the 'Group by' combo "
                          "(group_by_combo) was removed — per-region grouping moved to the ScopeBar "
                          "groups picker (_group_override / group_regions), covered by "
                          "test_scope_grouping_uses_chosen_groups. The combo and its auto-flip "
                          "between 'Named regions' / 'Segmentation clusters' no longer exist.",
                   strict=False)
def test_group_by_returns_to_regions_after_resegment(win):
    """Regression: the per-group tests must group by the user's ROIs. Running
    segmentation clears the regions and (correctly) parks 'Group by' on the clusters;
    once the user draws ≥2 ROIs it must switch BACK to 'Named regions' rather than
    silently keep grouping by clusters — and an explicit cluster pick must still stick."""
    win.regions = []
    win.seg = None
    win._group_by_auto_value = None
    win.group_by_combo.setCurrentText("Named regions")
    mzs = [p["mz"] for p in win.peaks]

    # run segmentation → regions wiped → only 'Segmentation clusters' is valid
    win._on_seg(M.run_segment(win.ds, mzs, 5, True, False, win.ppm, win.norm))
    assert win.group_by_combo.currentText() == "Segmentation clusters"

    # draw two ROIs → must flip back to the user's regions
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    assert win.group_by_combo.currentText() == "Named regions"
    _labels, names = win._region_grouping()
    assert names is not None and set(names) == {r["name"] for r in win.regions}

    # an explicit cluster pick is respected across later region tweaks
    win.group_by_combo.setCurrentText("Segmentation clusters")
    win._sync_region_combos()
    assert win.group_by_combo.currentText() == "Segmentation clusters"

    win.regions = []
    win.seg = None
    win._group_by_auto_value = None


def test_readout_tracks_global_feature_changes(win):
    """With the working set chosen, hiding a feature updates the readout count through
    _sync_feature_consumers — no explicit pick needed."""
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    bar = win.seg_scope
    bar._feature_choice = None
    bar.refresh()
    n = len(win._visible_mzs())
    assert f"{n} features" in bar._feature_text()

    win.peaks[0]["hidden"] = True
    win._sync_feature_consumers()                            # what a visibility toggle fires
    assert f"{n - 1} features" in bar._feature_text()
    win.peaks[0]["hidden"] = False
    win._sync_feature_consumers()
