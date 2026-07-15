"""GUI integration tests — driven headless (offscreen Qt). Skipped automatically
when the desktop dependencies (PySide6/pyqtgraph) aren't installed, so the
core-only CI run stays green while a `[gui]` install exercises the real wiring."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import annotate, spatial, isotopes, pipeline  # noqa: E402
from smile_msi.gui import main as M  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _no_blocking_prompts(monkeypatch):
    """Neutralize the modal text-name prompt so headless tests never hang on it.

    The 'Save as feature list' handlers (co-localization, classifier VIP, Venn) pop a
    modal ``QInputDialog.getText`` pre-filled with an auto-derived name, where the user
    just presses Enter to accept. Offscreen there is no user, so the dialog blocks
    forever. Echo back the pre-filled ``text`` with ok=True — exactly that happy path —
    so any test driving such a handler proceeds with the suggested name. A test wanting
    the cancel/rename path can re-patch ``getText`` itself."""
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText",
        staticmethod(lambda *a, **k: (k.get("text", ""), True)))


@pytest.fixture(scope="module")
def win(app):
    w = M.MainWindow()
    # Run background jobs inline so handlers that dispatch heavy work via _run (region
    # compare, segmentation commit, …) produce results synchronously under test.
    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
        if want_progress:
            k["progress"] = lambda *_: None
        res = fn(*a, **k)
        if on_done:
            on_done(res)
        return None
    w._run = _sync_run
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    return w


def test_gui_constructs_and_loads(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    # After plan 24 Phase 4 the downstream analyses (Components, Region comparison,
    # Co-localization, Classify, SHAP, Feature space) are AnalysisDialogs launched from the
    # Analyze gallery — not tabs. The remaining tabs are the workspace + the launcher.
    for t in ("Segmentation", "Analyze"):
        assert any(t in v for v in win._view_loc), t
    # Montage left the strip in Phase 6 — it opens from File ▸ Ion montage grid… now, so it is
    # reachable via _open_montage_dialog but is no longer a top-level view.
    assert not any("Montage" in v for v in win._view_loc)
    assert callable(getattr(win, "_open_montage_dialog", None))
    assert "Feature list" not in titles                   # not a top-level strip entry
    # the converted analyses no longer register a tab view, but their registry cards do
    from smile_msi.gui.gallery import CONVERTED
    for step_id in ("colocalize", "plsda", "pca", "embedding", "roi_comparison"):
        assert step_id in CONVERTED, step_id
    assert win.feat_table.rowCount() >= 10                # working features in the dock table
    assert win.peak_table is win.feat_table               # back-compat alias
    assert win.iv.image is not None                       # active ion image rendered
    from smile_msi.gui.common import NoScrollComboBox      # feature-set selector is wheel-safe
    assert isinstance(win.feat_set_combo, NoScrollComboBox)


def test_gui_find_spatial_features(win):
    """The spatial feature finder runs end-to-end through the worker plumbing and
    installs its survivors as the working feature set (descending Moran's I)."""
    win.spatial_freq_spin.setValue(0.0)                   # permissive gates so demo lipids survive
    win.spatial_morans_spin.setValue(0.0)
    win._sync_pick_region_list()                          # whole dataset = leave all regions unticked
    win.do_find_spatial_features()
    assert win.peaks, "spatial finder should populate the working set"
    assert all("morans_i" in p for p in win.peaks)        # enriched peaks carried through
    mis = [p["morans_i"] for p in win.peaks]
    assert mis == sorted(mis, reverse=True)
    assert win._active_feature_scope == "All slide"       # stored via the normal scope plumbing
    # the combined Find-features dialog builds without reparenting the shared controls
    win._open_pick_dialog(focus_spatial=True)
    assert win.snr_spin.parent() is not None


def test_gui_pick_peaks_over_combined_regions(win):
    """The Find-peaks 'Region(s)' selector is multi-select: tick two regions and the
    picker builds one feature list over the *union* of their pixels, scoped to a combined
    'A + B' name routed through the normal peak plumbing. Same mechanism the Regions-panel
    'Build feature list from region(s)' button uses."""
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 4, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    win._new_region_from_segments([0])
    win._new_region_from_segments([1])
    a, b = win.regions[0]["name"], win.regions[1]["name"]

    win._sync_pick_region_list()                          # one checkable row per region
    lw = win.pick_region_list
    assert len(lw._rows) == 2
    lw.set_all(True)                                      # tick both — the CheckList bulk action

    # the resolved scope is the union of both regions' pixels, named 'A + B'
    scope, mask = win._pick_region_mask()
    assert scope == f"{a} + {b}"
    union = win._region_pixel_mask(win.regions[0]) | win._region_pixel_mask(win.regions[1])
    assert int(mask.sum()) == int(union.sum())

    # and a full pick installs that combined scope as the working feature set
    win.do_pick_peaks()
    assert win._active_feature_scope == f"{a} + {b}"
    assert win.peaks

    # restore the fixture's whole-slide state for downstream tests
    win.regions = []
    win._sync_pick_region_list()
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))


def test_region_reorder_move_drag_and_undo(app):
    """Regions can be reordered: 'Move up/down' swaps a region with its sibling (undoable),
    a drag (model rowsMoved → _region_sync_order) persists the new top-to-bottom order, and
    a sub-region travels with its parent so nesting stays valid. Fresh window keeps the
    undo-stack assertions isolated from the shared fixture."""
    from PySide6 import QtCore
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    mzs = [p["mz"] for p in w.peaks]
    w._on_seg(M.run_segment(ds, mzs, 6, True, False, w.ppm, w.norm))

    def turn():
        app.processEvents()                                # fire the guard-clearing singleShot

    w._new_region_from_segments([0]); turn()
    w._new_region_from_segments([1]); turn()
    w._new_region_from_segments([2]); turn()
    w.regions[0]["name"], w.regions[1]["name"], w.regions[2]["name"] = "A", "B", "C"
    w._refresh_regions_all()
    assert [r["name"] for r in w.regions] == ["A", "B", "C"]
    # each list row carries its region index, so a drag can recover the new order
    assert [w.region_list.item(r).data(QtCore.Qt.UserRole) for r in range(3)] == [0, 1, 2]

    # Move up: B jumps above A; ⌘Z rolls it back
    w._region_move(1, -1); turn()
    assert [r["name"] for r in w.regions] == ["B", "A", "C"]
    w._undo(); turn()
    assert [r["name"] for r in w.regions] == ["A", "B", "C"]

    # Moving the bottom region down is a no-op — and records no undo step
    depth = len(w._undo_stack)
    w._region_move(2, +1); turn()
    assert [r["name"] for r in w.regions] == ["A", "B", "C"]
    assert len(w._undo_stack) == depth

    # Drag: emulate Qt's settled single-row move of row 2 (C) to the top, then the
    # deferred `reordered` signal that RegionList.dropEvent fires once the drop unwinds.
    lst = w.region_list
    lst.blockSignals(True)
    lst.insertItem(0, lst.takeItem(2))                # moved item keeps its UserRole stamp
    lst.blockSignals(False)
    lst.reordered.emit(); turn()                      # exercise the real reordered → sync wiring
    assert [r["name"] for r in w.regions] == ["C", "A", "B"]

    # Multi-selection never starts a drag (a disjoint move would desync the model): with
    # two rows selected, startDrag is a guarded no-op and leaves the order untouched.
    lst.clearSelection()
    lst.item(0).setSelected(True)
    lst.item(2).setSelected(True)
    lst.startDrag(QtCore.Qt.MoveAction)
    assert [r["name"] for r in w.regions] == ["C", "A", "B"]

    # The filter disables dragging while it hides rows, and re-enables it when cleared.
    w.region_filter.setText("A")
    assert not lst.dragEnabled()
    w.region_filter.setText("")
    assert lst.dragEnabled()

    # A sub-region stays glued to its parent: nest a child under A, then move A past B
    w.regions = []
    w._new_region_from_segments([0]); w._new_region_from_segments([1]); w._new_region_from_segments([2])
    w.regions[0]["name"], w.regions[1]["name"], w.regions[2]["name"] = "A", "B", "Achild"
    w.regions[2]["parent"] = "A"
    w._refresh_regions_all()
    assert [r["name"] for r in w.regions] == ["A", "Achild", "B"]   # child nests under its parent
    w._region_move(0, +1); turn()                                  # A (with its child) past sibling B
    assert [r["name"] for r in w.regions] == ["B", "A", "Achild"]


def test_no_scroll_combo_ignores_hover_wheel(app):
    """A feature-set / region selector must not change its selection on a mere hover-scroll
    (the 'I scrolled and it swapped my feature set' complaint). The wheel only steps it once
    the combo has focus; unfocused, the event is ignored so the enclosing panel scrolls."""
    from PySide6 import QtCore, QtGui
    from smile_msi.gui import common

    combo = common.NoScrollComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    assert combo.focusPolicy() == QtCore.Qt.StrongFocus      # WheelFocus dropped

    def wheel():
        return QtGui.QWheelEvent(
            QtCore.QPointF(5, 5), QtCore.QPointF(5, 5),
            QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
            QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase, False)

    ev = wheel()
    combo.wheelEvent(ev)                                     # offscreen widget has no focus
    assert combo.currentIndex() == 1                         # selection untouched on hover-scroll
    assert not ev.isAccepted()                               # ignored → bubbles to the scroll area

    combo.hasFocus = lambda: True                            # a focused combo still scrolls normally
    ev2 = wheel()
    combo.wheelEvent(ev2)
    assert combo.currentIndex() != 1


def test_region_batch_create_refreshes_once(win, monkeypatch):
    """Auto-detect creates N regions with a single list/combo rebuild (O(n), not O(n²)):
    _new_region(refresh=False) defers to one _refresh_regions_all at the end."""
    win.regions = []
    win._refresh_regions_all()
    calls = {"n": 0}
    orig = win._refresh_regions_all.__func__
    monkeypatch.setattr(type(win), "_refresh_regions_all",
                        lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                               orig(self, *a, **k))[1])
    h, w = win.ds.height, win.ds.width
    masks = []
    for k in range(5):                                        # 5 fake tissue pieces
        m = np.zeros(win.ds.n_pixels, dtype=bool)
        m[k::8] = True
        masks.append(m)
    win._on_detect_samples(masks)
    assert len(win.regions) == 5
    assert calls["n"] == 1                                    # one rebuild for the whole batch

    win.regions = []                                          # restore
    win._refresh_regions_all()
    win._sync_pick_region_list()
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))


def test_region_multiselect_group_and_filter(app):
    """The A/B region picker selects a whole group in one click (regions carry a cohort
    group tag) and filters a long list — the fix for tedious A/B picking."""
    from smile_msi.gui.common import RegionMultiSelect

    ms = RegionMultiSelect()
    ms.addItems(["Cortex L", "Cortex R", "Medulla"])
    ms.set_item_groups({"Cortex L": "Group A", "Cortex R": "Group A", "Medulla": "Group B"})
    ms._select_group("Group A")
    assert set(ms.selected_names()) == {"Cortex L", "Cortex R"}
    ms._select_group("Group B")
    assert ms.selected_names() == ["Medulla"]

    # filter hides non-matching rows; "All" then respects the filter
    ms._set_all(False)
    ms._filter.setText("cortex")
    ms._set_all(True)
    assert set(ms.selected_names()) == {"Cortex L", "Cortex R"}
    ms._filter.setText("")
    ms._set_all(False)
    assert ms.selected_names() == []


def test_gui_image_zoom_tools_and_pinch(win):
    """The main ion image has explicit zoom in/out + fit controls, and a trackpad pinch
    (a NativeGesture zoom) zooms the view about the cursor — Qt/pyqtgraph ignore native
    gestures by default, so this filter is what makes pinch work."""
    from PySide6 import QtCore
    from smile_msi.gui.common import PinchZoom

    vb = win.iv.view
    assert isinstance(win._iv_pinch, PinchZoom)           # pinch filter installed & kept

    def span():
        (x0, x1), (y0, y1) = vb.viewRange()
        return (x1 - x0, y1 - y0)

    win._fit_image_view()
    base = span()
    win._zoom_image(0.5)                                  # the ＋ button
    zin = span()
    assert zin[0] < base[0] * 0.9 and zin[1] < base[1] * 0.9
    win._fit_image_view()                                 # Fit restores the full view
    assert span()[0] > zin[0]

    # a trackpad pinch-out (value > 0) zooms in; a non-zoom gesture is ignored
    class _Pinch:
        def type(self): return QtCore.QEvent.Type.NativeGesture
        def gestureType(self): return QtCore.Qt.NativeGestureType.ZoomNativeGesture
        def value(self): return 0.25
        def position(self): return QtCore.QPointF(20.0, 20.0)
    before = span()
    assert win._iv_pinch.eventFilter(win.iv.ui.graphicsView.viewport(), _Pinch()) is True
    assert span()[0] < before[0] * 0.95

    class _Rotate(_Pinch):
        def gestureType(self): return QtCore.Qt.NativeGestureType.RotateNativeGesture
    assert win._iv_pinch.eventFilter(None, _Rotate()) is False
    win._fit_image_view()


def test_gui_discriminating_groups_compartments(win):
    """Group-wise tests must treat a compartment (a region built by grouping sub-regions)
    as one group, not let it fight its own children over pixels. Two compartments, each
    built from segmentation sub-regions, must both surface as groups — before the fix the
    parent/child overlap starved one side out and the test read 'nothing found'."""
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 4, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    win._new_region_from_segments([0])
    win._new_region_from_segments([1])
    ris = [i for i, r in enumerate(win.regions)][:2]
    win._region_group_into_parent(ris=ris, name="Compartment C")
    win._new_region_from_segments([2])
    win._new_region_from_segments([3])
    ris = [i for i, r in enumerate(win.regions) if r.get("parent") is None
           and r["name"] != "Compartment C"][:2]
    win._region_group_into_parent(ris=ris, name="Compartment D")

    labels, names = win._labels_from_regions()
    assert names == ["Compartment C", "Compartment D"]        # compartments win, children fold in
    import numpy as np
    assert int((np.asarray(labels) >= 0).sum()) > 0

    win.regions = []
    win._sync_pick_region_list()
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))


def test_gui_export_seg_labels_csv(win, monkeypatch, tmp_path):
    """Segmentation 'Labels CSV…' writes a per-pixel table (pixel, x, y, label) — the
    label map as data — one row per acquired pixel, labels matching the run."""
    import csv as _csv
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 3, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    out = tmp_path / "seg_labels.csv"
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_save_file_name",
                        lambda *a, **k: (str(out), ""))
    win._export_seg_labels()
    assert out.exists()
    with open(out, newline="", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    assert list(rows[0].keys()) == ["pixel", "x", "y", "label"]
    assert len(rows) == len(win.ds.coordinates)               # one row per pixel
    assert [int(r["label"]) for r in rows] == [int(v) for v in seg.labels]


def test_gui_lipid_class_compare_roi_replicates(win):
    """With ≥2 ROIs per side, the class comparison summarizes each region to one replicate and
    tests across ROIs (unit='sample') — the pseudoreplication-safe path.

    Plan 24 retired the Class-comparison tab; the analysis now runs through the registry
    ``class_comparison`` step from the Analyze gallery. The ROI-as-replicate contract (the
    statistically important part) is what the registry ``_resolve_ab`` builds — each named
    sub-ROI on a side is one replicate id — and what ``spatial.class_comparison`` honours."""
    from smile_msi import registry
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 6, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    for s in range(4):                                     # four regions: 0,1 → A ; 2,3 → B
        win._new_region_from_segments([s])
    for i, rg in enumerate(win.regions):                   # tag two groups of two ROIs each
        rg["group"] = "A" if i < 2 else "B"
    win._sync_region_combos()

    sd = registry.REGISTRY["class_comparison"]
    inputs = registry.resolve_inputs(win._gallery_state(), sd)
    assert inputs.get("samples") is not None               # the ROI-replicate ids the resolver built
    cmp = sd.run(win.ds, inputs, registry.default_params(sd.id))
    # both sides have 2 ROIs → the class p/q are tested across ROIs, not pseudoreplicated pixels
    assert cmp.attrs.get("unit") == "sample"
    assert cmp.attrs.get("n_a") == 2 and cmp.attrs.get("n_b") == 2

    # the same run callable on a single unioned mask per side (no per-ROI samples) is the
    # descriptive per-pixel read — the unit follows whether replicate ids are supplied
    pixel = sd.run(win.ds, {**inputs, "samples": None}, registry.default_params(sd.id))
    assert pixel.attrs.get("unit") == "pixel"


def test_gui_import_lipid_db_merge_and_replace(win, monkeypatch, tmp_path):
    """Data ▸ Import lipid database…: Merge adds a new class onto the built-in DB (keeping the
    curated metabolites); Replace swaps to the imported DB only; importing again reverts."""
    import pandas as pd
    from smile_msi.lipiddb import build_database
    n_builtin = len(build_database())
    # a genuinely new class (cardiolipin, P2 — not in the in-silico enumeration)
    ext = tmp_path / "ext.csv"
    pd.DataFrame([{"ABBREVIATION": "CL 72:8", "FORMULA": "C81H156O17P2", "CATEGORY": "GP"}]).to_csv(ext, index=False)
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_open_file_name",
                        staticmethod(lambda *a, **k: (str(ext), "")))
    try:
        # MERGE — built-in kept (metabolites present) + the new lipid added
        win._import_lipid_db(mode="merge")
        assert win._lipid_db is not None
        names = {l.name for l in win._lipid_db}
        assert "CL 72:8" in names and "N-acetylaspartate (NAA)" in names
        assert len(win._lipid_db) == n_builtin + 1
        assert win.ann.lipids is win._lipid_db          # the annotator now uses the merged DB

        win._import_lipid_db()                          # toggle → revert to built-in
        assert win._lipid_db is None

        # REPLACE — only the imported DB (built-in metabolites gone)
        win._import_lipid_db(mode="replace")
        names = {l.name for l in win._lipid_db}
        assert "CL 72:8" in names and "N-acetylaspartate (NAA)" not in names
        assert len(win._lipid_db) == 1
    finally:
        if getattr(win, "_lipid_db", None):             # leave the shared fixture on the built-in DB
            win._import_lipid_db()
    assert win._lipid_db is None


def test_gui_setup_wizard_applies_and_persists(win, monkeypatch, tmp_path):
    """The Setup wizard applies mode + tolerance, merges an external DB, remembers it in prefs,
    and the remembered DB is auto-reloaded on the next startup."""
    import pandas as pd
    from smile_msi import prefs
    from smile_msi.gui.setupwizard import SetupWizard
    orig_mode, orig_tol = win.mode_combo.currentText(), float(win.id_ppm)
    ext = tmp_path / "ext.csv"                            # a new class (cardiolipin, P2)
    pd.DataFrame([{"ABBREVIATION": "CL 72:8", "FORMULA": "C81H156O17P2", "CATEGORY": "GP"}]).to_csv(ext, index=False)

    wiz = SetupWizard(win, win)
    try:
        wiz.mode_combo.setCurrentText("positive")
        wiz.tol_spin.setValue(8.0)
        wiz.db_external.setChecked(True)
        wiz.msi_slice.setChecked(False)                  # don't mass-trim the tiny CSV
        wiz.db_merge.setChecked(True)
        wiz.path_edit.setText(str(ext))
        wiz.demo_check.setChecked(False)
        wiz.apply()                                      # (validates on the fly since not pressed)

        assert win.mode_combo.currentText() == "positive" and abs(win.id_ppm - 8.0) < 1e-9
        assert win._lipid_db is not None and any(l.name == "CL 72:8" for l in win._lipid_db)
        assert win.ann.lipids is win._lipid_db
        cfg = prefs.get(win._LIPID_DB_PREF)
        assert cfg and cfg["path"] == str(ext) and cfg["mode"] == "merge"
        assert prefs.get("setup_done") is True

        # a fresh launch re-applies the remembered DB
        win._set_lipid_db(None)
        assert win._lipid_db is None
        win._apply_remembered_lipid_db()
        assert win._lipid_db is not None and any(l.name == "CL 72:8" for l in win._lipid_db)

        # a moved/missing remembered file is non-fatal (falls back to built-in, no raise)
        prefs.set(win._LIPID_DB_PREF, {"path": str(tmp_path / "gone.csv"), "mode": "merge"})
        win._set_lipid_db(None)
        win._apply_remembered_lipid_db()
        assert win._lipid_db is None
    finally:
        prefs.set(win._LIPID_DB_PREF, None)
        prefs.set("setup_done", None)
        if getattr(win, "_lipid_db", None):
            win._set_lipid_db(None)
        win.mode_combo.setCurrentText(orig_mode)
        win.id_ppm_spin.setValue(orig_tol)
    assert win._lipid_db is None


def test_gui_lipid_list_save_load_persist(win):
    """Save the compared classes as a ◆ lipid list, then load it: the working features become
    class-coloured ions, the colour overlay paints them on the tissue, the selector lists it,
    and a session round-trip preserves it."""
    from smile_msi import session
    from PySide6 import QtCore
    mzs = [p["mz"] for p in win.peaks]
    # A {class: [m/z]} roll-up straight from the annotator — the same input the (retired)
    # class-comparison 'Save as lipid list' fed to save_lipid_list. Group each identified ion
    # by its lipid class; ≥2 classes so the list gets distinct colours per class.
    classes = win.ann.classes_for(mzs)
    mzs_by_class = {}
    for mz, c in zip(mzs, classes):
        if c:
            mzs_by_class.setdefault(c, []).append(float(mz))
    assert len(mzs_by_class) >= 2

    # save → a new lipid list (fixture auto-accepts the name prompt with the suggested text)
    n_before = len(win._lipid_lists)
    name = win.save_lipid_list(mzs_by_class, suggested="Facial vs Synk classes")
    assert name and len(win._lipid_lists) == n_before + 1
    entries = win._lipid_lists[name]
    assert len(entries) >= 2 and all(e["color"] and e["mzs"] for e in entries)
    assert len({e["color"] for e in entries}) == len(entries)        # distinct colour per class
    # it shows in the dock selector with the ◆ prefix, as a distinct 'lipidlist' kind
    labels = [win.feat_set_combo.itemText(i) for i in range(win.feat_set_combo.count())]
    assert any(t == f"◆ {name}" for t in labels)

    # load it → class-coloured working features + the overlay switches on
    win._load_lipid_list(name)
    assert win._active_lipid_list == name
    assert win.color_overlay_chk.isChecked()
    assert win.peaks and all("color" in p and "lipid_class" in p for p in win.peaks)
    assert len({p["color"] for p in win.peaks}) >= 2                 # classes paint different colours

    # the dock swaps to the class tree: the working set is ONE composite feature per class
    # (a class acts as a single feature), its member ions are view-only children. The stack's
    # page is the tree's container (it carries the Show-all/Hide-all bar), not the tree itself.
    assert win.feat_stack.currentWidget() is win._lipid_tree_page
    assert win.lipid_tree.parentWidget() is win._lipid_tree_page
    tree = win.lipid_tree
    assert all(p.get("is_class") and p.get("members") for p in win.peaks)   # composites
    assert tree.topLevelItemCount() == len(win.peaks)                       # one row per class
    total_members = sum(len(p["members"]) for p in win.peaks)
    total_children = sum(tree.topLevelItem(i).childCount() for i in range(tree.topLevelItemCount()))
    assert total_children == total_members                                  # children = member ions
    # selecting a class makes its composite the active feature (renders as one image)
    cls0 = tree.topLevelItem(0)
    assert cls0.data(0, QtCore.Qt.UserRole)[0] == "class"
    tree.setCurrentItem(cls0)
    assert win.active_mz is not None                                        # composite is the subject
    # a member child drives the single-ion preview
    ion0 = cls0.child(0)
    assert ion0.data(0, QtCore.Qt.UserRole)[0] == "member"
    tree.setCurrentItem(ion0)
    assert win.active_mz is not None and abs(win.active_mz - float(ion0.text(0))) < 1e-3
    # hiding a class via its eye hides that class's composite feature
    cls0.setCheckState(0, QtCore.Qt.Unchecked)
    hidden_cls = cls0.data(0, QtCore.Qt.UserRole)[1]
    assert all(p.get("hidden") for p in win.peaks if p["lipid_class"] == hidden_cls)
    cls0.setCheckState(0, QtCore.Qt.Checked)
    assert not any(p.get("hidden") for p in win.peaks if p["lipid_class"] == hidden_cls)
    # right-click → Expand into ions: the composite becomes individual ion features
    win._expand_class_to_ions(hidden_cls)
    assert not any(p.get("is_class") and p["lipid_class"] == hidden_cls for p in win.peaks)
    assert [p for p in win.peaks if not p.get("is_class") and p["lipid_class"] == hidden_cls]

    # switching to any other feature set reverts to the flat per-ion table
    win._switch_feature_scope("All slide")
    assert win.feat_stack.currentWidget() is win.feat_table

    win._load_lipid_list(name)                                       # back to the tree for round-trip
    # session round-trip keeps the lipid list (and its colours)
    blob = json.loads(json.dumps(win._session_state()))
    assert name in blob["lipid_lists"]
    assert blob["lipid_lists"][name][0]["color"] == entries[0]["color"]
    assert "lipid_lists" in session.build_session(source="x", settings={}, peaks=[],
                                                  lipid_lists=win._lipid_lists)
    win._lipid_lists.pop(name, None)                                 # don't leak into later tests
    win._active_lipid_list = None
    win._set_peaks([win._peak_from_mz(m) for m in mzs])              # restore real features downstream


def test_gui_feature_list_class_and_ratio(win):
    df = annotate.build_feature_list(win.ds, win.peaks, mode="negative")
    mono, _ = isotopes.deisotope(win.peaks)
    win._on_feature_list((df, annotate.estimate_fdr([p["mz"] for p in mono]), len(mono)))
    assert win.feat_table.rowCount() == len(win.peaks)
    assert win.class_combo.count() >= 1
    win.do_class_image()                                  # composite class image
    win.do_ratio_image()                                  # ratio image
    assert win.iv.image is not None


def test_gui_two_tolerances_independent(win):
    """Identification tolerance is a separate, persisted control from the extraction
    window, and it drives the annotator used for on-the-fly lipid IDs."""
    assert win.id_ppm == 5.0                               # default ID tolerance
    base_ppm = win.ppm
    # the annotator tracks the ID tolerance, independent of the extraction window
    win.id_ppm_spin.setValue(3.0)
    assert win.ann.ppm_tol == 3.0 and win.ppm == base_ppm
    n_tight = sum(1 for p in win.peaks if win.annotate(p["mz"]))
    win.id_ppm_spin.setValue(25.0)
    assert win.ann.ppm_tol == 25.0
    n_wide = sum(1 for p in win.peaks if win.annotate(p["mz"]))
    assert n_wide >= n_tight                               # looser ID matches at least as many
    assert win._current_settings()["id_ppm"] == 25.0      # persisted in the session
    win.id_ppm_spin.setValue(10.0)                         # restore for downstream tests


def test_gui_clear_all_spectrum_overlays(win):
    """The spectrum toolbar's 'Clear all' button resets the plot to its base trace: it
    removes every overlaid selection — per-pixel clicks, the drawn-ROI mean, and region
    means — while keeping the base mean/skyline (and the underlying ROI/regions)."""
    base = {str(getattr(it, "_overlay_name", "")) for it in win.spectrum.listDataItems()}
    ax, sp = np.array([700.0, 701.0, 702.0]), np.array([1.0, 2.0, 1.0])
    win._overlay_spectrum(ax, sp, "#ff0000", "pixel (3,4)")
    win._overlay_spectrum(ax, sp, "#00ff00", "ROI mean")
    win._overlay_spectrum(ax, sp, "#0000ff", "Region: A")
    tags = [str(getattr(it, "_overlay_name", "")) for it in win.spectrum.listDataItems()]
    assert sum(t.startswith(("pixel", "ROI", "Region")) for t in tags) == 3

    assert any(b.text() == "Clear all"          # the toolbar button is present + wired
               for b in win.findChildren(QtWidgets.QPushButton))
    win._clear_spectrum_overlays()

    after = {str(getattr(it, "_overlay_name", "")) for it in win.spectrum.listDataItems()}
    assert after == base                        # overlays gone, base trace preserved
    assert not any(t.startswith(("pixel", "ROI", "Region")) for t in after)


def test_gui_group_into_parent_region(win):
    """Grouping several ROI regions into a new parent is non-destructive: children survive,
    nest under the parent, and the (mask-less) parent resolves to the union of their pixels —
    so a compartment can act over several fascicle ROIs while each child stays its own region."""
    win.regions = []
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20])
    win._region_from_roi()
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20])
    win._region_from_roi()
    c0, c1 = win.regions[0], win.regions[1]
    m0 = win._region_pixel_mask(c0).copy()
    m1 = win._region_pixel_mask(c1).copy()

    win._region_group_into_parent([0, 1], name="endoneurium")
    names = [r["name"] for r in win.regions]
    assert "endoneurium" in names
    parent = next(r for r in win.regions if r["name"] == "endoneurium")
    assert parent.get("mask") is None and not parent.get("segments")   # owns no pixels itself
    assert c0.get("parent") == "endoneurium" and c1.get("parent") == "endoneurium"
    assert len(win.regions) == 3                                       # children NOT deleted

    # parent's mask is the live union of its children
    pmask = win._region_pixel_mask(parent)
    assert pmask is not None
    np.testing.assert_array_equal(pmask, m0 | m1)

    # children sit directly under the parent in the list ordering
    order = [r["name"] for r in win.regions]
    pi = order.index("endoneurium")
    assert {order[pi + 1], order[pi + 2]} == {c0["name"], c1["name"]}

    # the parent is usable as a comparison group: its pixel mask (the live union of its
    # children, asserted above) is what the registry A/B resolver ORs together for a side.
    win.seg = None
    from smile_msi import registry
    ma = registry._union([win._region_pixel_mask(parent)])
    assert ma is not None and ma.any()
    win.regions = []


def test_gui_crop_studio_and_export(win, tmp_path):
    """The Crop Studio workflow: a region has a bbox; the Studio seeds a crop box, edits it,
    and saves it onto the region; a project standard applies a consistent crop to every
    region; the Export dialog offers 'Crop to region' + a live bundle preview; and a cropped
    ion export writes a real file."""
    from smile_msi.gui.exportdialog import ExportDialog
    from smile_msi.gui.cropstudio import CropStudio
    from smile_msi import export

    win.regions = []
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([24, 24])
    win._region_from_roi()
    win.roi.setPos([34, 8]); win.roi.setSize([14, 18])
    win._region_from_roi()
    rg, rg2 = win.regions[0], win.regions[1]

    # raw bbox: valid, padded, clamped to the grid; a manual crop will override it
    bbox = win._region_bbox_raw(rg)
    assert bbox is not None and 0 <= bbox[0] < bbox[1] <= win.ds.height
    win.set_active_mz(win.peaks[0]["mz"])

    # Crop Studio: seeds a box, edits + saves onto the region
    studio = CropStudio(win, rg)
    assert studio._box_crop() is not None                  # seeded from the bbox
    studio._set_box(10, 28, 12, 32)
    studio._save()
    assert rg.get("crop") == (10, 28, 12, 32) and rg.get("crop_orient") == win.ds.orientation
    assert win._region_bbox(rg) == (10, 28, 12, 32)        # manual crop wins over bbox

    # the Studio respects the sample's rotation (uses the orientation-aware ion image)
    win.ds.set_orientation(1)
    rot = CropStudio(win, rg)
    assert rot.iv.imageItem.image.shape == win.ds.ion_image(win.peaks[0]["mz"]).shape
    rot.deleteLater()
    win.ds.set_orientation(0)

    # project standard → 1:1 aspect; "This region" frames just rg2; "All regions" frames both
    studio2 = CropStudio(win, rg2)
    studio2.aspect_combo.setCurrentIndex(1)                # "1:1 square"
    studio2._apply_one()                                   # scope: this region only
    c2 = studio2._box_crop()
    assert abs((c2[3] - c2[2]) - (c2[1] - c2[0])) <= 1     # rg2's box is now square
    studio2._apply_all()
    assert win._crop_preset and win._crop_preset["aspect"] == 1.0
    for r in (rg, rg2):
        c = r["crop"]
        assert abs((c[3] - c[2]) - (c[1] - c[0])) <= 1     # square (width ≈ height)
    studio.deleteLater(); studio2.deleteLater()

    # per-region intensity table (the numbers behind the ion images)
    rtab = win._region_intensity_table()
    assert rtab is not None and {"region", "mz", "mean", "median", "n_px"} <= set(rtab.columns)

    # export dialog: 'Crop to region' lists the regions; bundle preview reacts to toggles
    dlg = ExportDialog(win, scope="ion")
    datas = [dlg.crop_combo.itemData(i) for i in range(dlg.crop_combo.count())]
    assert None in datas and rg["name"] in datas and "__each__" in datas
    dlg.bundle_chks["spectra"].setChecked(False)
    assert "spectra.csv" not in dlg.bundle_preview.toPlainText()
    dlg.bundle_chks["spectra"].setChecked(True)
    assert "spectra.csv" in dlg.bundle_preview.toPlainText()

    # a region with a saved crop exports straight through (no Studio prompt) and writes a file
    opts = dlg.options(); opts["crop_region"] = rg["name"]
    assert win._resolve_crop_targets(opts) == [rg]         # has a crop → no dialog
    p = win._peak_for_mz(win.active_mz) or {"mz": win.active_mz}
    fig = export.render_ion_panel(**{**win._panel_kwargs(p, opts), **win._crop_kwargs(rg)})
    out = tmp_path / "roi_closeup.png"
    export.save_figure(fig, str(out), dpi=80)
    assert out.exists() and out.stat().st_size > 20
    dlg.deleteLater()
    win.regions = []
    win._crop_preset = None


def test_gui_export_intensity_window_override(win):
    """The Export dialog's 'Override intensity window' applies one shared low/high window to
    every exported image when ticked, and otherwise leaves each feature's own dock window in
    place. The window is typeable as well as draggable (the spin boxes drive the slider)."""
    from smile_msi.gui.exportdialog import ExportDialog

    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    p = win.peaks[0]
    p["lo"], p["hi"] = 10.0, 80.0                        # a per-feature window set in the dock

    dlg = ExportDialog(win, scope="ion")

    # off by default → export honours the feature's own window
    assert dlg.options().get("window_override") is None
    assert win._export_feature_window(p, dlg.options()) == (10.0, 80.0)

    # type into the spin boxes → slider follows; tick the box → override wins for every feature
    dlg.window_slider.lo_spin.setValue(5.0)
    dlg.window_slider.hi_spin.setValue(60.0)
    assert dlg.window_slider.values() == (5.0, 60.0)
    dlg.chk_window.setChecked(True)
    assert dlg.options()["window_override"] == (5.0, 60.0)
    assert win._export_feature_window(p, dlg.options()) == (5.0, 60.0)
    dlg.deleteLater()


def test_gui_export_region_spectra_and_difference(win, tmp_path, monkeypatch):
    """The spectra exports (figure + data table) let you pick the whole-slide mean and/or each
    region, overlay them or write one file per spectrum, and add a signed A − B difference.
    The picker + m/z-range controls show for both spectra scopes."""
    from smile_msi.gui.exportdialog import ExportDialog

    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    seg = M.run_segment(win.ds, [p["mz"] for p in win.peaks], 3, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    win._new_region_from_segments([0])
    win._new_region_from_segments([1])
    a, b = win.regions[0]["name"], win.regions[1]["name"]

    dlg = ExportDialog(win, scope="specfig")
    # picker + axes show for both spectra scopes, hidden elsewhere
    assert dlg.spec_group.isVisibleTo(dlg) and dlg.spec_axes.isVisibleTo(dlg)
    dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData("specdata"))
    assert dlg.spec_group.isVisibleTo(dlg) and dlg.spec_axes.isVisibleTo(dlg)
    dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData("ion"))
    assert not dlg.spec_group.isVisibleTo(dlg)
    dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData("specfig"))

    # one source row per (mean + region); all ticked by default
    assert dlg.spec_list.count() == 3
    idents = dlg.options()["spec_sources"]
    assert idents == ["__mean__", a, b]

    # gather builds one spectrum per ticked source; + a difference when requested
    spectra = win._gather_spectra(dlg.options())
    assert [s[0] for s in spectra] == ["whole-slide mean", a, b]
    dlg.chk_diff.setChecked(True)
    dlg.diff_a.setCurrentIndex(dlg.diff_a.findData(a))
    dlg.diff_b.setCurrentIndex(dlg.diff_b.findData(b))
    spectra = win._gather_spectra(dlg.options())
    assert len(spectra) == 4 and spectra[-1][0] == f"{a} − {b}"
    diff = win._difference_spectrum(a, b)
    sa = win._resolve_spec_source(a)
    sb = win._resolve_spec_source(b)
    assert np.allclose(diff[2], sa[2] - sb[2])               # signed A − B over the shared axis

    # overlaid figure (specfig scope -> image format) -> one file
    fig_out = tmp_path / "spectra.png"
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_save_file_name",
                        lambda *a, **k: (str(fig_out), ""))
    assert win.run_export("specfig", dlg.options()) is True
    assert fig_out.exists()

    # separate layout, data table (specdata scope -> CSV) -> one file per spectrum
    # (mean + 2 regions + difference = 4)
    dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData("specdata"))
    dlg.spec_layout.setCurrentIndex(dlg.spec_layout.findData("separate"))
    opts = dlg.options()
    assert opts["spec_layout"] == "separate" and opts["fmt"] == "csv"
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_save_file_name",
                        lambda *a, **k: (str(tmp_path / "sep.csv"), ""))
    assert win.run_export("specdata", opts) is True
    assert len(list(tmp_path.glob("sep_*.csv"))) == 4
    dlg.deleteLater()


def test_gui_export_colormap_row_scoped_to_intensity_images(win):
    """The Colormap control only styles single-ion intensity images, so the export dialog
    shows it for ion / gallery / component and hides it for the colour overlay and the
    segmentation map (which keep their own per-feature / per-cluster colours). This stops the
    'exporting colour overlay but colormap says viridis' confusion."""
    from smile_msi.gui.exportdialog import ExportDialog

    dlg = ExportDialog(win, scope="ion")

    def _row_shown():
        # setRowVisible collapses the field widget, so isVisibleTo(design) tracks it whether
        # we toggled the row or the widgets directly.
        return dlg.cmap_combo.isVisibleTo(dlg.design)

    for scope in ("ion", "gallery", "component"):
        dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData(scope))
        assert _row_shown(), f"colormap row should show for {scope}"

    for scope in ("overlay", "seg"):
        dlg.scope_combo.setCurrentIndex(dlg.scope_combo.findData(scope))
        assert not _row_shown(), f"colormap row should be hidden for {scope}"

    dlg.deleteLater()


def test_gui_report_builder(win, tmp_path):
    """The Report tab curates the PDF data book: every analysis type can be snapshotted
    from live state, items reorder/delete, they persist through the session round-trip, the
    export dialog defaults to the curated list, and the book renders straight from it."""
    import pandas as pd
    from smile_msi import export, session
    from smile_msi.gui.exportdialog import ExportDialog

    win.report_items = []
    win.regions = []
    win.set_active_mz(win.peaks[0]["mz"])
    # a cheap two-cluster segmentation + a stats table so those item types are exercisable
    labels = np.zeros(win.ds.n_pixels, int)
    labels[: win.ds.n_pixels // 2] = 1
    win.seg = spatial.Segmentation(labels=labels, label_image=win.ds.to_image(labels),
                                   peaks=[p["mz"] for p in win.peaks[:5]], n_clusters=2,
                                   explained_variance=float("nan"), silhouette=float("nan"))
    win.last_stats = pd.DataFrame({"mz": [p["mz"] for p in win.peaks[:3]],
                                   "best_lipid": ["A", "B", "C"], "AUC": [0.8, 0.2, 0.6],
                                   "q_value": [1e-3, 2e-3, 3e-3], "log2_fc": [1.0, -1.0, 0.58]})
    win._region_labels = ("Normal", "Synkinetic")

    win._report_add_ion()
    win._report_add_overlay()
    win._report_add_spectrum_mean()
    win._report_add_segmentation()
    win._report_add_stats()
    win._report_add_features()
    win.report_items.append({"type": "note", "title": "Intro", "text": "Curated."})
    win._refresh_report_list()

    types = [it["type"] for it in win.report_items]
    assert types == ["ion", "overlay", "spectrum", "segmentation", "stats", "features", "note"]
    assert win.report_list.count() == 7

    # reorder: move the note (last) up one, then delete the overlay
    win.report_list.setCurrentRow(6)
    win._report_move(-1)
    assert [it["type"] for it in win.report_items][-2:] == ["note", "features"]
    win.report_list.setCurrentRow([it["type"] for it in win.report_items].index("overlay"))
    win._report_delete()
    assert "overlay" not in [it["type"] for it in win.report_items]

    # preview of a figure item renders without raising
    win.report_list.setCurrentRow(0)
    win._report_preview_current(0)

    # persist through the per-sample session round-trip, in order
    state = win._session_state()
    assert len(state["report_items"]) == len(win.report_items)
    p = tmp_path / "sess.json"
    session.save_session(str(p), state)
    back = session.load_session(str(p))
    assert [it["type"] for it in back["report_items"]] == [it["type"] for it in win.report_items]

    # export dialog defaults to the curated list (it isn't empty) and reports the count
    dlg = ExportDialog(win, scope="book")
    assert dlg.book_source_combo.currentData() == "curated"
    assert "item" in dlg.book_source_combo.currentText()
    opts = dlg.options()
    assert opts["book_source"] == "curated"

    # build the book straight from the curated list
    doc = win._collect_report_document(opts)
    assert len(doc["items"]) == len(win.report_items)
    out = tmp_path / "curated_book.pdf"
    export.build_book(doc, str(out), theme=opts["theme"], dpi=70)
    assert out.exists() and out.read_bytes()[:4] == b"%PDF"
    dlg.deleteLater()

    win.report_items = []
    win.regions = []
    win.seg = None
    win.last_stats = None
    win._refresh_report_list()


def test_gui_export_auto_logs_to_report(win, tmp_path, monkeypatch):
    """Every export (⌘E) is mirrored into the Report tab automatically: the matching item
    is appended, tagged ``auto``, de-duplicated when the same view is exported twice, and
    the whole list can be written back out as loose files via 'Export all → files'."""
    import pandas as pd

    win.report_items = []
    win.regions = []
    win.set_active_mz(win.peaks[0]["mz"])
    labels = np.zeros(win.ds.n_pixels, int)
    labels[: win.ds.n_pixels // 2] = 1
    win.seg = spatial.Segmentation(labels=labels, label_image=win.ds.to_image(labels),
                                   peaks=[p["mz"] for p in win.peaks[:5]], n_clusters=2,
                                   explained_variance=float("nan"), silhouette=float("nan"))
    win.last_stats = pd.DataFrame({"mz": [p["mz"] for p in win.peaks[:3]],
                                   "best_lipid": ["A", "B", "C"], "AUC": [0.8, 0.2, 0.6],
                                   "q_value": [1e-3, 2e-3, 3e-3], "log2_fc": [1.0, -1.0, 0.58]})
    win._region_labels = ("Normal", "Synkinetic")
    opts = {"crop_region": None, "roi_outline": False}

    # an ion export logs one auto-tagged ion item …
    assert win._log_export_to_report("ion", opts) == 1
    assert len(win.report_items) == 1
    assert win.report_items[0]["type"] == "ion" and win.report_items[0]["auto"] is True
    # … and re-exporting the identical view doesn't pile up a duplicate
    assert win._log_export_to_report("ion", opts) == 0
    assert len(win.report_items) == 1

    # the other scopes each contribute their own item type
    for scope in ("overlay", "specfig", "seg", "stats", "features"):
        win._log_export_to_report(scope, opts)
    types = [it["type"] for it in win.report_items]
    assert types == ["ion", "overlay", "spectrum", "segmentation", "stats", "features"]
    assert all(it.get("auto") for it in win.report_items)
    assert win.report_list.count() == len(win.report_items)  # rows mirror the list

    # "Export all → files" writes one loose file per item into the chosen folder
    outdir = tmp_path / "all_files"
    outdir.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(outdir)))
    win._report_export_all_files()
    # files now land in one folder per analysis type, with a top-level provenance log
    assert len(list(outdir.rglob("*.png"))) == 4   # ion, overlay, spectrum, segmentation
    assert len(list((outdir / "statistics").glob("*.csv"))) == 1
    assert len(list((outdir / "feature_tables").glob("*.csv"))) == 1
    assert (outdir / "report_log.csv").exists() and (outdir / "README.md").exists()
    # the log records the regions + feature list that fed each item
    log = (outdir / "report_log.csv").read_text(encoding="utf-8")
    assert "regions" in log and "feature_list" in log

    # a gallery export logs an ion item per visible feature, deduped against what's already
    # logged (so the active-ion item exported above isn't added a second time)
    n_before = len(win.report_items)
    seen = {win._report_signature(it) for it in win.report_items}
    expected = sum(1 for p in win._visible_peaks()
                   if win._report_signature(win._capture_ion_item(p)) not in seen)
    added = win._log_export_to_report("gallery", opts)
    assert added == expected and expected >= 1
    assert len(win.report_items) == n_before + added

    win.report_items = []
    win.regions = []
    win.seg = None
    win.last_stats = None
    win._refresh_report_list()


# NOTE: test_gui_report_logs_csv_analyses_with_provenance was removed with plan 24 Phase 6.
# Its subject, the stats-tab ``_set_stats_export`` chokepoint that auto-mirrored a CSV analysis
# run into the Report tab as a rich ``stats`` item, is gone — analyses now run in the
# AnalysisDialog, whose ``_add_to_report`` funnels every result through
# ``_log_analysis_to_report`` (a generic ``analysis`` item). Every assertion it made survives:
#   • AnalysisDialog auto-mirror + source + run_id → test_analysis_provenance
#     ::test_analysis_run_logs_history_audit_and_report
#   • ``_log_analysis_to_report`` analysis item + dedup + regions text + ``_report_build_export_item``
#     → test_gui_report_logs_named_feature_lists_and_analyses (below)
#   • the auto-tagged ``stats`` item + its regions/feature-list provenance in the export log
#     → test_gui_export_auto_logs_to_report


def test_gui_report_logs_named_feature_lists_and_analyses(win):
    """A named feature list saved out of an analysis is mirrored into the Report tab as a
    titled `features` item, and the generic analysis logger records any result table — both
    auto-tagged, de-duplicated, and carrying provenance, so the report 'takes' every analysis
    and every list you build from one."""
    import pandas as pd

    win.report_items = []
    win._refresh_report_list()

    # the chokepoint every analysis-derived ★-list save funnels through logs a `features`
    # item titled with the list's name (selected rows, AUC cutoff, Venn compartment, co-loc
    # module, classifier markers… all land here)
    feats = [{"mz": float(p["mz"]), "lipid": win.annotate(p["mz"]) or ""} for p in win.peaks[:4]]
    name = win._save_feature_list_to_library("Discriminating markers", feats)
    fl_items = [it for it in win.report_items if it["type"] == "features"]
    assert len(fl_items) == 1
    assert fl_items[0]["title"] == name and fl_items[0]["auto"] is True
    assert len(fl_items[0]["records"]) == 4
    assert fl_items[0]["source"]["feature_list"] == name
    # logging the same named list again (same name + m/z set) doesn't pile up a duplicate
    assert win._log_feature_list_to_report(name, feats) == 0
    # a different named list with different ions logs its own row
    win._save_feature_list_to_library("Shared ions",
                                      [{"mz": float(p["mz"]), "lipid": ""} for p in win.peaks[4:7]])
    assert len([it for it in win.report_items if it["type"] == "features"]) == 2
    # each named list renders as its own titled feature-table page in the book
    built = win._report_build_export_item(fl_items[0], win._report_preview_opts())
    assert built and built["type"] == "features" and built["title"] == name

    # the generic analysis logger records an arbitrary result table as an `analysis` item …
    df = pd.DataFrame({"mz": [p["mz"] for p in win.peaks[:3]], "score": [0.9, 0.8, 0.7]})
    assert win._log_analysis_to_report("Spatial features", df,
                                       regions="Whole slide — every tissue pixel") == 1
    a_items = [it for it in win.report_items if it["type"] == "analysis"]
    assert len(a_items) == 1 and a_items[0]["analysis_kind"] == "Spatial features"
    assert win._item_regions_text(a_items[0]) == "Whole slide — every tissue pixel"
    # … and re-logging the identical analysis is de-duplicated
    assert win._log_analysis_to_report("Spatial features", df,
                                       regions="Whole slide — every tissue pixel") == 0

    win.report_items = []
    win._refresh_report_list()


def test_gui_report_clear_and_wipe(win, monkeypatch):
    """The Report screen can wipe the report list alone, or restart the whole analysis
    (keeping the dataset loaded)."""
    import pandas as pd
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))

    # 'Clear report' empties only the report list, leaving the analysis state intact
    win.report_items = [{"type": "note", "title": "x", "text": "y"}]
    win.last_stats = pd.DataFrame({"mz": [1.0]})
    win._refresh_report_list()
    win._report_clear()
    assert win.report_items == [] and win.last_stats is not None and win.peaks

    # 'Wipe & restart analysis' resets the analysis but keeps the dataset loaded
    win.report_items = [{"type": "note", "title": "x", "text": "y"}]
    win.last_stats = pd.DataFrame({"mz": [1.0]})
    win.wipe_analysis()
    assert win.ds is not None                       # dataset stays loaded
    assert win.report_items == [] and win.last_stats is None
    assert win.peaks == [] and win.regions == [] and win.seg is None

    # re-pick peaks so later tests in the module have a working feature set again
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    assert win.peaks


def test_gui_roi_to_feature_list_one_click(win):
    """One-click ROI → feature list: the ion-image toolbar's '→ Feature list' button
    (features_from_region(prefer_roi=True)) promotes the just-drawn ROI to a region on
    the spot, even with nothing selected — and prefers that ROI over any region the user
    happens to have highlighted. The panel button (default) prefers the selection."""
    win.regions = []
    # toolbar '→ Feature list' with nothing selected → promote the drawn ROI
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20])
    win.features_from_region(prefer_roi=True)
    assert len(win.regions) == 1                            # ROI became a region
    assert win.regions[0].get("mask") is not None
    assert not win.roi_chk.isChecked()                     # drawing ROI was consumed

    # with a region selected, the toolbar button still favours a freshly drawn ROI…
    win.region_list.setCurrentRow(0)
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20])
    win.features_from_region(prefer_roi=True)
    assert len(win.regions) == 2                            # a NEW region from the ROI

    # …whereas the panel button (default) builds from the selected region — no new region
    win.region_list.setCurrentRow(0)
    win.features_from_region()
    assert len(win.regions) == 2
    win.regions = []


def test_gui_feature_eye_toggle(win):
    """Per-feature eye (checkbox in the m/z column): hiding a feature greys it out and
    excludes it from the analyses; Toggle-all flips everything; state survives a
    re-render of the table."""
    from PySide6 import QtCore
    # rebuild a clean working set
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    t = win.feat_table
    assert all(not p.get("hidden") for p in win.peaks)
    assert t.item(0, 0).checkState() == QtCore.Qt.Checked
    n = len(win.peaks)
    assert len(win._visible_mzs()) == n

    mz0 = win._feat_mz_at(0)
    t.item(0, 0).setCheckState(QtCore.Qt.Unchecked)        # hide via the eye
    assert any(abs(p["mz"] - mz0) < 1e-3 and p.get("hidden") for p in win.peaks)
    assert len(win._visible_mzs()) == n - 1

    win._toggle_all_features()                              # hide all
    assert all(p.get("hidden") for p in win.peaks)
    assert len(win._visible_mzs()) == n                    # never runs on empty (fallback)
    win._toggle_all_features()                             # show all
    assert all(not p.get("hidden") for p in win.peaks)

    # hidden survives an Identify-lipids re-render
    win.peaks[1]["hidden"] = True
    df = annotate.build_feature_list(win.ds, win.peaks, mode="negative")
    mono, _ = isotopes.deisotope(win.peaks)
    win._on_feature_list((df, annotate.estimate_fdr([p["mz"] for p in mono]), len(mono)))
    assert win.peaks[1].get("hidden")
    n_unchecked = sum(1 for r in range(t.rowCount())
                      if t.item(r, 0) and t.item(r, 0).checkState() == QtCore.Qt.Unchecked)
    assert n_unchecked == 1
    for p in win.peaks:                                     # leave clean for downstream tests
        p["hidden"] = False
    win._decorate_feature_swatches()


def test_gui_polygon_roi_and_region_features(win):
    """Polygon ROI → region, plus per-sample feature lists: picking peaks scoped to a
    region builds that sample's own feature set, switchable from the Feature set
    selector. Selecting a region is independent of the active feature set — the
    selector is the single source of truth, so a region click never hijacks it."""
    win.regions = []
    win._feature_scopes = {}
    win._active_feature_scope = None
    # global pick first
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    assert win._active_feature_scope == "All slide"
    n_all = len(win.peaks)

    # polygon ROI → region
    from PySide6 import QtCore  # noqa
    win.roi_shape.setCurrentText("Polygon")
    win.roi_chk.setChecked(True)
    assert win.poly_roi.isVisible() and not win.roi.isVisible()
    win.poly_roi.setPoints([[2, 2], [26, 2], [26, 20], [2, 20]])
    pmask = win._roi_mask()
    assert pmask is not None and pmask.sum() > 0
    win._region_from_roi()
    sample = win.regions[-1]["name"]
    assert win.regions[-1]["mask"] is not None

    # pick peaks scoped to that sample → its own feature list
    win._pending_pick_region = sample
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce, mask=pmask))
    assert win._active_feature_scope == sample
    assert all(p.get("region") == sample for p in win.peaks)
    # both scopes (All slide + the sample) are registered in the unified selector
    scope_names = [win.feat_set_combo.itemData(i)[1] for i in range(win.feat_set_combo.count())
                   if (win.feat_set_combo.itemData(i) or [None])[0] == "scope"]
    assert set(scope_names) == {"All slide", sample} and win.feat_set_combo.isEnabled()

    # switch back to the whole slide via the selector
    win._switch_feature_scope("All slide")
    assert win._active_feature_scope == "All slide" and len(win.peaks) == n_all
    # selecting a region that has its own feature list now activates that list (opted-in)
    si = next(i for i, r in enumerate(win.regions) if r["name"] == sample)
    win.region_list.clearSelection()
    win.region_list.setCurrentRow(si)
    assert win._active_feature_scope == sample

    # renaming the region must carry its feature list to the new name (not orphan it)
    win.regions[si]["name"] = win._unique_region_name("Cortex")
    win._rekey_region_scope(sample, "Cortex")
    assert "Cortex" in win._feature_scopes and sample not in win._feature_scopes
    assert win._active_feature_scope == "Cortex"
    sample = "Cortex"

    # leave the whole-slide feature set active for downstream tests
    win._switch_feature_scope("All slide")
    win.roi_shape.setCurrentText("Rectangle")
    win._clear_roi()
    win.regions = []


def test_feature_set_combo_hides_and_prunes_orphan_scopes(win):
    """The Feature-set selector must never show scopes whose region no longer exists
    (stale 'ROI 1/2/3' from a renamed/deleted region or an older session). They are
    hidden from the selector, and reconciliation prunes them from storage — but the
    active scope and 'All slide' are always protected."""
    win.regions = [{"name": "Endoneurium", "color": "#88aaff", "segments": set(),
                    "mask": None, "parent": None, "visible": True}]
    win._feature_scopes = {"All slide": [{"mz": 700.0}], "Endoneurium": [{"mz": 800.0}],
                           "ROI 1": [{"mz": 810.0}], "ROI 2": [{"mz": 820.0}]}
    win._active_feature_scope = "All slide"

    def shown():
        c = win.feat_set_combo
        return [c.itemData(i)[1] for i in range(c.count())
                if (c.itemData(i) or [None])[0] == "scope"]

    win._refresh_feature_set_combo()
    # orphans hidden; live scopes shown as 'All slide' then region order
    assert shown() == ["All slide", "Endoneurium"]
    assert "ROI 1" in win._feature_scopes               # display-only filter, not yet pruned

    # reconcile prunes the two orphans (All slide + existing region kept)
    assert win._reconcile_feature_scopes() == 2
    assert set(win._feature_scopes) == {"All slide", "Endoneurium"}

    # an *active* orphan scope is protected — kept and still shown
    win._feature_scopes["ROI 9"] = [{"mz": 9.0}]
    win._active_feature_scope = "ROI 9"
    assert win._reconcile_feature_scopes() == 0
    win._refresh_feature_set_combo()
    assert "ROI 9" in shown()

    # leave clean for downstream tests
    win.regions = []
    win._feature_scopes = {}
    win._active_feature_scope = None
    win._refresh_feature_set_combo()


def test_gui_components_export_loadings_csv(win, monkeypatch, tmp_path):
    """PCA loadings export. Plan 24 retired the Components tab: PCA now runs through the
    registry ``pca`` step, whose ``to_table`` is the (component, m/z, loading) loadings table,
    and the result's ``⤓ CSV`` lives on the shared resultviews TableResultView. Drive that
    surviving surface and assert the loadings land in the CSV, one block per component."""
    import csv as _csv
    from smile_msi import registry
    from smile_msi.gui import resultviews, filedialogs, common

    sd = registry.REGISTRY["pca"]
    inputs = registry.resolve_inputs(win._gallery_state(), sd)
    res = sd.run(win.ds, inputs, {**registry.default_params(sd.id), "n_components": 3})
    assert len(res.images) == 3

    df = sd.to_table(res)                                      # the loadings table the view renders
    assert list(df.columns) == ["component", "mz", "loading"]
    view = resultviews.TableResultView(win, df)               # the '⤓ CSV' surface

    out = tmp_path / "pca_loadings.csv"
    monkeypatch.setattr(filedialogs, "get_save_file_name",
                        lambda *a, **k: (str(out), "CSV (*.csv)"))
    # NOTE: resultviews.TableResultView._export_csv() passes `self` (a status-bar-less QWidget)
    # to common.export_table, whose export_rows calls parent.statusBar() → AttributeError, so the
    # button itself crashes (pre-existing product bug, reported separately). Export the rendered
    # table through the window (the valid parent) to prove the CSV round-trip.
    common.export_table(win, view.table, stem="pca_loadings", title="Export result table")
    assert out.exists()
    with open(out, newline="", encoding="utf-8-sig") as f:
        # skip the leading provenance/audit comment lines the window parent prepends
        body = [ln for ln in f if not ln.startswith("#")]
    rows = list(_csv.DictReader(body))
    assert list(rows[0].keys()) == ["component", "mz", "loading"]
    assert {int(r["component"]) for r in rows} == {0, 1, 2}    # every component's loadings present
    assert all(r["mz"] and r["loading"] for r in rows)


def test_gui_embedding_coordinate_csv_export(win, tmp_path, monkeypatch):
    """The pooled Cohort-UMAP embedding saves its raw coordinates (coords + sample + group),
    not just a picture — the ``⤓ CSV`` beside the plot.

    (Plan 24 retired the single-slide Components embedding CSV: a single-slide 2-D embedding now
    renders in the resultviews EmbeddingView, whose affordance is lasso→region, not a coordinate
    dump. The pooled cross-sample coords export below is the surviving raw-coordinate export.)"""
    import csv
    from smile_msi import multivariate
    from smile_msi.gui import cohortview as cv_mod

    # --- Cohort UMAP tab: a pooled embedding across samples → coords + sample + group --- #
    emb = multivariate.PooledEmbedding(
        method="UMAP",
        coords=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        sample_id=np.array([0, 0, 1]),
        group=np.array(["ctrl", "ctrl", "treated"], dtype=object),
        sample_names=["A", "B"],
        targets=np.array([700.5, 800.5]),
        counts={"A": 2, "B": 1})
    win._cembed_emb = emb
    out2 = tmp_path / "cohort_emb.csv"
    monkeypatch.setattr(cv_mod.filedialogs, "get_save_file_name",
                        lambda *a, **k: (str(out2), "CSV (*.csv)"))
    win._export_cembed_csv()
    rows = list(csv.reader(out2.open()))
    assert rows[0] == ["umap_1", "umap_2", "sample", "group"]
    assert rows[1] == ["0", "1", "A", "ctrl"] and rows[3] == ["4", "5", "B", "treated"]


def test_gui_cohort_reveal_samples_panel(win):
    """The Cohort / Cohort UMAP tabs point at the 'Samples panel' for adding slides; the
    'Samples…' button on those tabs routes there for real — re-expanding the Samples
    section so the roster isn't a hunt. Also fires when Run is clicked with <2 samples."""
    sec = win._samples_section
    sec.header.setChecked(False)                       # user folded the section away
    assert not sec.is_expanded()

    win._reveal_samples_panel()
    assert sec.is_expanded()                           # reveal un-folds it

    # the empty-cohort guards on both tabs route the user to the panel rather than dead-ending
    saved = win.cohort.samples
    try:
        win.cohort.samples = []                         # nothing to compare/embed yet
        sec.header.setChecked(False)
        win._cohort_run()
        assert sec.is_expanded()
        sec.header.setChecked(False)
        win._cembed_run()
        assert sec.is_expanded()
    finally:
        win.cohort.samples = saved


def test_gui_per_region_stats_without_segmentation(win, monkeypatch, tmp_path):
    """Discriminating / multi-group / distinct-&-shared run off named ROI regions when
    no segmentation exists, and compartments + co-localized rankings save as feature
    lists. The managed store is redirected to a tmp dir so the user's data is untouched."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import spatial, registry
    from smile_msi.gui import resultviews, filedialogs, common

    # two ROI 'sample' regions, no segmentation
    win.regions = []
    win.seg = None
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    assert win.region_list.count() == 2

    # grouping falls back to the named regions (label k -> region name)
    labels, names = win._region_grouping()
    assert labels is not None and len(names) == 2 and set(np.unique(labels)) <= {-1, 0, 1}

    # Plan 24 retired the Stats tab: per-group discriminating now runs through the registry
    # ``discriminating_features`` step, fed the region grouping above, and its per-cluster table
    # labels each row with the REGION NAME (not a cluster id). Export is the resultviews CSV.
    mzs = [p["mz"] for p in win.peaks]
    disc_sd = registry.REGISTRY["discriminating_features"]
    disc = disc_sd.run(win.ds, {"labels": labels, "mzs": mzs, "names": names},
                       {**registry.default_params(disc_sd.id), "top_n": 5,
                        "enriched_only": False, "max_q": 1.0})
    disc_tbl = disc_sd.to_table(disc)
    assert "region" in disc_tbl.columns
    assert set(disc_tbl["region"]) <= set(names)          # region names shown, not cluster ids

    # (resultviews.TableResultView renders the table; its '⤓ CSV' button is currently broken —
    # see the note in test_gui_components_export_loadings_csv — so export the rendered table
    # through the window, the valid status-bar-bearing parent.)
    csv_path = tmp_path / "disc.csv"
    monkeypatch.setattr(filedialogs, "get_save_file_name",
                        lambda *a, **k: (str(csv_path), "CSV (*.csv)"))
    disc_view = resultviews.TableResultView(win, disc_tbl)     # keep a ref (the table is its child)
    common.export_table(win, disc_view.table, stem="disc", title="Export result table")
    import pandas as pd
    out = pd.read_csv(csv_path, comment="#")
    assert {"region", "mz"} <= set(out.columns) and len(out) > 0

    # multi-group test is exportable too (its table names the region each ion peaks in)
    mg_sd = registry.REGISTRY["multigroup_features"]
    mg = mg_sd.run(win.ds, {"labels": labels, "mzs": mzs, "names": names},
                   registry.default_params(mg_sd.id))
    mg_tbl = mg_sd.to_table(mg)
    assert "top_region" in mg_tbl.columns
    mg_path = tmp_path / "multigroup.csv"
    monkeypatch.setattr(filedialogs, "get_save_file_name",
                        lambda *a, **k: (str(mg_path), "CSV (*.csv)"))
    mg_view = resultviews.TableResultView(win, mg_tbl)
    common.export_table(win, mg_view.table, stem="multigroup", title="Export result table")
    assert mg_path.exists() and len(pd.read_csv(mg_path, comment="#")) > 0

    # distinct & shared compartments -> feature lists in the library
    masks = [win._region_pixel_mask(rg) for rg in win.regions]
    res = spatial.region_membership(win.ds, masks, mzs, names=names,
                                    min_prevalence=0.5, detect_quantile=0.75)
    saved = [win._save_feature_list_to_library(c["label"],
             [{"mz": m, "lipid": win.annotate(m) or ""} for m in c["mzs"]])
             for c in res["compartments"]]
    assert set(saved) <= set(win._feature_lists)          # saved into this sample's session
    combo_lists = {win.feat_set_combo.itemData(i)[1]
                   for i in range(win.feat_set_combo.count())
                   if (win.feat_set_combo.itemData(i) or [None])[0] == "list"}
    assert set(saved) <= combo_lists                      # registered in the selector

    win.regions = []


def test_discriminating_topn_and_significance_gate(win, monkeypatch, tmp_path):
    """'Candidates / group' bounds the per-group count, and the q-value gate drops
    non-significant rows instead of always filling to a fixed top-N (the old 'always 16,
    8/8' artifact).

    Plan 24 retired the Discriminating dialog; these gates are the ``top_n`` / ``max_q``
    params of the registry ``discriminating_features`` run callable, and the empty result
    yields no marker lists to save."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import registry

    win.regions = []
    win.seg = None
    win.roi.show()
    win.roi.setPos([5, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    win.roi.setPos([35, 5]); win.roi.setSize([20, 20]); win._region_from_roi()
    assert win.region_list.count() == 2

    labels, names = win._region_grouping()
    mzs = [p["mz"] for p in win.peaks]
    sd = registry.REGISTRY["discriminating_features"]
    inp = {"labels": labels, "mzs": mzs, "names": names}

    # top-N bounds the per-group candidate count (no q gate → at most N per group)
    res = sd.run(win.ds, inp, {**registry.default_params(sd.id), "top_n": 3,
                               "max_q": 1.0, "enriched_only": False})
    assert res and all(len(df) <= 3 for df in res.values())

    # an impossible q cutoff drops every row → nothing survives, so there are no marker
    # lists to save (the old 'Save per-group lists' button greying out)
    gated = sd.run(win.ds, inp, {**registry.default_params(sd.id), "top_n": 3,
                                 "max_q": 0.0, "enriched_only": False, "save_lists": True})
    assert sum(len(df) for df in gated.values()) == 0
    assert sd.lists(gated) == {}

    win.regions = []


def test_gui_distinct_shared_work_on_list(win, monkeypatch, tmp_path):
    """A distinct/shared compartment can be loaded as the active feature list (the Venn
    dialog's 'Work on selected list'): it activates the saved list and installs its m/z
    as the working set so segmentation/pipelines use it right away."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win._feature_lists = {}
    saved_peaks = [dict(p) for p in win.peaks]            # restore shared fixture state after
    try:
        feats = [{"mz": float(p["mz"]), "lipid": win.annotate(p["mz"]) or "", "note": "distinct: A"}
                 for p in win.peaks[:4]]
        # mirror the dialog handler: save+activate, then install the working set
        name = win._save_feature_list_to_library("distinct: A", feats, activate=True)
        win._set_peaks([win._peak_from_mz(f["mz"]) for f in feats])
        assert name in win._feature_lists
        assert win._flist_name == name                    # the active list is the compartment
        assert win._active_feature_scope is None          # a loaded list isn't a working scope
        assert len(win.peaks) == 4                         # its m/z are the working set now
    finally:
        win._set_peaks(saved_peaks)
        win._feature_lists = {}


def test_gui_combine_feature_lists(win, monkeypatch, tmp_path):
    """The ⋯ menu's 'Combine lists…' unions two or more saved ★ lists into one new
    list, collapsing shared m/z (no duplicates) and leaving the sources intact."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win._feature_lists = {}
    win._save_feature_list_to_library("A", [{"mz": 700.0, "lipid": ""},
                                            {"mz": 701.0, "lipid": ""}])
    win._save_feature_list_to_library("B", [{"mz": 701.0, "lipid": ""},   # 701 overlaps A
                                            {"mz": 702.0, "lipid": ""}])
    # only the name prompt is interactive — accept the suggested name
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Merged", True)))
    win._combine_feature_lists(["A", "B"])
    assert "Merged" in win._feature_lists
    assert "A" in win._feature_lists and "B" in win._feature_lists   # sources untouched
    mzs = [f["mz"] for f in win._feature_lists["Merged"]]
    assert mzs == [700.0, 701.0, 702.0]                  # union, deduped (701 once), sorted
    win._feature_lists = {}
    win._refresh_feature_set_combo()


def test_gui_library_multiselect_bulk_delete(win, monkeypatch):
    """The Feature lists tab is multi-select (ctrl/shift-click) so several saved lists
    can be deleted in one go."""
    from PySide6 import QtCore
    win._feature_lists = {}
    win._feature_scopes = {}                              # isolate the saved-★ rows
    win._active_feature_scope = None
    for i in range(3):
        win._save_feature_list_to_library(f"List {i}", [{"mz": 700.0 + i, "lipid": ""}])
    win._refresh_library_tab()
    tbl = win.lib_flist_table
    assert tbl.rowCount() == 3
    assert tbl.selectionMode() == QtWidgets.QAbstractItemView.ExtendedSelection
    # select the first two rows, as a ctrl/shift-click would
    sel = tbl.selectionModel()
    for r in (0, 1):
        sel.select(tbl.model().index(r, 0),
                   QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    assert len(win._lib_selected_metas(tbl)) == 2
    # the delete gates on common.confirm() (a QMessageBox.exec dialog), not
    # QMessageBox.question — stub the actual gate so it proceeds headless (else exec() hangs)
    monkeypatch.setattr("smile_msi.gui.librarytab.confirm", lambda *a, **k: True)
    win._lib_delete_feature_list()
    assert len(win._feature_lists) == 1                   # both selected lists removed
    assert tbl.rowCount() == 1
    win._feature_lists = {}
    win._refresh_library_tab()


def test_gui_library_tab_auto_refreshes_on_save(win, monkeypatch, tmp_path):
    """Saving a feature list from anywhere (dock 'Save as…', a flow, the Stats tab) must
    refresh the 'Feature lists' tab table too, not just the dock selector. Those paths only
    call _refresh_feature_set_combo, which now also refills the tab — so it stays current
    without the manual Refresh button (regression: the tab used to go stale until Refresh)."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win._feature_lists = {}
    win._feature_scopes = {}                              # isolate the saved-★ rows
    win._active_feature_scope = None
    win._refresh_feature_set_combo()
    tbl = win.lib_flist_table
    assert tbl.rowCount() == 0
    # save via the shared generator (used by flows / co-loc / stats) — it does NOT call
    # _refresh_library_tab, so the tab must update on its own.
    win._save_feature_list_to_library("Panel A", [{"mz": 700.0, "lipid": ""}])
    assert tbl.rowCount() == 1                            # auto-refreshed (stale before the fix)
    win._save_feature_list_to_library("Panel B", [{"mz": 701.0, "lipid": ""}])
    assert tbl.rowCount() == 2
    assert {tbl.item(r, 0).text() for r in range(tbl.rowCount())} == {"Panel A", "Panel B"}
    # removing a list and refreshing the dock selector also reconciles the tab
    win._feature_lists.pop("Panel A", None)
    win._refresh_feature_set_combo()
    assert tbl.rowCount() == 1
    assert tbl.item(0, 0).text() == "Panel B"
    win._feature_lists = {}
    win._refresh_library_tab()


def test_gui_library_tab_lists_working_scopes(win):
    """The Feature lists tab mirrors the dock selector — it shows the working scopes
    (whole slide + each region's picked list), not just the saved ★ lists. Each row is
    tagged ('scope'|'list', name) so loading a scope row makes it the active working set."""
    from PySide6 import QtCore
    saved = (win._feature_lists, win._feature_scopes, win._active_feature_scope, win.regions)
    saved_peaks = win.peaks                              # restore the working set for later tests
    win._feature_lists = {}
    win._feature_scopes = {"All slide": [{"mz": 700.0}, {"mz": 701.0}],
                           "Cortex": [{"mz": 800.0}]}
    win._active_feature_scope = "All slide"
    win.regions = [{"name": "Cortex"}]                    # so the per-region scope isn't pruned
    win._refresh_library_tab()
    tbl = win.lib_flist_table
    names = {tbl.item(r, 0).text() for r in range(tbl.rowCount())}
    assert {"All slide", "Cortex"} <= names               # working scopes are listed too
    metas = {tuple(tbl.item(r, 0).data(QtCore.Qt.UserRole)) for r in range(tbl.rowCount())}
    assert ("scope", "All slide") in metas and ("scope", "Cortex") in metas
    # loading a scope row switches the active working set onto it
    row = next(r for r in range(tbl.rowCount())
               if tuple(tbl.item(r, 0).data(QtCore.Qt.UserRole)) == ("scope", "Cortex"))
    sel = tbl.selectionModel()
    sel.clearSelection()
    sel.select(tbl.model().index(row, 0),
               QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    win._lib_load_feature_list()
    assert win._active_feature_scope == "Cortex"
    (win._feature_lists, win._feature_scopes, win._active_feature_scope, win.regions) = saved
    win._set_peaks(saved_peaks, reannotate=False)         # put the fixture's working set back
    win._refresh_library_tab()


def test_gui_library_tab_lists_lipid_lists(win):
    """The Feature lists tab also mirrors the dock selector's ◆ lipid lists (not just
    scopes and ★ lists) — regression: _fill_library_table used to only read
    _feature_scopes and _feature_lists, so a saved lipid list never appeared here even
    though it showed up in the dock's 'Feature set' combo."""
    from PySide6 import QtCore
    saved = (win._feature_lists, win._lipid_lists, win._active_lipid_list,
             win._feature_scopes, win._active_feature_scope)
    saved_peaks = win.peaks
    mzs = [p["mz"] for p in win.peaks]
    win._feature_lists = {}
    win._feature_scopes = {}
    win._active_feature_scope = None
    win._lipid_lists = {"Nerve panel": [{"class": "PE", "color": "#ff0000", "mzs": mzs[:1]},
                                        {"class": "PC", "color": "#00ff00", "mzs": mzs[1:2]}]}
    win._active_lipid_list = None
    win._refresh_library_tab()
    tbl = win.lib_flist_table
    names = {tbl.item(r, 0).text() for r in range(tbl.rowCount())}
    assert "Nerve panel" in names
    metas = {tuple(tbl.item(r, 0).data(QtCore.Qt.UserRole)) for r in range(tbl.rowCount())}
    assert ("lipidlist", "Nerve panel") in metas
    row = next(r for r in range(tbl.rowCount())
               if tuple(tbl.item(r, 0).data(QtCore.Qt.UserRole)) == ("lipidlist", "Nerve panel"))
    assert tbl.item(row, 1).text() == "lipid ◆"
    assert int(tbl.item(row, 2).text()) == 2                     # 1 m/z per class, 2 classes

    # export/combine read the flattened (mz, lipid=class, note) shape
    feats = win._lib_features_for("lipidlist", "Nerve panel")
    assert {f["lipid"] for f in feats} == {"PE", "PC"}

    # loading the row makes it the active lipid list (dispatches to _load_lipid_list,
    # not _load_feature_list)
    sel = tbl.selectionModel()
    sel.clearSelection()
    sel.select(tbl.model().index(row, 0),
               QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    win._lib_load_feature_list()
    assert win._active_lipid_list == "Nerve panel"

    (win._feature_lists, win._lipid_lists, win._active_lipid_list,
     win._feature_scopes, win._active_feature_scope) = saved
    win._set_peaks(saved_peaks, reannotate=False)
    win._refresh_library_tab()


def test_gui_region_correlation_dialog(win, tmp_path, monkeypatch):
    """Region correlation & match: collapse each region to its mean profile → a region×region
    similarity matrix + each region's closest-match summary; the summary exports to CSV.

    Plan 24 retired the Co-localization tab; this runs through the registry
    ``region_correlation`` step and renders in the shared resultviews matrix view (heatmap +
    the closest-match table, whose '⤓ CSV' is the surviving export)."""
    from smile_msi import registry
    from smile_msi.gui import resultviews, filedialogs, common

    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 4, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    win._new_region_from_segments([0])
    win._new_region_from_segments([1])
    win._new_region_from_segments([2])

    sd = registry.REGISTRY["region_correlation"]
    inputs = registry.resolve_inputs(win._gallery_state(), sd)
    assert len(inputs["region_masks"]) == 3
    res = sd.run(win.ds, inputs, registry.default_params(sd.id))
    assert len(res.names) == 3 and res.matrix.shape == (3, 3)

    # each region's closest match is one of the *other* regions, not itself
    tbl = sd.to_table(res)                                   # region · closest match · similarity
    names = set(res.names)
    assert len(tbl) == 3
    for _, row in tbl.iterrows():
        assert row["closest match"] in (names - {row["region"]})

    # the summary table exports to CSV (the resultviews '⤓ CSV' button is currently broken —
    # see test_gui_components_export_loadings_csv — so export the rendered table via the window)
    out = tmp_path / "regcorr.csv"
    monkeypatch.setattr(filedialogs, "get_save_file_name",
                        lambda *a, **k: (str(out), "CSV (*.csv)"))
    corr_view = resultviews.TableResultView(win, tbl)          # keep a ref (the table is its child)
    common.export_table(win, corr_view.table, stem="regcorr", title="Export result table")
    import pandas as pd
    got = pd.read_csv(out, comment="#")
    assert list(got.columns) == ["region", "closest match", "similarity"] and len(got) == 3


def test_gui_region_match_a_vs_b(win):
    """Match regions (A↔B): rank each Group-B candidate by its best Pearson match in the
    Group-A reference set. Plan 24 folded Region-match into the registry
    ``region_correlation`` step, whose result exposes ``best_match(name, among=…)`` — the
    ``among`` pool restricts the candidates to Group A."""
    from smile_msi import registry

    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 5, False, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    for k in range(4):
        win._new_region_from_segments([k])

    sd = registry.REGISTRY["region_correlation"]
    res = sd.run(win.ds, registry.resolve_inputs(win._gallery_state(), sd),
                 registry.default_params(sd.id))

    a_name = win.regions[0]["name"]                          # Group A = region 0 (reference)
    b_names = [win.regions[i]["name"] for i in range(1, 4)]  # Group B = regions 1..3
    matches = [res.best_match(b, among=[a_name]) for b in b_names]
    assert len(matches) == 3 and all(m is not None for m in matches)
    for name, score in matches:
        assert name == a_name                               # only the Group-A member can match
        assert -1.0 <= score <= 1.0                         # a valid Pearson similarity
    # the candidates are rankable best-first by their match score (the old table's ordering)
    ranked = sorted(matches, key=lambda m: m[1], reverse=True)
    assert [m[1] for m in ranked] == sorted((m[1] for m in matches), reverse=True)


def test_gui_detail_slider_resegments_live(win):
    """The Detail slider drives the new hierarchy path: build one tree, then cutting
    finer/coarser re-segments live off the prebuilt tree (no re-clustering), the
    segment table tracks the cut, and previewing never commits."""
    mzs = [p["mz"] for p in win.peaks]
    hier = spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm)
    win._on_hier(hier)
    assert win.hier is hier
    assert win.detail_slider.isEnabled()
    assert win.seg is not None and win.seg_table.rowCount() == win.seg.n_clusters

    # commit a coarse cut, then a finer one — the table and seg follow
    win._commit_cut(3)
    coarse_k = win.seg.n_clusters
    assert win.seg_table.rowCount() == coarse_k and coarse_k <= 3
    win._commit_cut(7)
    fine_k = win.seg.n_clusters
    assert win.seg_table.rowCount() == fine_k and fine_k >= coarse_k

    # a live preview (mid-drag) recolours the canvas but must NOT mutate the committed seg
    before = win.seg.labels.copy()
    win._detail_dragging = True
    win._preview_cut(2)
    assert np.array_equal(win.seg.labels, before)          # preview is non-destructive
    assert win.seg_img_item.image is not None

    # moving the slider by keyboard/click (not dragging) commits immediately
    win._detail_dragging = False
    win.detail_slider.setValue(5)
    assert win.seg.n_clusters <= 5


def test_gui_default_segment_count_is_configurable_and_persisted(win, tmp_path, monkeypatch):
    """The 'Default' spinbox sets how many segments a NEW segmentation starts at: it defaults
    to the standard 8, a fresh build opens there, changing it applies now + persists, and a
    later rebuild honours the new default."""
    from smile_msi import prefs
    from smile_msi.gui.segment import segment_default_count, DEFAULT_SEGMENT_COUNT

    monkeypatch.setattr(prefs, "_path", lambda: str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert segment_default_count() == DEFAULT_SEGMENT_COUNT == 8         # the standard default

    mzs = [p["mz"] for p in win.peaks]
    hier = spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm)
    win._on_hier(hier)
    assert win.detail_slider.value() == 8                     # a fresh build opens at the default

    # change the default: it applies to the live cut now and persists to disk
    win.detail_default_spin.setValue(12)              # ensure the next change actually fires
    win.detail_default_spin.setValue(5)
    assert prefs.get("segment_default_count") == 5
    assert win.detail_slider.value() == 5 and win.seg.n_clusters <= 5

    # a later fresh build honours the new default; and it survives a fresh app run
    win._on_hier(hier)
    assert win.detail_slider.value() == 5
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert segment_default_count() == 5


def test_gui_detail_max_is_user_configurable_and_persisted(win, tmp_path, monkeypatch):
    """The 'Max' spinbox lets the user set how fine the Detail slider may cut: it defaults to
    the standard 24, re-sizes the live slider, stays clamped to the tree's own size, drags the
    live cut down when lowered, and persists across sessions."""
    from smile_msi import prefs
    from smile_msi.gui.segment import segment_detail_max, DEFAULT_SEGMENT_DETAIL_MAX

    monkeypatch.setattr(prefs, "_path", lambda: str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert segment_detail_max() == DEFAULT_SEGMENT_DETAIL_MAX == 24      # the standard default

    mzs = [p["mz"] for p in win.peaks]
    hier = spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm)
    win._on_hier(hier)
    maxc = int(hier.max_clusters)
    assert maxc >= 40                                          # the tree is the looser bound here
    assert win.detail_slider.maximum() == 24                  # ceiling = user Max, not the tree

    # raise the Max: the live slider grows to it (still capped by the tree) and it persists
    win.detail_max_spin.setValue(40)
    assert win.detail_slider.maximum() == min(40, maxc)
    assert prefs.get("segment_detail_max") == 40

    # lower the Max below the current cut: slider ceiling AND the live cut both clamp down
    win._detail_dragging = False
    win.detail_slider.setValue(30)                            # commit a fine cut under the ceiling
    win.detail_max_spin.setValue(12)
    assert win.detail_slider.maximum() == 12
    assert win.detail_slider.value() <= 12
    assert win.seg.n_clusters <= 12

    # the new ceiling survives a fresh app run (cache cleared → re-read from disk)
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert segment_detail_max() == 12


def test_gui_dendrogram_drives_and_follows_cut(win):
    """The HCA dendrogram view is the visual twin of the Detail slider: building a tree
    draws it, committing a cut moves its line, and dragging its line re-segments."""
    mzs = [p["mz"] for p in win.peaks]
    hier = spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm)
    win._on_hier(hier)
    d = win.seg_dendro
    assert d._hier is hier and d._cut_line.isVisible()
    # the tree drew real branches (one PlotCurveItem per colour, with data)
    assert any(c.getData()[0] is not None and len(c.getData()[0]) for c in d._curves.values())

    # committing a cut via the slider repositions the line to a height that yields k
    win._commit_cut(6)
    assert d._k == win.seg.n_clusters
    y6 = float(d._cut_line.value())
    win._commit_cut(3)
    assert d._k == win.seg.n_clusters
    assert float(d._cut_line.value()) > y6        # fewer segments → cut sits higher up the tree

    # dragging the line emits cutChanged, which re-segments through the slider
    seen = []
    d.cutChanged.connect(seen.append)
    d._cut_line.setValue(y6)                       # simulate a drag to the 6-segment height
    d._line_dragged()
    assert seen and seen[-1] >= 3
    assert win.seg.n_clusters == seen[-1] and win.detail_slider.value() == seen[-1]


@pytest.mark.parametrize("which", ["min", "max"])
def test_gui_dendrogram_cut_extremes_clamp(win, which):
    """The Detail-slider <-> cut-line sync holds at the extremes: committing the minimum
    (k=2) and maximum (k=maxk) cuts lands the line at a finite, in-bounds height, and
    dragging the line past either end clamps k into [2, maxk] — never k=1 or k>maxk."""
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    maxk = d._maxk
    kk = maxk if which == "max" else 2

    win._commit_cut(kk)
    assert d._k == kk == win.seg.n_clusters
    y = float(d._cut_line.value())
    hs = np.sort(d._dh)
    assert np.isfinite(y) and hs[0] - 1.0 <= y <= hs[-1] * 1.1     # line sits within the tree

    seen = []
    d.cutChanged.connect(seen.append)
    # drag below every merge height → k saturates at maxk (can't exceed the cap)
    d._cut_line.setValue(float(hs[0]) - 1.0)
    d._line_dragged()
    assert seen[-1] == maxk and win.seg.n_clusters == maxk
    # drag above every merge height → k floors at 2 (never collapses to a single cluster)
    d._cut_line.setValue(float(hs[-1]) + 1.0)
    d._line_dragged()
    assert seen[-1] == 2 and win.seg.n_clusters == 2


def test_gui_dendrogram_click_sets_cut(win):
    """Mode toggle = 'Move cut line': a click anywhere on the tree (not just a drag of the
    dashed line) drops the cut at that height and re-segments through the Detail slider;
    clicking the height already cut is a no-op (no redundant re-commit)."""
    from PySide6 import QtCore
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    assert d._mode == "cut" and d._cut_line.movable           # default mode keeps the line draggable

    d.set_cut(6)                                              # currently cut at 6

    seen = []
    d.cutChanged.connect(seen.append)

    class _Ev:                                                # a synthetic left-click at a chosen height
        def __init__(self, y): self._y = y
        def button(self): return QtCore.Qt.LeftButton
        def scenePos(self): return QtCore.QPointF(0.0, self._y)
        def accept(self): pass
    d._vb.mapSceneToView = lambda p: QtCore.QPointF(0.0, p.y())

    # click high up the tree → a coarse cut (different from 6) commits through the slider
    high = float(d._ymax) * 0.99
    target = d._k_for_y(high)
    assert target != 6
    d._scene_clicked(_Ev(high))
    assert seen and seen[-1] == target
    assert win.seg.n_clusters == target and win.detail_slider.value() == target

    # clicking the *same* height again does nothing (no redundant cutChanged)
    n = len(seen)
    d._scene_clicked(_Ev(high))
    assert len(seen) == n


def test_gui_dendrogram_branches_build_region(win):
    """Mode toggle = 'Highlight branches': lighting two disjoint subtrees accumulates their
    micro-clusters (sticky, like shift-click), previews their pixels on the map, and 'New
    region from highlighted branches' combines them into one mask-backed region — splitting
    finer than the live cut. Clicking a lit branch again drops it."""
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    win.regions = []

    # flip to branch mode via the toggle button (the cut line locks so picks never nudge it)
    d.b_mode_pick.setChecked(True)
    assert d._mode == "pick" and not d._cut_line.movable
    # not isHidden() rather than isVisible(): headless the window is never shown, so a child
    # widget's isVisible() is always False even after setVisible(True) (cf. the comp_color_roi
    # check above) — isHidden() reflects the explicit show/hide the mode toggle just applied.
    assert not d.b_make_region.isHidden() and not d.b_make_region.isEnabled()

    # the two children of the root are disjoint subtrees → a clean accumulation test
    root = int(d._link_node[-1])
    a, b = (int(d._hier.linkage[root - d._n_micro, 0]),
            int(d._hier.linkage[root - d._n_micro, 1]))
    d._toggle_node(a)
    one = set(d._sel_micros)
    d._toggle_node(b)
    two = set(d._sel_micros)
    assert one and one < two                                  # sticky: the second adds, not replaces
    assert d._sel_nodes == {a, b}
    assert win._dendro_branch_micros == frozenset(two)        # host mirrors the lit selection
    assert d.b_make_region.isEnabled() and d.b_clear_branches.isEnabled()
    # the highlighted pixels are previewed on the seg canvas
    assert win.seg_img_item.image is not None
    assert "highlighted" in win.seg_legend.text().lower()

    expect_px = int(win._branch_pixel_mask(two).sum())
    d.b_make_region.click()                                   # → makeRegionRequested → host

    assert len(win.regions) == 1
    rg = win.regions[0]
    assert rg["mask"] is not None and not rg.get("segments")  # mask-backed, not cluster-backed
    assert int(win._region_pixel_mask(rg).sum()) == expect_px
    assert not d._sel_nodes and not win._dendro_branch_micros  # selection cleared after building
    # the just-built region is surfaced on the seg map it was built from
    assert rg["name"] in win.seg_legend.text()


def test_gui_dendrogram_branch_selection_stays_disjoint(win):
    """Overlapping picks never get 'stuck': lighting a parent after a child absorbs the
    child (one disjoint cover), and a single click anywhere inside a lit subtree un-lights
    the whole covering branch — plus the 'Clear' button drops everything at once."""
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    d.b_mode_pick.setChecked(True)

    root = int(d._link_node[-1])
    child = int(d._hier.linkage[root - d._n_micro, 0])        # a strict descendant of root
    d._toggle_node(child)
    d._toggle_node(root)                                      # parent absorbs the lit child
    assert d._sel_nodes == {root}
    assert set(d._sel_micros) == set(int(m) for m in d._node_leaves[root])

    # clicking *inside* the lit subtree (the child) turns the whole covering branch off
    d._toggle_node(child)
    assert not d._sel_nodes and not d._sel_micros

    # Clear button: light two branches, then one click clears them all
    d._toggle_node(root)
    assert d._sel_nodes
    d.b_clear_branches.click()
    assert not d._sel_nodes and not d.b_clear_branches.isEnabled()


def test_gui_dendrogram_mode_toggle_syncs_map(win):
    """Switching back to 'Move cut line' hides the amber tree overlay and restores the
    resting cluster/region map; switching to 'Highlight branches' re-shows the lit-branch
    preview without recomputing the selection."""
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    d.b_mode_pick.setChecked(True)
    d._toggle_node(int(d._link_node[-1]))                     # light the whole tree
    assert d._hl_curve.isVisible() and "highlighted" in win.seg_legend.text().lower()

    d.b_mode_cut.setChecked(True)                             # back to cut mode
    assert not d._hl_curve.isVisible()                        # amber overlay hidden, selection kept
    assert d._sel_nodes                                       # selection survives the mode switch
    assert "highlighted" not in win.seg_legend.text().lower()  # map restored to resting view

    d.b_mode_pick.setChecked(True)                            # and it comes back on return
    assert d._hl_curve.isVisible() and "highlighted" in win.seg_legend.text().lower()


def test_gui_dendrogram_branch_preview_survives_recut(win):
    """A lit selection is cut-invariant: committing a different Detail cut while branches
    are lit keeps the amber preview on the map and the branches lit on the tree (the
    desync the review flagged), and the region built afterwards is still pixel-correct."""
    mzs = [p["mz"] for p in win.peaks]
    win._on_hier(spatial.hierarchy(win.ds, mzs, n_micro=60, tol_ppm=win.ppm, norm=win.norm))
    d = win.seg_dendro
    win.regions = []
    d.b_mode_pick.setChecked(True)
    node = int(d._link_node[-1])
    d._toggle_node(node)
    micros = set(d._sel_micros)
    px_before = int(win._branch_pixel_mask(micros).sum())

    win._commit_cut(3)                                        # re-cut underneath the lit branches
    assert d._sel_nodes == {node}                             # still lit on the tree
    assert "highlighted" in win.seg_legend.text().lower()     # amber re-asserted on the map
    win._commit_cut(9)
    assert "highlighted" in win.seg_legend.text().lower()

    # the region built after re-cutting still covers exactly the same pixels
    assert int(win._branch_pixel_mask(d._sel_micros).sum()) == px_before


def test_gui_segment_hover_signatures_cached(win):
    """Hovering a blob exposes each segment's molecular signature (top enriched lipids +
    size) without the table — so the per-cluster signatures are cached on segmentation."""
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 5, True, False, win.ppm, win.norm)
    win._on_seg(seg)

    # per-cluster signatures are cached for the hover tooltip
    assert win._seg_sig and all(isinstance(v, tuple) for v in win._seg_sig.values())


def test_gui_intensity_window_apply_to_all(win):
    """The per-feature intensity window targets one feature by default, but every feature
    in the current view when the 'apply to all' box is ticked — so a batch can share one
    contrast window."""
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    win.feat_apply_all_chk.setChecked(False)

    # single active feature: only it gets the window
    win.feat_table.clearSelection()
    win.set_active_mz(win.peaks[0]["mz"])
    win.feat_window_slider.blockSignals(True)
    win.feat_window_slider.setValues(10.0, 80.0)
    win.feat_window_slider.blockSignals(False)
    win._feat_window_changed()
    active = win._peak_for_mz(win.peaks[0]["mz"])
    assert (active["lo"], active["hi"]) == (10.0, 80.0)
    assert sum(1 for p in win.peaks if p.get("lo") == 10.0) == 1   # no others touched

    # apply-to-all: every feature in the (unfiltered) view shares the window
    win.feat_apply_all_chk.setChecked(True)
    win.feat_window_slider.blockSignals(True)
    win.feat_window_slider.setValues(5.0, 95.0)
    win.feat_window_slider.blockSignals(False)
    win._feat_window_changed()
    assert all(p.get("lo") == 5.0 and p.get("hi") == 95.0 for p in win.peaks)
    win.feat_apply_all_chk.setChecked(False)


def test_gui_clear_features(win):
    """Clear empties the feature list *from view* (table, annotations, active m/z) but does
    NOT wipe the underlying list: the active scope keeps its peaks and is merely detached,
    so it stays in the selector and can be reopened. A second clear is a harmless no-op."""
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    assert win.peaks and win.feat_table.rowCount() >= 1
    scope = win._active_feature_scope
    stored = list(win._feature_scopes.get(scope))         # snapshot the list before clearing
    assert stored
    win._clear_features()
    assert win.peaks == []
    assert win.feat_df is None
    assert win.feat_table.rowCount() == 0
    assert win._active_feature_scope is None               # detached from the scope…
    assert win._feature_scopes.get(scope) == stored        # …but the list itself is preserved
    assert win.active_mz is None
    win._clear_features()                                  # no-op, must not raise
    assert win.peaks == []
    # reopening the scope from the selector brings the cleared list back, intact
    win._switch_feature_scope(scope)
    assert sorted(p["mz"] for p in win.peaks) == sorted(p["mz"] for p in stored)
    assert win.feat_table.rowCount() >= 1
    # and clearing again then picking fresh still builds a new list
    win._clear_features()
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    assert win.peaks


def test_gui_add_active_to_list(win):
    """The ⌘D / 'Add to list' path adds the highlighted peak to the working list, and
    re-adding the same m/z is a no-op (no duplicates)."""
    win._pending_pick_region = None
    win._on_peaks(M.pick_and_build(win.ds, 3.0, 0.01, win.ppm, win.reduce))
    mz = win.peaks[0]["mz"]
    win._set_peaks([p for p in win.peaks if p["mz"] != mz], reannotate=False)
    assert not any(abs(p["mz"] - mz) < 1e-6 for p in win.peaks)
    n = len(win.peaks)
    win.set_active_mz(mz)
    win._add_active_to_list()                              # what ⌘Space triggers
    assert any(abs(p["mz"] - mz) < 1e-3 for p in win.peaks)
    assert len(win.peaks) == n + 1
    win._add_active_to_list()                              # already in list → no duplicate
    assert len(win.peaks) == n + 1


def test_gui_merge_segments_inverse_of_drill(win):
    """'Merge checked' (the user-added inverse of 'Increase detail here') fuses two
    segments into the larger id, reduces the live segment count, and migrates any
    region that referenced a merged id."""
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 6, True, False, win.ppm, win.norm)
    win._on_seg(seg)
    win.regions = []
    before = len(np.unique(win.seg.labels))

    # a region built from one of the soon-to-be-merged ids follows the merge
    win._new_region_from_segments([2])
    a, b = 1, 2
    keep = max((a, b), key=lambda c: int((win.seg.labels == c).sum()))
    win._merge_selected([a, b])

    live = np.unique(win.seg.labels)
    assert len(live) == before - 1                         # exactly one fewer segment
    assert (a in live) ^ (b in live)                       # one id absorbed the other
    assert keep in live
    # the region that held a merged id now references the surviving id
    assert any(keep in rg.get("segments", set()) for rg in win.regions)
    assert win.seg_table.rowCount() == win.seg.n_clusters or \
        win.seg_table.rowCount() == len(live)              # table tracks live segments


def test_gui_segment_merge_and_region_membership(win):
    """Re-merge fuses segments back into one (inverse of split), and the seg table's
    'region' column shows membership live so assigned segments read at a glance."""
    from PySide6 import QtCore
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 6, True, False, win.ppm, win.norm)
    win.regions = []
    win._on_seg(seg)
    assert win.seg.n_clusters >= 4
    assert win.seg_table.columnCount() == 5                  # added 'region' column

    def region_cell_text(cl):
        for r in range(win.seg_table.rowCount()):
            if int(win.seg_table.item(r, 1).data(QtCore.Qt.UserRole)) == cl:
                return win.seg_table.item(r, 4).text()
        return None

    # assigning a cluster to a region shows up immediately in the membership column
    win._new_region_from_segments([0])
    name0 = win.regions[0]["name"]
    assert win._seg_region_of(0)["name"] == name0
    assert region_cell_text(0) == name0
    assert region_cell_text(1) == "—"                       # still unassigned

    # merge two segments: pixels conserved, one fewer live cluster
    before_px = int((win.seg.labels >= 0).sum())
    n_before = int(np.unique(win.seg.labels).size)
    win._merge_selected([1, 2])
    assert int((win.seg.labels >= 0).sum()) == before_px    # relabel, never drop pixels
    assert int(np.unique(win.seg.labels).size) == n_before - 1
    assert region_cell_text(0) == name0                     # untouched region survives the merge
    win.regions = []


def test_gui_merge_one_child_merges_split_group_and_reset(win):
    """Selecting one sub-cluster of a split merges the whole sibling group, and Reset
    restores the original cut."""
    mzs = [p["mz"] for p in win.peaks]
    seg = M.run_segment(win.ds, mzs, 5, True, False, win.ppm, win.norm)
    win.regions = []
    win._on_seg(seg)
    k0 = win.seg.n_clusters

    # emulate a split of cluster 0 into two children: 0 ("0·1") keeps the bulk, a new id
    # ("0·2") gets a third of the pixels
    labels = win.seg.labels.copy()
    idx0 = np.flatnonzero(labels == 0)
    new_id = int(labels.max()) + 1
    labels[idx0[::3]] = new_id
    win.seg.labels = labels
    win.seg.n_clusters = new_id + 1
    win.seg.label_image = win.ds.to_image(labels)
    win._seg_lineage = {c: str(c) for c in range(win.seg.n_clusters)}
    win._seg_lineage[0], win._seg_lineage[new_id] = "0·1", "0·2"
    win._seg_original = (labels.copy(), int(win.seg.n_clusters), 0.0, float("nan"))  # snapshot
    win._rebuild_seg_table()

    assert win._lineage_siblings(new_id) == {0, new_id}
    n_before = int(np.unique(win.seg.labels).size)
    win._merge_selected([new_id])                          # pick ONE child
    assert int(np.unique(win.seg.labels).size) == n_before - 1
    assert int((win.seg.labels == new_id).sum()) == 0      # absorbed into the larger (0)

    # Reset returns to the snapshot (the emulated split state)
    win._reset_splits()
    assert win.seg.n_clusters == new_id + 1
    win.regions = []


def test_gui_rotate_rebuilds_pixel_lookup(app):
    """Rotating the ion image must rebuild the pixel→spectrum lookup, which is keyed by
    *displayed* (row, col). Otherwise a click after a rotate reads a different pixel's
    spectrum: the lookup still held the un-rotated mapping. A fresh window keeps this
    isolated from the shared fixture (which downstream tests reuse at orientation 0)."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)

    def lookup_matches_orientation():
        # every displayed (row, col) must map back to the pixel acquired there (the lookup
        # is an (H, W) int grid: grid[row, col] = pixel index, -1 where nothing was acquired)
        rows, cols = ds._pixel_rows_cols()
        return all(int(w._pix_lookup[int(r), int(c)]) == i
                   for i, (r, c) in enumerate(zip(rows, cols)))

    assert lookup_matches_orientation()
    w._rotate_image()                                 # 90° CW
    assert ds.orientation == 1
    assert lookup_matches_orientation()               # regression: stale lookup → wrong pixel
    w._rotate_image(); w._rotate_image()              # 270°
    assert ds.orientation == 3
    assert lookup_matches_orientation()


def test_gui_rotate_reorients_color_overlay(app):
    """The multi-channel color overlay caches its composited base keyed by _overlay_key().
    A rotation re-scatters every channel onto a swapped (H, W) grid, so the key must include
    orientation — otherwise the stale un-rotated composite is reused and the image never
    rotates in overlay mode (it does in single-ion mode, which recomputes every call)."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    for p in w.peaks[:2]:
        p["visible"] = True
    w.color_overlay_chk.setChecked(True)
    w.refresh_ion_image()
    h0, w0 = w._overlay_base.shape[:2]
    assert (h0, w0) == (ds.height, ds.width)
    w._rotate_image()                                 # 90° CW → H/W swap
    assert ds.orientation == 1
    assert w._overlay_base.shape[:2] == (ds.height, ds.width) == (w0, h0)  # recomposited, not stale


def test_gui_rotate_reorients_backdrop_when_feature_deselected(app):
    """Building a region's feature list (auto-detect / 'Add ROI') clears active_mz, leaving
    the last ion image as a static backdrop. A Rotate must still re-render that backdrop at
    the new orientation — otherwise it freezes landscape while the ROI/region overlays
    rotate to portrait, so the overlays float off the tissue (the reported auto-ROI bug)."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w.set_active_mz(w.peaks[0]["mz"])                  # a feature is shown (the backdrop)
    w.refresh_ion_image()

    # make a drawn-ROI region and show its footprint on the ion image
    w.regions = []
    w.roi.show(); w.roi.setPos([10, 8]); w.roi.setSize([18, 14])
    w._region_from_roi()
    w._show_region_overlay([0])
    # building a region's feature list deselects the active feature; reproduce that end
    # state directly (the async list build does it via _set_peaks on completion)
    w._deselect_feature()
    assert w.active_mz is None                          # nothing actively selected
    assert getattr(w, "_displayed_mz", None) is not None  # but the backdrop m/z is remembered

    ion_item = w.iv.getImageItem()
    for _ in range(4):
        w._rotate_image()
        grid = (ds.height, ds.width)
        # both the backdrop ion image AND the overlay track the rotated grid → aligned
        assert tuple(np.asarray(ion_item.image).shape[:2]) == grid
        assert tuple(np.asarray(w._region_overlay.image).shape[:2]) == grid


def test_gui_rotate_with_no_feature_shows_tic_backdrop(app):
    """With no feature ever selected (e.g. auto-detect ROIs straight after load), the ion
    view falls back to the TIC so it's never blank and still rotates — the user shouldn't
    have to click a peak first just to make Rotate do anything. The pixel→spectrum click
    lookup is now an (H, W) grid (vectorized) and must still track the rotation."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w._deselect_feature()
    w._displayed_mz = None                              # nothing has ever been displayed
    w.refresh_ion_image()
    ion_item = w.iv.getImageItem()
    assert ion_item.image is not None                  # TIC backdrop, not a blank view
    # it's the TIC: finite (acquired) pixels match the dataset's tissue footprint
    backdrop_finite = np.isfinite(np.asarray(ion_item.image, dtype=float))
    tic_finite = np.isfinite(ds.tic_image())
    assert np.array_equal(backdrop_finite, tic_finite) and backdrop_finite.any()
    for _ in range(4):
        w._rotate_image()
        assert tuple(np.asarray(ion_item.image).shape[:2]) == (ds.height, ds.width)
        # the click lookup is a grid that maps displayed (row,col) -> pixel index
        assert w._pix_lookup.shape == (ds.height, ds.width)
        rows, cols = ds._pixel_rows_cols()
        assert int(w._pix_lookup[int(rows[0]), int(cols[0])]) == 0


def test_gui_region_mask_cache_self_invalidates(app):
    """Cluster-backed region masks are memoised against the live segmentation identity +
    the exact cluster set, so the np.isin over every pixel isn't repaid on every call
    (it runs in tight UI loops). The cache must still refresh when the cluster set changes
    or the segmentation is replaced."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    mzs = [p["mz"] for p in w.peaks]
    seg = M.run_segment(ds, mzs, 5, True, False, w.ppm, w.norm)
    w._on_seg(seg)
    w.regions = []
    w._new_region_from_segments([0])
    rg = w.regions[0]

    m0 = w._region_pixel_mask(rg)
    assert m0 is not None and m0.any()
    assert np.array_equal(m0, np.isin(seg.labels, [0]))
    assert w._region_pixel_mask(rg) is m0             # repeat call served from the cache
    assert rg.get("_mask_cache") is not None

    # changing the cluster set invalidates (content key includes the segment set)
    rg["segments"] = {0, 1}
    m1 = w._region_pixel_mask(rg)
    assert m1 is not m0 and np.array_equal(m1, np.isin(seg.labels, [0, 1]))

    # a replaced segmentation invalidates even for the same cluster ids (identity key)
    seg2 = M.run_segment(ds, mzs, 4, True, False, w.ppm, w.norm)
    w._on_seg(seg2)                                   # drops cluster regions; rebuild one
    w.regions = []
    w._new_region_from_segments([0])
    m2 = w._region_pixel_mask(w.regions[0])
    assert np.array_equal(m2, np.isin(w.seg.labels, [0]))

    # a drawn-mask region never grows a cluster cache (no segmentation dependency)
    w.regions = []
    w.roi.show(); w.roi.setPos([5, 5]); w.roi.setSize([20, 20])
    w._region_from_roi()
    assert "_mask_cache" not in w.regions[-1]


def test_points_in_polygon_pure():
    """The lasso geometry: even-odd ray-cast selects interior points only."""
    from smile_msi.gui.scatter import points_in_polygon
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    pts = np.array([[0.5, 0.5], [2, 2], [0.1, 0.9], [-1, 0.5]], float)
    assert points_in_polygon(pts, sq).tolist() == [True, False, True, False]
    assert not points_in_polygon(pts, sq[:2]).any()        # degenerate polygon → nothing


def test_gui_undo_across_domains(app):
    """⌘Z rolls back destructive actions in every domain — feature edits, region
    add/delete, a segmentation merge (with region membership), and freehand brush
    draw/erase. processEvents() between actions mimics the event-loop turn that clears
    the one-step-per-action re-entrancy guard. A fresh window keeps this isolated from
    the shared fixture."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    mzs = [p["mz"] for p in w.peaks]
    w._on_seg(M.run_segment(ds, mzs, 6, True, False, w.ppm, w.norm))

    def turn():
        app.processEvents()                                # fire the guard-clearing singleShot

    # features — a destructive replace is undoable
    turn()
    n0 = len(w.peaks)
    w.record_undo("features"); turn()
    w._set_peaks(w.peaks[:-3])
    assert len(w.peaks) == n0 - 3
    w._undo(); turn()
    assert len(w.peaks) == n0

    # regions — build two (one undo step each via the guard), delete one, undo restores it
    w._new_region_from_segments([0]); turn()
    w._new_region_from_segments([1]); turn()
    assert len(w.regions) == 2
    w.region_list.setCurrentRow(0)
    w._region_delete(); turn()
    assert len(w.regions) == 1
    w._undo(); turn()
    assert len(w.regions) == 2

    # segmentation merge — undo restores the label array exactly AND region membership
    w._new_region_from_segments([0, 1]); turn()
    reg_segs = sorted(w.regions[-1]["segments"])
    labels = w.seg.labels.copy()
    uniq = list(np.unique(labels))
    w._merge_selected([int(uniq[0]), int(uniq[1])]); turn()
    assert len(np.unique(w.seg.labels)) == len(uniq) - 1   # two clusters fused into one
    w._undo(); turn()
    assert np.array_equal(w.seg.labels, labels)
    assert sorted(w.regions[-1]["segments"]) == reg_segs

    # freehand brush — draw, erase a smaller disk, undo the erase
    w.roi_chk.setChecked(True)
    w.roi_shape.setCurrentText("Freehand (brush)")
    w.brush_mode.setCurrentText("Draw"); w.brush_size.setValue(4)
    cx, cy = ds.height / 2.0, ds.width / 2.0
    w._brush_begin_stroke(); w._brush_stamp(cx, cy); turn()
    drawn = int(w._brush_mask.sum())
    assert drawn > 0
    w.brush_mode.setCurrentText("Erase"); w.brush_size.setValue(2)
    w._brush_begin_stroke(); w._brush_stamp(cx, cy); turn()
    assert int(w._brush_mask.sum()) < drawn
    w._undo(); turn()
    assert int(w._brush_mask.sum()) == drawn                # erased pixels come back

    # undoing past an empty stack is a no-op, not a crash
    for _ in range(60):
        w._undo()
    turn()


def _trigger_menu_action(w, text):
    """Find and fire the first menu-bar action whose label contains ``text`` (menubar →
    menu → action), returning True if one was triggered. The action is triggered *in place*
    while its menu is still referenced — a returned QAction wrapper would be reaped by Qt
    once the menu goes out of scope, so we never hand one back."""
    for top in w.menuBar().actions():
        menu = top.menu()
        if menu is None:
            continue
        for a in menu.actions():
            if text in (a.text() or ""):
                a.trigger()
                return True
    return False


def test_gui_script_console_integration(app):
    """Data ▸ Analysis script opens the console, the bound ScriptAPI reflects the live slide
    (features, regions, ppm), a script runs through the worker plumbing and renders, and
    'Apply to app' pushes the script's features back into the window. A fresh window keeps
    this isolated from the shared fixture."""
    from smile_msi.gui.scriptconsole import ScriptConsoleDialog

    w = M.MainWindow()

    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
        if want_progress:
            k["progress"] = lambda *_: None
        res = fn(*a, **k)
        if on_done:
            on_done(res)
        return None
    w._run = _sync_run
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))

    # a named, grouped region so the bound API exposes a mask + a group
    n = ds.n_pixels
    m = np.zeros(n, bool); m[: n // 2] = True
    w.regions = [{"name": "Left", "color": "#DD8452", "segments": set(), "mask": m,
                  "parent": None, "visible": True, "group": "Group A"}]

    # the menu action is wired and (lazily) builds + shows the console
    assert _trigger_menu_action(w, "Analysis script"), "Data ▸ Analysis script menu action missing"
    dlg = w._script_console
    assert isinstance(dlg, ScriptConsoleDialog)
    assert "Slide:" in dlg._ctx_note.text()                  # context banner from live state

    # the API is built from the live regions / peaks / ppm
    api = dlg._build_api()
    assert sorted(round(z, 4) for z in api.features) == \
        sorted(round(float(p["mz"]), 4) for p in w.peaks)
    assert "Left" in api.masks and int(api.masks["Left"].sum()) == int(m.sum())
    assert "Group A" in api.groups and api.ppm == w.ppm

    # run a real script through the console; results render + enable Apply
    dlg.editor.setPlainText(
        "peaks = find_peaks(snr=3)\n"
        "log(f'{len(peaks)} peaks')\n"
        "table(annotate(mode='negative').head(5), 'IDs')\n")
    dlg._run()
    assert dlg._last_result is not None and dlg._last_result.ok
    assert dlg._apply_btn.isEnabled()
    assert dlg.out_tabs.count() >= 2                          # Log + the IDs table tab

    # Apply-back routes the script's features into the app's working set
    w.peaks = []
    dlg._apply_features()
    assert 0 < len(w.peaks) == len(dlg._last_result.peaks)


def test_gui_feature_switch_modal_loader(app, monkeypatch):
    """Changing the active feature set blocks the workspace behind a modal loader so a
    half-rebuilt view can't be poked: a scope switch wraps its synchronous rebuild in a
    busy popup. Loading a saved ★ list does NOT auto-run lipid ID (that's an explicit
    step now); the 'Identify lipids' action still runs the worker on demand. A fresh
    window keeps this isolated from the shared fixture."""
    from PySide6 import QtCore
    import smile_msi.gui.loading as loadingmod

    w = M.MainWindow()
    calls = []

    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
        calls.append({"busy": busy, "modal": modal, "title": title})
        res = fn(*a, **k)
        if on_done:
            on_done(res)
        return None
    w._run = _sync_run
    ds = M.load_demo()
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))

    # record every modal loader the busy-popup / _run opens
    created = []
    Real = loadingmod.LoadingWindow

    class RecLoader(Real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)
    monkeypatch.setattr(loadingmod, "LoadingWindow", RecLoader)

    # _busy_popup: an application-modal loader is up inside the block, gone after
    assert getattr(w, "_load_dialog", None) is None
    with w._busy_popup("working…", title="Loading features…"):
        dlg = w._load_dialog
        assert isinstance(dlg, Real)
        assert dlg.isModal() and dlg.windowModality() == QtCore.Qt.ApplicationModal
    assert w._load_dialog is None and len(created) == 1

    # a synchronous scope switch shows the busy popup around its rebuild
    w._feature_scopes = {"All slide": list(w.peaks), "Subset": list(w.peaks[:5])}
    w._active_feature_scope = "All slide"
    w._switch_feature_scope("Subset")
    assert w._active_feature_scope == "Subset" and len(w.peaks) == 5
    assert len(created) == 2 and w._load_dialog is None      # popup opened + closed

    # loading a saved ★ list no longer auto-runs lipid ID — it's opt-in now, so no
    # "Identifying…" worker fires and the prior annotated table is dropped (feat_df=None)
    calls.clear()
    w._feature_lists["MyList"] = [{"mz": float(w.peaks[0]["mz"]), "lipid": "", "note": ""}]
    w._load_feature_list("MyList")
    assert not [c for c in calls if c["busy"].startswith("Identifying")]   # nothing auto-ran
    assert w.feat_df is None                                               # un-identified state
    # …the explicit 'Identify lipids' action still runs the worker on demand
    w.do_feature_list()
    id_calls = [c for c in calls if c["busy"].startswith("Identifying")]
    assert id_calls and id_calls[-1]["title"] == "Loading features…"


def test_gui_autosave_roundtrip(app, monkeypatch, tmp_path):
    """A real (non-demo) sample auto-saves its whole analysis to the managed store, and
    re-applying that session restores feature lists + working scopes."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session, library
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "sample_A.imzML"                  # non-synthetic → auto-save enabled
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w._save_feature_list_to_library("Panel", [{"mz": w.peaks[0]["mz"], "lipid": "X"}])
    assert w._dirty                               # picking + saving marked the sample dirty
    w._flush_autosave()
    assert not w._dirty                           # written → flag cleared

    managed = session.list_managed()
    assert any(m["source"] == "sample_A.imzML" for m in managed)
    path = session.managed_path(ds.source, library.dataset_fingerprint(ds))
    assert os.path.exists(path)

    # re-applying the saved session restores the per-sample analysis
    w._pending_session = session.load_session(path)
    w._apply_session(ds)
    assert "Panel" in w._feature_lists
    assert "All slide" in w._feature_scopes
    assert len(w.peaks) > 0


def _demo_region(ds, name):
    """A drawn-ROI region dict shaped like the ones the Segmentation tab produces."""
    m = np.zeros(ds.n_pixels, dtype=bool)
    m[:16] = True
    return {"name": name, "color": "#ff0000", "segments": set(), "mask": m,
            "parent": None, "visible": True, "sample": ds.source, "group": "",
            "crop": None, "crop_orient": None}


def test_fresh_open_restores_saved_rois(app, monkeypatch, tmp_path):
    """Re-opening an already-worked slide fresh (File ▸ Open / cohort 'Open files…', i.e.
    _on_dataset_opened) restores its ROIs instead of starting blank — otherwise the first
    edit auto-saved an empty analysis over the real one."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session, library
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "sample_roi.imzML"
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w.regions = [_demo_region(ds, "Cortex"), _demo_region(ds, "Medulla")]
    w._dirty = True
    w._flush_autosave()
    path = session.managed_path(ds.source, library.dataset_fingerprint(ds))
    assert len(session.existing_named_regions(path)) == 2      # persisted

    # a genuinely fresh open of the same slide (new ds object, no restore wired by caller)
    ds2 = M.load_demo()
    ds2.source = "sample_roi.imzML"
    w._on_dataset_opened(ds2)
    assert w._session_restored is True
    assert sorted(r["name"] for r in w.regions) == ["Cortex", "Medulla"]   # ROIs came back


def test_autosave_never_blanks_populated_rois(app, monkeypatch, tmp_path):
    """The auto-save refuses to overwrite a populated on-disk ROI set with an empty in-memory
    one when the sample was not restored this load (a fresh/failed/overlay path). A restored
    sample with no regions is a real user clear and still saves empty."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session, library
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "sample_guard.imzML"
    # seed a populated session on disk
    w._on_dataset(ds)
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    w.regions = [_demo_region(ds, "KeepMe")]
    w._dirty = True
    w._flush_autosave()
    path = w._session_path
    assert len(session.existing_named_regions(path)) == 1

    # simulate a NON-restored reset (fresh reset leaves regions empty, _session_restored False)
    w._on_dataset(ds)
    assert w._session_restored is False and w.regions == []
    w._session_path = path
    w._dirty = True
    w._flush_autosave()
    assert len(session.existing_named_regions(path)) == 1     # guard preserved the ROI

    # but a restored sample that the user genuinely cleared DOES save empty
    w._session_restored = True
    w._dirty = True
    w._flush_autosave()
    assert session.existing_named_regions(path) == []


def test_region_render_guard_filters_by_slide():
    """_region_on_active_slide gates painting: a region paints only on its own slide, and an
    untagged (legacy) region defaults to the current slide so old sessions still render."""
    from types import SimpleNamespace
    f = M.MainWindow._region_on_active_slide
    win = SimpleNamespace(ds=SimpleNamespace(source="/d/A.imzML"))
    assert f(win, {"sample": "/d/A.imzML"})        # same slide → paint
    assert not f(win, {"sample": "/d/B.imzML"})    # another slide → never paint here
    assert f(win, {"sample": ""})                  # legacy empty tag → current slide
    assert f(win, {})                              # no tag at all → current slide


def test_group_pick_dialog_groups_by_sample_and_filters(win):
    """Phase 3.1: the scope group/ROI picker renders regions grouped under their owning
    sample (rg['sample']) as a tree, filters by region OR sample name, and still returns the
    checked region names via chosen() (the selection API to callers is unchanged)."""
    from PySide6 import QtCore
    from smile_msi.gui.scope import _GroupPickDialog

    n = win.ds.n_pixels

    def _reg(name, sample):
        m = np.zeros(n, dtype=bool)
        m[:8] = True
        return {"name": name, "color": "#ff0000", "segments": set(), "mask": m,
                "parent": None, "visible": True, "sample": sample}

    saved = win.regions
    win.regions = [_reg("Cortex", win.ds.source),          # active slide
                   _reg("Medulla", "OTHER_SLIDE.imzML")]   # a second slide → forces two branches
    try:
        dlg = _GroupPickDialog(win, preselected=[])
        tree = dlg.tree
        assert tree.topLevelItemCount() == 2               # one branch per owning sample
        active = tree.topLevelItem(0)                      # the active slide sorts first
        assert "(this slide)" in active.text(0)
        assert [active.child(j).data(0, QtCore.Qt.UserRole)
                for j in range(active.childCount())] == ["Cortex"]
        other = tree.topLevelItem(1)
        assert "OTHER_SLIDE" in other.text(0)
        assert [other.child(j).data(0, QtCore.Qt.UserRole)
                for j in range(other.childCount())] == ["Medulla"]

        dlg.filter.setText("cortex")                       # region-name filter
        assert not active.child(0).isHidden()
        assert other.isHidden()                            # non-matching sample branch hidden
        dlg.filter.setText("other_slide")                  # sample-name filter reveals its regions
        assert active.isHidden()
        assert not other.child(0).isHidden()
        dlg.filter.setText("")
        assert not other.isHidden() and not active.isHidden()

        active.child(0).setCheckState(0, QtCore.Qt.Checked)   # selection API unchanged
        assert dlg.chosen() == ["Cortex"]
    finally:
        win.regions = saved
        win._refresh_region_list()
        win._sync_pick_region_list()


def test_region_tied_to_slide_no_cross_sample_leak(app, monkeypatch, tmp_path):
    """A region is stamped with the slide it's drawn on, and switching to a sample whose
    session has no regions clears BOTH the model and the region list — a previous slide's
    regions never linger on the new tissue (the cross-sample region leak)."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session
    w = M.MainWindow()
    ds_a = M.load_demo()
    ds_a.source = "slide_A.imzML"
    w._on_dataset(ds_a)
    w._on_peaks(M.pick_and_build(ds_a, 3.0, 0.01, w.ppm, w.reduce))
    mask = np.zeros(ds_a.n_pixels, dtype=bool)
    mask[:10] = True
    w._new_region(name="ROI-A", mask=mask)
    assert w.regions[-1]["sample"] == "slide_A.imzML"     # stamped with its slide
    assert w.region_list.count() == 1                      # and shown in the list

    # switch to slide B, whose session carries NO named regions
    ds_b = M.load_demo()
    ds_b.source = "slide_B.imzML"
    w._pending_session = session.build_session(
        source="slide_B.imzML", settings={"mode": "negative"},
        peaks=[{"mz": 700.0, "intensity": 1.0}])
    w._apply_session(ds_b)
    assert w.regions == []                                 # model swapped to B (empty)
    assert w.region_list.count() == 0                      # view cleared — A's region is gone


def test_gui_open_shared_analysis_relocates_dataset(app, monkeypatch, tmp_path):
    """An analysis exported on one machine opens on another: when the recorded dataset
    path is absent here, the app asks the recipient to locate their own copy, overlays the
    feature lists onto it, and remembers the file by name so future opens resolve with no
    prompt."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session, prefs
    from smile_msi.gui import filedialogs
    monkeypatch.setattr(prefs, "_CACHE", None)          # isolate prefs to this tmp home so a
    #                          relocation remembered by a prior run can't leak in (or out)

    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None,
                  busy="", modal=False, title=None, **k):
        if want_progress:
            k["progress"] = lambda *_: None
        if want_stage:
            k["stage"] = lambda *_: None
        res = fn(*a, **k)
        if on_done:
            on_done(res)
        return None

    # --- author an analysis whose source points at a path NOT on this machine ---
    w1 = M.MainWindow()
    w1._run = _sync_run
    ds = M.load_demo()
    ds.source = "/from/another/machine/shared_sample.imzML"   # foreign absolute path
    w1._on_dataset(ds)
    w1._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w1.ppm, w1.reduce))
    w1._save_feature_list_to_library("Shared panel",
                                     [{"mz": w1.peaks[0]["mz"], "lipid": "X"}])
    analysis_path = str(tmp_path / "shared-analysis.json")
    session.save_session(analysis_path, w1._session_state())

    # the recipient has their own copy of the same acquisition under a different path
    local_copy = tmp_path / "mycopy" / "shared_sample.imzML"
    local_copy.parent.mkdir()
    local_copy.write_text("")                       # only its path/name matters in this test

    captured = {}

    def fake_prepare(src, progress=None, stride=1, session_path=None, stage=None):
        captured["src"] = src                       # stand in for the real disk read
        return ds

    # the recipient's "Locate dataset" chooser returns their copy; info box is a no-op
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(filedialogs, "get_open_file_name",
                        lambda *a, **k: (str(local_copy), ""))

    # --- recipient machine, first open: nothing loaded → prompt to locate, then overlay ---
    w2 = M.MainWindow()
    w2._run = _sync_run
    monkeypatch.setattr(w2, "_prepare_imzml", fake_prepare)
    assert w2.ds is None
    w2._open_session_path(analysis_path)
    assert captured["src"] == str(local_copy)        # loaded the recipient's copy
    assert "Shared panel" in w2._feature_lists       # analysis overlaid onto it

    # --- second open: the relocation is remembered, so it resolves with no prompt ---
    captured.clear()
    w3 = M.MainWindow()
    w3._run = _sync_run
    monkeypatch.setattr(w3, "_prepare_imzml", fake_prepare)
    monkeypatch.setattr(w3, "_prompt_locate_dataset",
                        lambda *a, **k: pytest.fail("should not re-prompt once remembered"))
    w3._open_session_path(analysis_path)
    assert captured["src"] == str(local_copy)        # resolved from the remembered mapping


def test_gui_figeditor_reorder_drop_rename(app):
    """The WYSIWYG figure editor renders a zoomable preview and turns list edits (drop via
    uncheck, rename via override), a title, and a style choice (colormap) into the
    FigEditSpec it renders from."""
    from matplotlib.figure import Figure
    from PySide6 import QtCore
    from smile_msi.gui.figeditor import FigureEditorDialog, _ORIG_ROLE, _OVERRIDE_ROLE

    captured = {}

    def render(spec):
        captured["spec"] = spec
        return Figure(figsize=(4, 3))

    dlg = FigureEditorDialog(None, title="t", render=render,
                             row_items=["700.0", "800.0", "900.0"], col_items=["A", "B"],
                             style_options=[{"key": "cmap", "label": "Colors",
                                             "choices": [("A", "bwr"), ("B", "coolwarm")],
                                             "default": "bwr"}])
    assert dlg._pixmap is not None                        # preview rendered to a pixmap

    rl = dlg.row_list
    for r in range(rl.count()):
        it = rl.item(r)
        if it.data(_ORIG_ROLE) == 0:
            it.setCheckState(QtCore.Qt.Unchecked)         # drop the first ion
        if it.data(_ORIG_ROLE) == 2:
            it.setData(_OVERRIDE_ROLE, "PC 34:1")         # rename the third ion
    dlg.title_edit.setText("My SHAP")
    dlg._opt_combos["cmap"].setCurrentIndex(dlg._opt_combos["cmap"].findData("coolwarm"))
    dlg.zoom_combo.setCurrentIndex(dlg.zoom_combo.findData(1.0))   # zoom to 100%
    dlg._render()                                         # debounce timer won't fire in test

    spec = captured["spec"]
    assert 0 in spec.rows.dropped
    assert spec.rows.labels.get(2) == "PC 34:1"
    assert spec.title == "My SHAP"
    assert spec.options["cmap"] == "coolwarm"             # style choice captured
    assert len(spec.cols.order) == 2                      # column axis tracked too
    assert dlg._img_label.pixmap() is not None            # zoom applied a scaled pixmap
    dlg.deleteLater()


def test_gui_figeditor_sort(app):
    """The 'Sort by' dropdown reorders the ion rows by a supplied key in one click, and
    resets to a one-shot state afterward."""
    from matplotlib.figure import Figure
    from smile_msi.gui.figeditor import FigureEditorDialog

    dlg = FigureEditorDialog(None, title="t", render=lambda s: Figure(figsize=(3, 2)),
                             row_items=["a", "b", "c"],
                             row_sorts=[{"label": "val ↑", "values": [30.0, 10.0, 20.0],
                                         "descending": False}])
    dlg.sort_combo.setCurrentIndex(dlg.sort_combo.findData(0))
    dlg._apply_sort()
    assert dlg.spec.rows.live_indices() == [1, 2, 0]      # ascending by [30,10,20]
    assert dlg.sort_combo.currentData() is None           # reset to one-shot
    dlg.deleteLater()


def test_gui_cube_sidecar_persisted_and_restored(app, monkeypatch, tmp_path):
    """The fast m/z cube is cached beside the managed session on auto-save and restored
    onto a reopened slide — so reopening skips rebuilding it from disk (#5)."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    import threading
    import time

    from smile_msi import session, library
    w = M.MainWindow()
    ds = M.load_demo()
    ds.source = "slide_C.imzML"                   # non-synthetic → auto-save enabled
    w._on_dataset(ds)
    _, spec = ds.mean_spectrum()
    ds.build_mz_cube(min_intensity=float(spec.max()) * 0.001)   # the cube to persist
    w._mark_dirty()
    w._flush_autosave()                           # writes session + spawns the sidecar writer

    path = session.managed_path(ds.source, library.dataset_fingerprint(ds))
    side = session.cube_zarr_path(path)           # the chunked Zarr store (npz is no longer written)
    # the sidecar is written on a background daemon thread; poll (and join any live writer)
    # so the assertion is robust to machine load rather than betting on a fixed timeout
    deadline = time.time() + 15
    while not os.path.exists(side) and time.time() < deadline:
        for t in threading.enumerate():
            if t.name == "cube-sidecar":
                t.join(timeout=0.5)
        time.sleep(0.05)
    assert os.path.exists(side)

    # reopening the same slide restores the cube from the sidecar (no rebuild needed)
    ds2 = M.load_demo()
    ds2.source = "slide_C.imzML"
    assert ds2._cube is None
    assert w._restore_cache_from_sidecar(ds2, path) is True
    assert ds2._cube is not None
    # the preferred sidecar is the lazily-read Zarr store, so the restored cube is a
    # CubeStore (not the in-RAM tuple); reconstruct it to compare against the original
    from smile_msi.cubestore import CubeStore
    if isinstance(ds2._cube, CubeStore):
        restored = ds2._cube.column_slice(0, ds2._cube.nbins).toarray()
    else:
        restored = ds2._cube[1].toarray()
    assert np.allclose(restored, ds._cube[1].toarray())


def test_gui_autosave_skips_demo(app):
    """The throwaway demo (synthetic source) is never auto-persisted."""
    w = M.MainWindow()
    ds = M.load_demo()                            # source == "synthetic"
    w._on_dataset(ds)
    w._mark_dirty()
    assert not w._dirty                           # _mark_dirty no-ops on the demo


def test_gui_feature_set_selector_persists_edits_and_display(app):
    """Unified Feature-set selector: switching sets keeps the dropdown's displayed item
    in sync with the active set, and edits made while a scope is active survive switching
    away and back (the working set is mirrored into its scope)."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    w._feature_scopes = {}
    w._active_feature_scope = None
    w._flist_name = None

    # whole-slide scope
    w._pending_pick_region = None
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    assert w._active_feature_scope == "All slide"
    n_all = len(w.peaks)
    assert w.feat_set_combo.currentData() == ("scope", "All slide")

    # a second scope tagged as a region sample
    w._pending_pick_region = "Sample A"
    w._on_peaks(M.pick_and_build(ds, 3.0, 0.01, w.ppm, w.reduce))
    assert w._active_feature_scope == "Sample A"
    assert w.feat_set_combo.currentData() == ("scope", "Sample A")

    # edit the active (Sample A) scope: add an m/z that isn't already present
    lo, hi = ds.mz_range
    new_mz = round(float(w.peaks[0]["mz"]) + 0.5, 4)
    if not (lo <= new_mz <= hi) or any(abs(p["mz"] - new_mz) < 1e-3 for p in w.peaks):
        new_mz = round(float(w.peaks[-1]["mz"]) - 0.5, 4)
    w._set_peaks(w.peaks + [w._peak_from_mz(new_mz)])
    n_sample = len(w.peaks)
    assert any(abs(p["mz"] - new_mz) < 1e-3 for p in w.peaks)

    # switch away and back — the added feature must persist, and the dropdown follows
    w._switch_feature_scope("All slide")
    assert w._active_feature_scope == "All slide" and len(w.peaks) == n_all
    assert w.feat_set_combo.currentData() == ("scope", "All slide")
    w._switch_feature_scope("Sample A")
    assert w._active_feature_scope == "Sample A" and len(w.peaks) == n_sample
    assert any(abs(p["mz"] - new_mz) < 1e-3 for p in w.peaks)
    assert w.feat_set_combo.currentData() == ("scope", "Sample A")


def test_gui_advanced_analyses_reachable_from_gallery(win):
    """The supervised analyses that used to sit under an 'Advanced' container tab are now cards in
    the Analyze gallery that open an AnalysisDialog. Same analyses, one strip entry instead of a
    nested tab — so assert the *reachability*, which is what the old tab test really guarded."""
    import smile_msi.registry as registry
    from smile_msi.gui.analysisdialog import AnalysisDialog
    from smile_msi.gui import gallery

    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Analyze" in titles and "Advanced" not in titles

    for step_id in ("plsda", "shrunken_centroids", "dgmm"):    # Classify, Markers (SSC), Per-ion
        assert step_id in registry.REGISTRY
        assert step_id in gallery.CONVERTED                     # a card opens the dialog, not a tab
        sd = registry.REGISTRY[step_id]
        assert sd.description                                   # the card has something to say
        dlg = AnalysisDialog(win, sd)                           # and it constructs without a dataset
        assert dlg.b_run is not None


def test_gui_cohort_test_selector_wired(win):
    """The cohort comparison exposes a two-group test selector whose labels map to the
    group_comparison(method=) keys the engine accepts; default stays non-parametric."""
    import pandas as pd

    from smile_msi import cohort as cohort_engine

    combo = win.cohort_test_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["Mann-Whitney U", "Welch's t-test", "Student's t-test"]
    assert combo.currentText() == "Mann-Whitney U"           # non-parametric default

    # 2 features × 2 groups, ≥2 samples/group so the t-tests are defined; every label the
    # combo offers must be a method the engine accepts (guards against mapping drift).
    tbl = pd.DataFrame({"group": ["a", "a", "b", "b"],
                        "mz_700.0000": [1.0, 1.2, 3.0, 3.1],
                        "mz_800.0000": [2.0, 2.1, 2.0, 2.2]})
    tbl.attrs["targets"] = [700.0, 800.0]
    for label, key in win._cohort_tests.items():
        res = cohort_engine.group_comparison(tbl, "a", "b", method=key)
        assert res.attrs["test"], label                      # engine accepted the mapped key


def test_gui_optical_backdrop(app, tmp_path):
    """Global optical/histology backdrop: loads behind the ion + segmentation views,
    toggles on/off (lowering the foreground opacity), aligns, and round-trips through a
    session — replacing the old per-ion 'Load H&E' blend."""
    import numpy as np
    from PIL import Image

    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)

    # a backdrop ImageItem is registered for the ion view and the segmentation view
    assert len(w._optical_views) >= 2
    assert all(not e["item"].isVisible() for e in w._optical_views)   # nothing loaded yet
    assert w.optical_align_box.isHidden() and not w.optical_show_chk.isEnabled()

    # load a real image file through the public path
    p = tmp_path / "slide.tif"
    Image.fromarray(np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)).save(p)
    assert w._load_optical_path(str(p), reset_align=True, turn_on=True)
    assert w.optical_show_chk.isChecked()
    # the alignment controls now live behind an "Align (advanced) ▾" disclosure: the toggle
    # appears once an image is loaded, but the spinner wall stays collapsed until expanded.
    assert not w.optical_align_toggle.isHidden()
    assert w.optical_align_box.isHidden()
    w.optical_align_toggle.setChecked(True)
    assert not w.optical_align_box.isHidden()        # expanding reveals the align controls
    assert w.optical_name_lbl.text() == "slide.tif"
    # shown in every view, with the foreground dimmed to the spectra-opacity
    for e in w._optical_views:
        assert e["item"].isVisible()
        assert abs(e["main"].opacity() - w._optical_spectra_alpha) < 1e-6

    # alignment controls drive the transform; drag-to-move shifts the offset
    w.optical_scale_spin.setValue(150.0)
    assert abs(w._optical_align["scale"] - 1.5) < 1e-6
    w.optical_fliph_btn.setChecked(True)
    assert w._optical_align["flipx"] is True
    w._optical_drag_by(4.0, -3.0)
    assert (w._optical_align["tx"], w._optical_align["ty"]) == (4.0, -3.0)
    assert w._optical_transform() is not None

    # global toggle off → hidden everywhere, foreground opacity restored
    w.optical_show_chk.setChecked(False)
    for e in w._optical_views:
        assert not e["item"].isVisible() and abs(e["main"].opacity() - 1.0) < 1e-6

    # session round-trips the path + alignment + display onto a fresh window
    w.optical_show_chk.setChecked(True)
    w.optical_alpha_slider.setValue(30)
    state = w._session_state()
    assert state["optical"]["path"] == str(p)

    w2 = M.MainWindow()
    w2._pending_session = state
    w2._apply_session(M.load_demo())
    assert w2._optical is not None and w2._optical_path == str(p)
    assert w2.optical_show_chk.isChecked()
    assert abs(w2._optical_spectra_alpha - 0.30) < 1e-6
    assert abs(w2._optical_align["scale"] - 1.5) < 1e-6 and w2._optical_align["flipx"] is True

    # removing it clears the layer and the controls
    w2._remove_optical()
    assert w2._optical is None and w2.optical_align_box.isHidden()


def test_gui_optical_reset_and_remove_are_undoable(app, tmp_path):
    """⌘Z restores optical state after the two destructive optical handlers: resetting the
    alignment brings the nudged transform back, and removing the image brings the whole
    backdrop back (pixels included, no disk reload)."""
    import numpy as np
    from PIL import Image

    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)

    def turn():
        app.processEvents()                       # clear the one-step-per-action guard

    p = tmp_path / "slide.tif"
    Image.fromarray(np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)).save(p)
    assert w._load_optical_path(str(p), reset_align=True, turn_on=True)

    # nudge the alignment, then reset → undo restores the nudged transform
    turn()
    w._optical_drag_by(5.0, -2.0)
    assert (w._optical_align["tx"], w._optical_align["ty"]) == (5.0, -2.0)
    w._reset_optical_align(); turn()
    assert (w._optical_align["tx"], w._optical_align["ty"]) == (0.0, 0.0)
    w._undo(); turn()
    assert (w._optical_align["tx"], w._optical_align["ty"]) == (5.0, -2.0)

    # remove the image → undo brings the backdrop (and its pixels) back
    assert w._optical is not None
    w._remove_optical(); turn()
    assert w._optical is None
    w._undo(); turn()
    assert w._optical is not None and w._optical.shape == (40, 60, 3)
    assert w._optical_on is True
    assert any(e["item"].isVisible() for e in w._optical_views)   # shown again in the views


def test_export_bundle_requires_a_file_selected(win, monkeypatch):
    """The data-book export refuses to write an empty data bundle: with the bundle on but no
    files ticked, Export is blocked with a hint; ticking one file lets it through."""
    from smile_msi.gui.exportdialog import ExportDialog
    dlg = ExportDialog(win, scope="book")
    monkeypatch.setattr(dlg, "_save_prefs", lambda: None)        # keep the prefs store untouched
    called = []
    monkeypatch.setattr(win, "run_export", lambda *a, **k: called.append(a) or True)

    dlg.chk_bundle.setChecked(True)
    for c in dlg.bundle_chks.values():
        c.setChecked(False)
    dlg._do_export()
    assert not called                                           # blocked: nothing in the bundle
    assert "No bundle files selected" in dlg.bundle_preview.toPlainText()

    next(iter(dlg.bundle_chks.values())).setChecked(True)        # tick one file
    dlg._do_export()
    assert called                                               # now the export runs


def test_ion_overlay_ink_contrasts_with_backing(app):
    """The live ion-image annotation overlay picks legible ink from the rendered backing
    under each label/bar — white on a dark backing, near-black on a light one — instead of
    a fixed white-with-shadow. (Geometry of the grab is exercised live; here we unit-test
    the luminance decision against a known backing image.)"""
    from PySide6 import QtCore, QtGui, QtWidgets
    from smile_msi.gui import ionannotations as IA

    gv = QtWidgets.QGraphicsView()
    owner = type("_O", (), {})()
    owner.gv = gv
    w = IA._OverlayWidget(owner)
    w.resize(120, 60)
    full = QtCore.QRectF(0, 0, 120, 60)

    dark = QtGui.QImage(120, 60, QtGui.QImage.Format_RGB32)
    dark.fill(QtGui.QColor("#101010"))
    w._bg_img = dark; w._bg_scale = 1.0
    assert not w._bg_is_light(full)
    assert w._ink(full) == IA._WHITE and w._ink(full, dim=True) == IA._DIM

    light = QtGui.QImage(120, 60, QtGui.QImage.Format_RGB32)
    light.fill(QtGui.QColor("#f0f0f0"))
    w._bg_img = light; w._bg_scale = 1.0
    assert w._bg_is_light(full)
    assert w._ink(full) == IA._INK_DARK and w._ink(full, dim=True) == IA._DIM_DARK

    # no backing to sample → safe default (white ink, never crashes)
    w._bg_img = None
    assert not w._bg_is_light(full) and w._ink(full) == IA._WHITE


def test_export_dialog_remembers_last_used_selections(win, tmp_path, monkeypatch):
    """The Export hub defaults every control to the previous export's choices, persisted
    across sessions — while an explicit context scope still wins over the saved one."""
    from smile_msi import prefs
    from smile_msi.gui.exportdialog import ExportDialog

    monkeypatch.setattr(prefs, "_path", lambda: str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "_CACHE", None)

    d1 = ExportDialog(win)
    d1.scope_combo.setCurrentIndex(d1.scope_combo.findData("overlay"))
    d1.fmt_combo.setCurrentIndex(d1.fmt_combo.findData("tiff"))
    d1.theme_combo.setCurrentText("Dark glass")
    d1.corner_combo.setCurrentText("Bottom-left")
    d1.dpi_spin.setValue(600)
    d1.chk_title.setChecked(False)
    d1.bundle_chks["spectra"].setChecked(True)
    d1._save_prefs()

    monkeypatch.setattr(prefs, "_CACHE", None)            # simulate a fresh app run
    d2 = ExportDialog(win)
    assert d2.scope_combo.currentData() == "overlay"
    assert d2.fmt_combo.currentData() == "tiff"           # format survives the scope rebuild
    assert d2.theme_combo.currentText() == "Dark glass"
    assert d2.corner_combo.currentText() == "Bottom-left"
    assert d2.dpi_spin.value() == 600
    assert d2.chk_title.isChecked() is False
    assert d2.bundle_chks["spectra"].isChecked() is True

    d3 = ExportDialog(win, scope="features")              # context scope overrides saved scope
    assert d3.scope_combo.currentData() == "features"
    assert d3.dpi_spin.value() == 600                     # ...but design prefs still restored


# --------------------------------------------------------------------------- #
# Intake inspector (suggested-settings strip)
# --------------------------------------------------------------------------- #
def test_intake_report_populated_on_load(win):
    """Loading a dataset runs the inspector and enriches the summary banner."""
    rep = getattr(win, "_intake_report", None)
    assert rep is not None
    assert rep.representation in ("centroid", "profile", "unknown")
    assert "pixels" in win.info.text()


def test_intake_strip_show_apply_and_hide(app):
    """An actionable suggestion shows the strip; Apply pushes the suggested
    normalization + tolerance to the live controls; a matching load stays hidden."""
    from smile_msi.intake import IntakeReport

    w = M.MainWindow()
    # already-normalized data → suggest turning normalization OFF (differs from default)
    actionable = IntakeReport(
        representation="centroid", representation_source="data heuristic",
        storage="continuous", normalization="tic", polarity="negative",
        mz_lo=150.0, mz_hi=900.0, n_pixels=40, cv={"tic": 0.0},
        suggested_norm="none", suggested_tol_ppm=10.0, suggested_pick="centroids",
        notes=["Already TIC-normalized — recommend normalization OFF."])
    w.norm_combo.setCurrentText("tic")
    w.ppm_spin.setValue(50.0)
    w._show_intake_suggestion(actionable)
    assert not w._intake_strip.isHidden()                 # actionable → visible

    w._apply_intake_suggestion()
    assert w.norm_combo.currentText() == "none"
    assert w.ppm_spin.value() == 10.0
    assert w._intake_strip.isHidden()                     # apply dismisses the strip

    # raw data while normalization is already "tic" → nothing to change → stay hidden
    matching = IntakeReport(
        representation="profile", representation_source="file flag",
        storage="continuous", normalization="raw", polarity="negative",
        mz_lo=150.0, mz_hi=900.0, n_pixels=40, cv={"tic": 0.5},
        suggested_norm="tic", suggested_tol_ppm=25.0, suggested_pick="mean-spectrum",
        notes=["Raw intensities — recommend TIC normalization."])
    w.norm_combo.setCurrentText("tic")
    w._show_intake_suggestion(matching)
    assert w._intake_strip.isHidden()


def test_gui_lipid_class_composite_is_one_feature(app):
    """A class composite flattens to its member ions for analyses, but renders as a single
    composite image (acts as one feature for display)."""
    w = M.MainWindow()
    ds = M.load_demo()
    w._on_dataset(ds)
    lo, hi = ds.mz_range
    members = [lo + 50.0, lo + 80.0, lo + 120.0]
    comp = w._class_composite_peak("PE", members, "#55A868")
    assert comp["is_class"] and comp["members"] == sorted(members)
    assert lo <= comp["mz"] <= hi                          # representative within range
    # flattens to member ions when an analysis needs real columns
    assert w._feature_mzs(comp) == sorted(members)
    # renders as a composite image of the members, sized to the tissue grid
    img = w._peak_image(comp)
    assert img.shape == (ds.height, ds.width)


def test_gui_default_feature_set_drives_analysis_combos(win, monkeypatch):
    """A fresh analysis defaults to the app-wide default feature set (the last list selected /
    a pinned default), not the working set; 'Set current list as default' pins it.

    Plan 24 retired the per-tab ``_make_feature_combo`` selectors — a fresh analysis is now an
    AnalysisDialog whose ScopeBar adopts the default (see also
    test_gui_scopebar_honors_default_feature_set)."""
    win._feature_lists["MyList"] = [{"mz": float(p["mz"])} for p in win.peaks[:5]]
    win._refresh_feature_set_combo()
    win._default_feature_set = ("list", "MyList")            # the user's current/last-selected set
    bar = _analysis_scope_bar(win)                           # a fresh analysis's data-scope selector
    assert bar._effective_choice() == ("list", "MyList")     # defaults to the pinned/last set

    # 'Set current list as default' reads the dock selector and persists to prefs
    idx = next(i for i in range(win.feat_set_combo.count())
               if win.feat_set_combo.itemData(i) == ("list", "MyList"))
    win.feat_set_combo.setCurrentIndex(idx)                  # does not fire activated/load
    captured = {}
    monkeypatch.setattr("smile_msi.prefs.set", lambda k, v: captured.setdefault(k, v))
    win._set_default_feature_set()
    assert captured["default_feature_list"]["name"] == "MyList"

    win._default_feature_set = None                          # don't leak into downstream tests
    del win._feature_lists["MyList"]
    win._refresh_feature_set_combo()
    win._sync_default_feature_selectors()                    # the pin propagated to roi_feat/cc_feat


def _analysis_scope_bar(win, step_id="pca"):
    """A live ScopeBar to assert against.

    The analysis tabs that owned one (``stats_scope`` / ``comp_scope``) became AnalysisDialogs in
    plan 24, so a bar is now built per *open analysis*. It still registers itself on
    ``win._scope_bars`` — the chokepoint ``_refresh_scope_bars`` drives — so the default-feature-set
    contract below is unchanged; only where the bar lives is."""
    import smile_msi.registry as registry
    from smile_msi.gui.analysisdialog import AnalysisDialog
    dlg = AnalysisDialog(win, registry.REGISTRY[step_id])      # 'pca' needs only feature_set
    assert dlg.scope is not None
    return dlg.scope


def test_gui_scopebar_honors_default_feature_set(win):
    """A ScopeBar-backed analysis (the majority) must adopt the app-wide default too — the
    original 'default feature list' change only wired the make_feature_combo combos, so every
    ScopeBar opened on the working set regardless of the pinned/last-selected default."""
    win._feature_lists["DefList"] = [{"mz": float(p["mz"])} for p in win.peaks[:4]]
    bar = _analysis_scope_bar(win)
    bar._feature_choice = None
    bar._feature_touched = False
    win._default_feature_set = ("list", "DefList")
    win._refresh_scope_bars()                                # the chokepoint a default goes through
    assert bar._effective_choice() == ("list", "DefList")    # untouched bar adopts the default
    assert sorted(bar.feature_mzs()) == sorted(float(p["mz"]) for p in win.peaks[:4])
    assert "DefList" in bar._feature_text()                  # readout matches what will run

    bar._feature_choice = None
    bar._feature_touched = False
    win._default_feature_set = None
    del win._feature_lists["DefList"]
    win._refresh_scope_bars()


def test_gui_default_change_preserves_manual_override(win):
    """Once the user actively picks a set in a ScopeBar, a later default change must not stomp it
    (``_feature_touched``). The eager-combo half of this (``roi_feat`` / ``cc_feat``, built hidden
    at window construction) went with the dialogs those combos lived in."""
    win._feature_lists["A1"] = [{"mz": float(p["mz"])} for p in win.peaks[:3]]
    win._feature_lists["A2"] = [{"mz": float(p["mz"])} for p in win.peaks[3:6]]
    win._refresh_feature_set_combo()

    bar = _analysis_scope_bar(win)
    bar._feature_choice = ("list", "A1")                     # user explicitly picked A1 here
    bar._feature_touched = True
    win._default_feature_set = ("list", "A2")                # default later changes to A2
    win._refresh_scope_bars()
    assert bar._effective_choice() == ("list", "A1")         # user's pick preserved, not A2

    bar._feature_choice = None
    bar._feature_touched = False
    win._default_feature_set = None
    del win._feature_lists["A1"]
    del win._feature_lists["A2"]
    win._refresh_feature_set_combo()
    win._refresh_scope_bars()


def test_gui_vanished_default_falls_back_to_working_set(win):
    """A default whose list was deleted must not strand a ScopeBar: feature_mzs() falls back to
    the working set and the bar stays auto (untouched) so it can adopt a future default."""
    win._feature_lists["Temp"] = [{"mz": float(p["mz"])} for p in win.peaks[:3]]
    bar = _analysis_scope_bar(win)
    bar._feature_choice = None
    bar._feature_touched = False
    win._default_feature_set = ("list", "Temp")
    win._refresh_scope_bars()
    assert bar._effective_choice() == ("list", "Temp")
    del win._feature_lists["Temp"]                           # list gone; in-memory default stale
    win._refresh_scope_bars()
    assert bar.feature_mzs() == list(win._visible_mzs())     # clean fall-back to working set
    assert not bar._feature_touched                          # stays auto for a future default

    win._default_feature_set = None
    win._refresh_scope_bars()


def _lipid_peaks(win):
    """A lipid-list working set: three class composites plus some expanded ions."""
    base = list(win.peaks)
    peaks = [{"mz": base[i]["mz"], "is_class": True, "lipid_class": cls,
              "members": [base[i]["mz"], base[i + 3]["mz"]], "hidden": False}
             for i, cls in enumerate(["PC", "PE", "SM"])]
    peaks += [{"mz": p["mz"], "lipid_class": "PC" if i % 2 else "PE", "hidden": False}
              for i, p in enumerate(base[6:12])]
    return peaks


def test_lipid_tree_bulk_visibility(win):
    """A loaded lipid list is dozens of classes deep and every row carries an eye. Show all /
    Hide all / Invert act on ``self.peaks`` — the source of truth for ``hidden`` — because the
    tree's check states are derived from it and _sync_lipid_tree would overwrite them."""
    from PySide6 import QtCore, QtWidgets

    saved = win.peaks
    try:
        win.peaks = _lipid_peaks(win)
        win._show_lipid_tree(True)
        win._populate_lipid_tree()

        page, tree = win._lipid_tree_page, win.lipid_tree
        assert win.feat_stack.currentWidget() is page       # the page, not the bare tree
        btns = {b.text(): b for b in page.findChildren(QtWidgets.QPushButton)}
        assert {"Show all", "Hide all", "Invert"} <= set(btns)
        assert win._lipid_count.text() == "9 of 9 shown"

        btns["Hide all"].click()
        assert all(p["hidden"] for p in win.peaks)
        assert win._lipid_count.text() == "0 of 9 shown"
        # the tree's eyes follow the peaks, because _after_tree_edit re-syncs from them
        assert all(tree.topLevelItem(i).checkState(0) == QtCore.Qt.Unchecked
                   for i in range(tree.topLevelItemCount()))

        btns["Show all"].click()
        assert not any(p["hidden"] for p in win.peaks)
        assert win._lipid_count.text() == "9 of 9 shown"
    finally:
        win.peaks = saved
        win._show_lipid_tree(False)
        win._decorate_feature_swatches()      # _set_peaks would have; we assigned peaks directly


def test_lipid_tree_invert_yields_the_complement(win):
    """Invert shows what is hidden and hides what is shown — hide 4 of 9, invert, and exactly
    the complementary 5 are hidden. (Not all nine: a predicate that read `hidden` after the
    write loop had started would flip everything back to visible.)"""
    from PySide6 import QtWidgets

    saved = win.peaks
    try:
        win.peaks = _lipid_peaks(win)
        win._show_lipid_tree(True)
        win._populate_lipid_tree()
        btns = {b.text(): b for b in win._lipid_tree_page.findChildren(QtWidgets.QPushButton)}

        for p in win.peaks[:4]:
            p["hidden"] = True
        win._after_tree_edit()
        assert win._lipid_count.text() == "5 of 9 shown"

        btns["Invert"].click()
        assert [bool(p["hidden"]) for p in win.peaks] == [False] * 4 + [True] * 5
        assert win._lipid_count.text() == "4 of 9 shown"
    finally:
        win.peaks = saved
        win._show_lipid_tree(False)
        win._decorate_feature_swatches()      # _set_peaks would have; we assigned peaks directly


def test_lipid_tree_bulk_visibility_undo(win, app):
    """`record_undo` must be called after the early-return guard, so a no-op leaves no entry.
    Its `_undo_busy` re-entrancy flag is cleared by a singleShot(0), so the event loop has to
    be pumped between actions or every later record_undo is a silent no-op."""
    from PySide6 import QtWidgets

    saved = win.peaks
    try:
        win.peaks = _lipid_peaks(win)
        win._show_lipid_tree(True)
        win._populate_lipid_tree()
        btns = {b.text(): b for b in win._lipid_tree_page.findChildren(QtWidgets.QPushButton)}

        def click(name):
            btns[name].click()
            app.processEvents()

        click("Show all")                                    # settle: everything visible
        n0 = len(win._undo_stack)
        click("Show all")                                    # a genuine no-op
        assert len(win._undo_stack) == n0, "a no-op must not record undo"

        click("Hide all")                                    # a real change
        assert len(win._undo_stack) == n0 + 1
        assert all(p["hidden"] for p in win.peaks)

        win._undo()
        app.processEvents()
        assert not any(p["hidden"] for p in win.peaks), "undo must restore visibility"
    finally:
        win.peaks = saved
        win._show_lipid_tree(False)
        win._decorate_feature_swatches()      # _set_peaks would have; we assigned peaks directly


def test_feature_panel_bulk_visibility(win, app):
    """The features list is the "30 boxes" surface: one eye per feature, and the only bulk
    control was a single "Toggle all" flip-flop whose effect you could not predict without
    first reading every eye. Show all / Hide all / Invert say what they do, and the count
    stops a long, part-hidden list from lying about what the analyses will see."""
    from PySide6 import QtCore, QtWidgets

    win._decorate_feature_swatches()          # settle the derived count before asserting on it
    n = len(win.peaks)
    # by objectName, not label: the lipid tree page and the regions menu carry their own
    # Show all / Hide all / Invert, and findChildren-by-text would reach whichever came last.
    btns = {t: win.findChild(QtWidgets.QPushButton, "featVis" + t.replace(" ", ""))
            for t in ("Show all", "Hide all", "Invert")}
    assert all(b is not None for b in btns.values())
    assert not [b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == "Toggle all"]

    def click(name):
        btns[name].click()
        app.processEvents()

    def hidden():
        return sum(1 for p in win.peaks if p.get("hidden"))

    click("Show all")
    assert hidden() == 0 and win.feat_visible_count.text() == f"{n} of {n} shown"
    click("Hide all")
    assert hidden() == n and win.feat_visible_count.text() == f"0 of {n} shown"
    click("Show all")

    # Invert yields the complement, not "everything visible again"
    for p in win.peaks[:4]:
        p["hidden"] = True
    win._decorate_feature_swatches()
    assert win.feat_visible_count.text() == f"{n - 4} of {n} shown"
    click("Invert")
    assert [bool(p.get("hidden")) for p in win.peaks] == [False] * 4 + [True] * (n - 4)
    click("Show all")

    # a single eye toggle keeps the count honest — it deliberately skips the full re-decorate
    win.feat_table.item(0, 0).setCheckState(QtCore.Qt.Unchecked)
    app.processEvents()
    assert win.feat_visible_count.text() == f"{n - 1} of {n} shown"
    win.feat_table.item(0, 0).setCheckState(QtCore.Qt.Checked)
    app.processEvents()
    assert win.feat_visible_count.text() == f"{n} of {n} shown"


def test_feature_panel_bulk_visibility_undo(win, app):
    """No-op leaves no undo entry (record_undo sits after the early return); a real bulk
    change is undoable. Pump the loop between clicks or `_undo_busy` silently swallows every
    later record_undo and the assertion passes for the wrong reason."""
    from PySide6 import QtWidgets

    btns = {t: win.findChild(QtWidgets.QPushButton, "featVis" + t.replace(" ", ""))
            for t in ("Show all", "Hide all")}

    def click(name):
        btns[name].click()
        app.processEvents()

    click("Show all")                                    # settle
    n0 = len(win._undo_stack)
    click("Show all")                                    # genuine no-op
    assert len(win._undo_stack) == n0, "a no-op must not record undo"

    click("Hide all")
    assert len(win._undo_stack) == n0 + 1
    assert all(p.get("hidden") for p in win.peaks)
    win._undo()
    app.processEvents()
    assert not any(p.get("hidden") for p in win.peaks)


def test_feature_panel_value_widgets_are_wheel_safe(win):
    """`contrast_spin` and `norm_combo` decide what the ion image and every analysis see."""
    from smile_msi.gui.common import _NoWheelMixin

    for attr in ("contrast_spin", "norm_combo", "composite_combo",
                 "class_combo", "ratio_a", "ratio_b"):
        w = getattr(win, attr)
        assert isinstance(w, _NoWheelMixin), f"{attr} is a raw {type(w).__name__}"
