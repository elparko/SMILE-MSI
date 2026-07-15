"""Cohort GUI test — drives the Samples roster + Cohort comparison tab headless.

Writes throwaway managed sessions under a tmp ``SMILE_MSI_HOME``, builds a cohort with
two groups, and runs the between-group comparison through the real tab wiring.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from smile_msi import cohort, library, session  # noqa: E402
from smile_msi.gui import main as M  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_session(source, peaks, fp):
    sess = session.build_session(source=source, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=100, dataset_fingerprint=fp)
    return session.save_session(session.managed_path(source, fp), sess)


def _write_session_scopes(source, peaks, scopes, fp):
    sess = session.build_session(source=source, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=100, dataset_fingerprint=fp, feature_scopes=scopes)
    return session.save_session(session.managed_path(source, fp), sess)


def _peak(mz, rel):
    return {"mz": float(mz), "intensity": rel * 100, "snr": 10.0, "rel_intensity": float(rel)}


def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
    """Run a backgrounded worker synchronously so a test can assert on the result — the
    cohort comparison/embedding now dispatch off-thread via win._run (no GUI freeze)."""
    if want_progress:
        k["progress"] = lambda *_: None
    res = fn(*a, **k)
    if on_done:
        on_done(res)


def test_cohort_tab_group_comparison(app, monkeypatch, tmp_path):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    t_up, t_flat = 750.0, 600.0
    win = M.MainWindow()
    win._run = _sync_run                              # comparison now runs off-thread
    win.cohort = cohort.Cohort(name="TestCohort")
    for i, rel in enumerate([0.10, 0.12, 0.11]):
        p = _write_session(f"/d/ctrl{i}.imzML", [_peak(t_up, rel), _peak(t_flat, 0.5)], fp=f"c{i}")
        win.cohort.add(cohort.ref_from_session(p, group="control"))
    for i, rel in enumerate([0.80, 0.78, 0.83]):
        p = _write_session(f"/d/syn{i}.imzML", [_peak(t_up, rel), _peak(t_flat, 0.5)], fp=f"s{i}")
        win.cohort.add(cohort.ref_from_session(p, group="synkinetic"))

    win._refresh_sample_tree()                       # populates the group combos
    # roster shows two group buckets, each with its samples
    assert win.sample_tree.topLevelItemCount() == 2
    assert {win.cohort_ga.itemText(i) for i in range(win.cohort_ga.count())} == \
        {"control", "synkinetic"}

    win.cohort_feat_combo.setCurrentIndex(0)         # consensus features
    win.cohort_prev_spin.setValue(0.0)
    win.cohort_tol_spin.setValue(50.0)
    win.cohort_ga.setCurrentText("control")
    win.cohort_gb.setCurrentText("synkinetic")
    win._cohort_run()

    # the comparison ran: table populated, volcano + heatmap rendered
    assert win.cohort_table.rowCount() >= 2
    assert win.cohort_heat.image is not None
    # the colored grid must fill the heatmap's x-range (feature axis), not be smushed into a
    # sliver near x=0 — guards the row-major axis order against a col-major regression.
    xr = win.cohort_heat_plot.viewRange()[0]
    assert abs(win.cohort_heat.boundingRect().width() - (xr[1] - xr[0])) < 1e-6
    assert len(win.cohort_volcano.listDataItems()) >= 1
    res = win._cohort_res
    up = res[abs(res["mz"] - t_up) < 0.01].iloc[0]
    assert up["log2_fc"] > 1.0                        # discriminating ion up in synkinetic
    assert up["q_value"] <= 0.2

    # CSV export writes a file
    out = tmp_path / "cmp.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")))
    win._cohort_export_csv()
    assert out.exists()

    # export buttons armed after the run
    assert win.b_cohort_list.isEnabled()
    assert win.b_cohort_volcano_img.isEnabled()

    # feature list built from the hits. Drive the dialog deterministically: keep ALL tested
    # ions (no q filter) so the saved list doesn't hinge on this fixture's exact q-values.
    def _accept(dlg):
        dlg.findChild(QtWidgets.QComboBox).setCurrentIndex(3)   # first combo = "Keep" target → All
        for cb in dlg.findChildren(QtWidgets.QCheckBox):
            cb.setChecked(False)                                # drop the q ≤ 0.05 filter
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(QtWidgets.QDialog, "exec", _accept)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("batch markers", True)))
    n_before = len(win._feature_lists)
    win._cohort_build_list_dialog()
    assert len(win._feature_lists) == n_before + 1
    saved = win._feature_lists["batch markers"]
    assert len(saved) >= 1


def test_cohort_tab_paired_comparison(app, monkeypatch, tmp_path):
    """Paired design end-to-end through the view: tick Paired, pick the 'subject' metadata key,
    and a within-donor shift the unpaired test can't separate becomes significant."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    t_up, t_flat = 750.0, 600.0
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="PairedCohort")
    ctrl_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    deltas = [0.05, 0.08, 0.11, 0.09, 0.07, 0.06]     # distinct positive within-pair shifts
    for i, (base, d) in enumerate(zip(ctrl_levels, deltas)):
        pc = _write_session(f"/d/ctrl{i}.imzML", [_peak(t_up, base), _peak(t_flat, 0.5)], fp=f"c{i}")
        rc = cohort.ref_from_session(pc, group="control"); rc.meta["subject"] = f"donor{i}"
        win.cohort.add(rc)
        pt = _write_session(f"/d/trt{i}.imzML", [_peak(t_up, base + d), _peak(t_flat, 0.5)], fp=f"t{i}")
        rt = cohort.ref_from_session(pt, group="treated"); rt.meta["subject"] = f"donor{i}"
        win.cohort.add(rt)

    win._refresh_sample_tree()
    # the pair-by selector picked up the 'subject' metadata key and the Paired control is live
    assert "subject" in {win.cohort_pair_combo.itemText(i)
                         for i in range(win.cohort_pair_combo.count())}
    assert win.cohort_paired_check.isEnabled()

    win.cohort_feat_combo.setCurrentIndex(0)
    win.cohort_prev_spin.setValue(0.0)
    win.cohort_tol_spin.setValue(50.0)
    win.cohort_ga.setCurrentText("control")
    win.cohort_gb.setCurrentText("treated")

    # unpaired first: the overlapping ranges don't separate
    win._cohort_run()
    up_unpaired = win._cohort_res[abs(win._cohort_res["mz"] - t_up) < 0.01].iloc[0]
    assert up_unpaired["q_value"] > 0.05

    # now paired on the donor
    win.cohort_paired_check.setChecked(True)
    win.cohort_pair_combo.setCurrentText("subject")
    win._cohort_run()
    res = win._cohort_res
    assert res.attrs["paired"] is True and res.attrs["pair_by"] == "subject"
    assert res.attrs["test"].startswith("Wilcoxon")
    up = res[abs(res["mz"] - t_up) < 0.01].iloc[0]
    assert up["log2_fc"] > 0 and up["q_value"] <= 0.05
    assert "paired" in win.cohort_info.text().lower()
    assert len(win.cohort_volcano.listDataItems()) >= 1


def _sample_rows(tree):
    """Flatten the roster tree to its sample rows (skipping group headers)."""
    out = []
    for i in range(tree.topLevelItemCount()):
        head = tree.topLevelItem(i)
        for j in range(head.childCount()):
            out.append(head.child(j))
    return out


def test_roster_filter_select_and_autogroup(app, monkeypatch, tmp_path):
    """The Samples roster filters live, 'select matching' ticks the visible set, and
    Auto-group derives groups from the folder in one pass — the cohort-scale flow."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="RosterCohort")
    for folder, names in (("ctrl", ["a", "b"]), ("treated", ["c"])):
        for nm in names:
            p = _write_session(f"/data/{folder}/{nm}.imzML", [_peak(700.0, 0.5)], fp=f"{folder}{nm}")
            ref = cohort.ref_from_session(p)
            ref.source = f"/data/{folder}/{nm}.imzML"
            win.cohort.add(ref)
    win._refresh_sample_tree()
    assert len(_sample_rows(win.sample_tree)) == 3

    # filter narrows the roster to the matching rows
    win.sample_filter.setText("a.imzML")
    assert len(_sample_rows(win.sample_tree)) == 1
    win.sample_filter.setText("")

    # auto-group by folder buckets all three in one click
    win._cohort_group_by_folder()
    assert win.cohort.by_group().keys() == {"ctrl", "treated"}
    assert win.sample_tree.topLevelItemCount() == 2

    # filter to one group, select matching → those samples are selected for a group change
    win.sample_filter.setText("treated")
    win._select_matching_samples()
    assert len(win.sample_tree.selectedItems()) == 1


def test_roster_import_metadata(app, monkeypatch, tmp_path):
    """Importing a metadata CSV fills meta + group and matches by the imzML basename."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="MetaCohort")
    for nm in ("slide1", "slide2"):
        p = _write_session(f"/data/{nm}.imzML", [_peak(700.0, 0.5)], fp=nm)
        ref = cohort.ref_from_session(p)
        ref.source = f"/data/{nm}.imzML"
        win.cohort.add(ref)

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("sample,group,sex\nslide1,ctrl,M\nslide2,treated,F\n")
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_open_file_name",
                        lambda *a, **k: (str(csv_path), "CSV"))
    # accept the column-mapping dialog with its auto-guessed key/group columns
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: QtWidgets.QDialog.Accepted)
    win._cohort_import_metadata()

    assert [s.group for s in win.cohort.samples] == ["ctrl", "treated"]
    assert win.cohort.find("/data/slide1.imzML").meta == {"sex": "M"}
    assert win.cohort.meta_keys() == ["sex"]


def test_roster_table_columns_group_and_numeric_sort(app, monkeypatch, tmp_path):
    """Phase 1.1: the samples roster is a multi-column table (Name/Group/Region/#px + meta
    columns), the group structure is preserved, and clicking a header sorts rows WITHIN a
    group numerically for #px (not lexically) while leaving the groups in place."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="TableCohort")
    for nm, npx in (("a", 300), ("b", 100), ("c", 200)):
        p = _write_session(f"/d/{nm}.imzML", [_peak(700.0, 0.5)], fp=nm)
        ref = cohort.ref_from_session(p)
        ref.source = f"/d/{nm}.imzML"
        ref.n_pixels = npx
        ref.meta = {"sex": "M"}
        win.cohort.add(ref)
    win.cohort.set_group(win.cohort.samples[0].key(), "g1")    # a → g1; b, c → Ungrouped
    win._refresh_sample_tree()
    tree = win.sample_tree

    headers = [tree.headerItem().text(c) for c in range(tree.columnCount())]
    assert headers[:4] == ["Name", "Group", "Region", "#px"]
    assert "sex" in headers                                    # metadata became a real column
    assert tree.topLevelItemCount() == 2                       # grouping preserved (contract)

    win._sort_sample_tree(3, QtCore.Qt.AscendingOrder)         # sort by #px
    ung = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if tree.topLevelItem(i).text(0).startswith("Ungrouped"))
    pxs = [int(ung.child(j).text(3).replace(",", "")) for j in range(ung.childCount())]
    assert pxs == [100, 200]                                   # numeric order, not lexical "100","200"
    assert tree.topLevelItemCount() == 2                       # sort did NOT reorder/flatten groups


def test_cohort_feature_list_picker_and_group_guard(app, monkeypatch, tmp_path):
    """The cohort comparison can run on a saved ★ list (not just Consensus/active), and a
    1-vs-N group choice is refused up front rather than producing an all-NaN 'clean negative'."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    t_list, t_other = 885.5, 700.0
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="PickerCohort")
    for i, (rel, grp) in enumerate([(0.20, "control"), (0.25, "control"),
                                    (0.80, "synkinetic"), (0.85, "synkinetic")]):
        p = _write_session(f"/d/s{i}.imzML", [_peak(t_list, rel), _peak(t_other, 0.5)], fp=f"p{i}")
        win.cohort.add(cohort.ref_from_session(p, group=grp))
    win._refresh_sample_tree()

    # a saved ★ list (only t_list) shows up in the cohort feature combo after a refresh
    win._feature_lists = {"MyList": [{"mz": t_list, "lipid": "", "note": ""}]}
    win._refresh_cohort_feature_combos()
    fc = win.cohort_feat_combo
    assert "Consensus" in fc.itemText(0)
    idx = next(i for i in range(fc.count()) if fc.itemData(i) == ("list", "MyList"))
    fc.setCurrentIndex(idx)
    fc.setProperty("user_touched", True)                 # simulate an explicit pick
    win.cohort_tol_spin.setValue(50.0)
    win.cohort_ga.setCurrentText("control"); win.cohort_gb.setCurrentText("synkinetic")
    win._cohort_run()
    res = win._cohort_res
    assert {round(float(m), 1) for m in res["mz"]} == {round(t_list, 1)}   # ran on the ★ list only

    # group-size guard: 1 control vs 1 synkinetic is refused (no comparison produced)
    win.cohort = cohort.Cohort(name="Tiny")
    pa = _write_session("/d/a.imzML", [_peak(t_list, 0.2)], fp="a")
    pb = _write_session("/d/b.imzML", [_peak(t_list, 0.8)], fp="b")
    win.cohort.add(cohort.ref_from_session(pa, group="control"))
    win.cohort.add(cohort.ref_from_session(pb, group="synkinetic"))
    win._refresh_sample_tree()
    win._cohort_res = None
    win.cohort_ga.setCurrentText("control"); win.cohort_gb.setCurrentText("synkinetic")
    win._cohort_run()
    assert win._cohort_res is None                         # refused before running a 1-vs-1 test


def test_cohort_roster_grouping_and_removal(app, monkeypatch, tmp_path):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="TestCohort2")
    p1 = _write_session("/d/a.imzML", [_peak(700.0, 0.5)], fp="a")
    p2 = _write_session("/d/b.imzML", [_peak(700.0, 0.6)], fp="b")
    r1 = win.cohort.add(cohort.ref_from_session(p1))
    r2 = win.cohort.add(cohort.ref_from_session(p2))
    win._refresh_sample_tree()
    assert win.sample_tree.topLevelItemCount() == 1   # both ungrouped → one bucket

    win._cohort_set_group([r1.key()], "control")
    assert win.cohort.find(r1.key()).group == "control"
    assert win.sample_tree.topLevelItemCount() == 2   # control + ungrouped buckets

    win._cohort_remove([r2.key()])
    assert win.cohort.find(r2.key()) is None
    # the roster persisted to disk under the tmp home
    assert os.path.exists(cohort.cohort_path("TestCohort2"))


def test_cohort_sample_picker_subsets_run_without_removing(app, monkeypatch, tmp_path):
    """The Samples ▾ picker subsets which samples a run uses, leaving the roster intact: an
    unticked sample drops from _cohort_included_refs but stays in the cohort, and re-ticking
    (or All) restores it. The selection is shared across the per-analysis pickers."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="PickerCohort")
    p1 = _write_session("/d/a.imzML", [_peak(700.0, 0.5)], fp="a")
    p2 = _write_session("/d/b.imzML", [_peak(700.0, 0.6)], fp="b")
    p3 = _write_session("/d/c.imzML", [_peak(700.0, 0.7)], fp="c")
    r1 = win.cohort.add(cohort.ref_from_session(p1))
    r2 = win.cohort.add(cohort.ref_from_session(p2))
    r3 = win.cohort.add(cohort.ref_from_session(p3))

    # the window already built the per-analysis pickers (compare + embed tabs), all sharing
    # one selection set — so a fresh roster defaults all-included
    assert len(win._cohort_sample_pickers) >= 2
    assert {r.key() for r in win._cohort_included_refs()} == {r1.key(), r2.key(), r3.key()}

    # untick r2 → excluded from runs, still in the roster
    win._cohort_sample_unchecked.add(r2.key())
    assert {r.key() for r in win._cohort_included_refs()} == {r1.key(), r3.key()}
    assert win.cohort.find(r2.key()) is not None       # roster untouched
    assert len(win.cohort.samples) == 3

    # every picker's popup list reflects the shared unchecked state
    for picker in win._cohort_sample_pickers:
        picker.refresh()
        lw = picker.list_widget
        states = {lw.item(i).data(QtCore.Qt.UserRole): lw.item(i).checkState()
                  for i in range(lw.count())}
        assert states[r2.key()] == QtCore.Qt.Unchecked
        assert states[r1.key()] == QtCore.Qt.Checked

    # None → nothing included; All → everything back (both act on the rows currently shown)
    picker = win._cohort_sample_pickers[0]
    picker.refresh()
    picker.set_all(False)
    assert win._cohort_included_refs() == []
    picker.set_all(True)
    assert len(win._cohort_included_refs()) == 3

    # toggling a row's check state updates the shared set
    picker.refresh()
    lw = picker.list_widget
    item = next(lw.item(i) for i in range(lw.count()) if lw.item(i).data(QtCore.Qt.UserRole) == r1.key())
    item.setCheckState(QtCore.Qt.Unchecked)            # fires CheckPicker._item_changed
    assert {r.key() for r in win._cohort_included_refs()} == {r2.key(), r3.key()}


def test_cohort_sample_picker_filter_sort(app, monkeypatch, tmp_path):
    """The Samples ▾ picker filters by name *or* group and its All/None act only on the rows
    currently shown — so you can bulk-(un)tick a whole group without clicking each sample —
    and sorting reorders the list."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="FilterCohort")
    ra = win.cohort.add(cohort.ref_from_session(_write_session("/d/ctrl_a.imzML", [_peak(700.0, 0.5)], fp="a")))
    rb = win.cohort.add(cohort.ref_from_session(_write_session("/d/ctrl_b.imzML", [_peak(700.0, 0.6)], fp="b")))
    rc = win.cohort.add(cohort.ref_from_session(_write_session("/d/treat_c.imzML", [_peak(700.0, 0.7)], fp="c")))
    win.cohort.set_group(ra.key(), "control")
    win.cohort.set_group(rb.key(), "control")
    win.cohort.set_group(rc.key(), "treated")

    picker = win._cohort_sample_pickers[0]
    picker.refresh()
    lw = picker.list_widget
    assert lw.count() == 3

    # filtering on the GROUP label (not in any sample name) narrows to the two controls
    picker._filter.setText("control")
    assert {lw.item(i).data(QtCore.Qt.UserRole) for i in range(lw.count())} == {ra.key(), rb.key()}

    # None while filtered unticks ONLY the shown (control) rows — the treated sample stays in
    picker.set_all(False)
    assert win._cohort_sample_unchecked == {ra.key(), rb.key()}
    assert {r.key() for r in win._cohort_included_refs()} == {rc.key()}

    # clearing the filter and sorting by name gives an alphabetical list of all three
    picker._filter.setText("")
    picker._sort_combo.setCurrentText("Name")
    labels = [lw.item(i).text() for i in range(lw.count())]
    assert len(labels) == 3 and labels == sorted(labels, key=str.lower)


def test_cohort_tab_region_samples(app, monkeypatch, tmp_path):
    """Regions promoted to cohort samples run through the real comparison tab — two
    regions per slide act as the replicates."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    t = 750.0
    win = M.MainWindow()
    win._run = _sync_run                              # comparison now runs off-thread
    win.cohort = cohort.Cohort(name="RegionCohort")
    p1 = _write_session_scopes("/d/s1.imzML", [_peak(t, 0.1)],
                               {"L": [_peak(t, 0.10)], "R": [_peak(t, 0.12)]}, fp="s1")
    p2 = _write_session_scopes("/d/s2.imzML", [_peak(t, 0.1)],
                               {"L": [_peak(t, 0.80)], "R": [_peak(t, 0.83)]}, fp="s2")
    for sp, grp in [(p1, "control"), (p2, "synkinetic")]:
        for roi in ("L", "R"):
            win.cohort.add(cohort.SampleRef(name=f"{os.path.basename(sp)} · {roi}",
                                            session_path=sp, region=roi, group=grp))
    win._refresh_sample_tree()
    # four region rows across two group buckets, each keyed distinctly (no collision)
    assert win.sample_tree.topLevelItemCount() == 2
    keys = set()
    for i in range(win.sample_tree.topLevelItemCount()):
        head = win.sample_tree.topLevelItem(i)
        for j in range(head.childCount()):
            keys.add(head.child(j).data(0, QtCore.Qt.UserRole))
    assert len(keys) == 4

    win.cohort_feat_combo.setCurrentIndex(0)
    win.cohort_prev_spin.setValue(0.0)
    win.cohort_tol_spin.setValue(50.0)
    win.cohort_norm_combo.setCurrentText("none")      # isolate scope-reading from normalization
    win.cohort_ga.setCurrentText("control")
    win.cohort_gb.setCurrentText("synkinetic")
    win._cohort_run()
    res = win._cohort_res
    up = res[abs(res["mz"] - t) < 0.05].iloc[0]
    assert up["n_A"] == 2 and up["n_B"] == 2          # two regions per group
    assert up["log2_fc"] > 1.0                        # marker up in synkinetic regions


def test_cohort_umap_pooled_embedding(app, monkeypatch, tmp_path):
    """The Cohort UMAP tab pools pixels from real imzML-backed samples on a shared axis
    into ONE embedding, then colours the cloud by group (2 categories) or sample (4)."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()

    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
        if want_progress:
            k["progress"] = lambda *_: None
        res = fn(*a, **k)
        if on_done:
            on_done(res)
    win._run = _sync_run

    win.cohort = cohort.Cohort(name="EmbedCohort")
    for i, grp in enumerate(["control", "control", "synkinetic", "synkinetic"]):
        src = str(tmp_path / f"s{i}.imzML")
        demo.write_synthetic_imzml(src, width=16, height=12, seed=i + 1)
        ds = MSIDataset.from_imzml(src); ds.prime()
        peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
        fp = library.dataset_fingerprint(ds)
        sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                     n_pixels=ds.n_pixels, dataset_fingerprint=fp)
        p = session.save_session(session.managed_path(src, fp), sess)
        win.cohort.add(cohort.ref_from_session(p, group=grp))

    win.cembed_feat_combo.setCurrentIndex(0)             # consensus shared axis
    win.cembed_prev_spin.setValue(0.0)
    win.cembed_tol_spin.setValue(50.0)
    win.cembed_cap_spin.setValue(80)                     # small, fast embedding (< 192 px/slide)
    win.cembed_method_combo.setCurrentText("TSNE")       # deterministic; no umap-learn dep
    win._cembed_run()

    emb = win._cembed_emb
    assert len(emb.sample_names) == 4                    # all four slides pooled
    assert emb.coords.shape[0] == sum(emb.counts.values())
    assert all(c == 80 for c in emb.counts.values())     # per-sample cap honoured
    assert set(emb.group) == {"control", "synkinetic"}
    assert win.b_cembed_png.isEnabled()

    win.cembed_color_combo.setCurrentText("Group")
    win._draw_cembed()
    assert len(win.cembed_plot.listDataItems()) == 2     # one cloud per group
    win.cembed_color_combo.setCurrentText("Sample")      # re-colours to one cloud per sample
    assert len(win.cembed_plot.listDataItems()) == 4


def test_cohort_umap_region_and_sample_means(app, monkeypatch, tmp_path):
    """The Cohort UMAP aggregate units: 'Region means' embeds one point per named region per
    slide, 'Sample means' one point per slide — the memory-flat 'reduce each slide, then
    combine' path. Each streams one cube at a time (cap is irrelevant, so greyed)."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="AggCohort")
    n_slides = 4
    for i, grp in enumerate(["control", "control", "synkinetic", "synkinetic"]):
        src = str(tmp_path / f"a{i}.imzML")
        demo.write_synthetic_imzml(src, width=16, height=12, seed=i + 1)
        ds = MSIDataset.from_imzml(src); ds.prime()
        peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
        fp = library.dataset_fingerprint(ds)
        npx = ds.n_pixels
        half = npx // 2
        regions = [{"name": "top", "color": "#f00", "segments": [],
                    "mask": np.arange(npx) < half},
                   {"name": "bot", "color": "#00f", "segments": [],
                    "mask": np.arange(npx) >= half}]
        sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                     n_pixels=npx, dataset_fingerprint=fp, named_regions=regions)
        p = session.save_session(session.managed_path(src, fp), sess)
        win.cohort.add(cohort.ref_from_session(p, group=grp))

    win.cembed_feat_combo.setCurrentIndex(0)             # consensus shared axis
    win.cembed_prev_spin.setValue(0.0)
    win.cembed_tol_spin.setValue(50.0)
    win.cembed_method_combo.setCurrentText("TSNE")

    # Region means → one point per region per slide (2 regions × 4 slides = 8)
    win.cembed_unit_combo.setCurrentText("Region means")
    assert not win.cembed_cap_spin.isEnabled()           # cap greyed for aggregate units
    win._cembed_run()
    emb = win._cembed_emb
    assert emb.coords.shape[0] == 2 * n_slides
    assert set(emb.region) == {"top", "bot"}             # RegionEmbedding carries region labels
    win.cembed_color_combo.setCurrentText("Region")
    win._draw_cembed()
    assert len(win.cembed_plot.listDataItems()) == 2     # top / bot clouds

    # Studio gets the SAME regions the run pooled over (emb.region), not a reconstruction
    # from the cohort roster — whole-slide samples carry no .region, so that path used to
    # leave the Region channel empty/mismatched for region-mean runs.
    monkeypatch.setattr(win, "open_umap_studio", lambda data: setattr(win, "_studio_data", data))
    win._open_cembed_studio()
    assert set(win._studio_data.categorical["Region"]) == {"top", "bot"}

    # the Regions picker lists every region in the cohort, and deselecting one drops it
    assert set(win._cembed_region_names()) == {"top", "bot"}
    win._cembed_region_unchecked = {"bot"}               # keep only 'top'
    win._cembed_run()
    emb2 = win._cembed_emb
    assert emb2.coords.shape[0] == n_slides               # one 'top' point per slide
    assert set(emb2.region) == {"top"}
    win._cembed_region_unchecked = set()                  # restore for the next unit

    # Sample means → one point per slide
    win.cembed_unit_combo.setCurrentText("Sample means")
    win._cembed_run()
    emb_sm = win._cembed_emb
    assert emb_sm.coords.shape[0] == n_slides

    # the embedding hands cleanly to UMAP Studio carrying the per-sample mean-spectrum matrix
    # (one row per slide, one column per shared target) — the aggregate 'features' the cohort
    # export/colour-by-lipid path retains since feat(cohort) 8942e70; no per-pixel geometry.
    monkeypatch.setattr(win, "open_umap_studio", lambda data: setattr(win, "_studio_data", data))
    win._open_cembed_studio()
    assert win._studio_data.features is not None
    assert win._studio_data.features.shape == (n_slides, len(emb_sm.targets))


def test_cohort_umap_region_means_single_sample(app, monkeypatch, tmp_path):
    """Region means and Pixels are both meaningful within ONE slide — region means compares
    that slide's named regions, and a per-pixel UMAP of one section is the classic lipid-atlas
    figure — so both run on a single-sample cohort (only Sample means, one point per slide,
    still needs ≥2 slides). This is what lets one slide be embedded without padding the cohort."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="SoloCohort")

    src = str(tmp_path / "solo.imzML")
    demo.write_synthetic_imzml(src, width=16, height=12, seed=1)
    ds = MSIDataset.from_imzml(src); ds.prime()
    peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
    fp = library.dataset_fingerprint(ds)
    npx = ds.n_pixels
    half = npx // 2
    regions = [{"name": "top", "color": "#f00", "segments": [], "mask": np.arange(npx) < half},
               {"name": "bot", "color": "#00f", "segments": [], "mask": np.arange(npx) >= half}]
    sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=npx, dataset_fingerprint=fp, named_regions=regions)
    p = session.save_session(session.managed_path(src, fp), sess)
    win.cohort.add(cohort.ref_from_session(p, group="control"))

    win.cembed_feat_combo.setCurrentIndex(0)             # consensus shared axis
    win.cembed_prev_spin.setValue(0.0)
    win.cembed_tol_spin.setValue(50.0)
    win.cembed_method_combo.setCurrentText("TSNE")

    # one sample → Pixels and Region means enable Run (single-slide atlas / region compare);
    # only Sample means (one point per slide) stays disabled
    win.cembed_unit_combo.setCurrentText("Pixels")
    win._cembed_sync_unit()
    assert win.b_cembed_run.isEnabled()
    win.cembed_unit_combo.setCurrentText("Sample means")
    win._cembed_sync_unit()
    assert not win.b_cembed_run.isEnabled()
    win.cembed_unit_combo.setCurrentText("Region means")
    win._cembed_sync_unit()
    assert win.b_cembed_run.isEnabled()

    # and it actually runs on the lone slide: one point per named region
    win._cembed_run()
    emb = win._cembed_emb
    assert emb.coords.shape[0] == 2
    assert set(emb.region) == {"top", "bot"}


def test_fresh_open_registers_in_roster(app, monkeypatch, tmp_path):
    """A freshly opened slide joins the Samples roster (and shows in the tree) immediately,
    before any analysis — registration must not wait for the first auto-save. Regression for
    'samples don't show in the menu when I load them in'."""
    from smile_msi import demo

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="FreshOpen")

    src = str(tmp_path / "fresh.imzML")
    demo.write_synthetic_imzml(src, width=12, height=10, seed=1)

    # the real File ▸ Open / launcher path — no Find peaks, no manual roster add
    win._load_path(src)

    # it must be on the roster and rendered in the tree right after the load completes
    assert len(win.cohort.samples) == 1
    assert os.path.abspath(win.cohort.samples[0].source) == os.path.abspath(src)
    assert len(_sample_rows(win.sample_tree)) == 1


def test_cohort_umap_pixels_confined_to_picked_regions(app, monkeypatch, tmp_path):
    """The Pixels unit can be confined to chosen regions via the Regions ▾ picker: with all
    regions ticked it pools the whole slide, and unticking one restricts the per-pixel cloud to
    the regions kept (background + unticked regions dropped) while geometry stays for back-map."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="PixRegionCohort")

    src = str(tmp_path / "pixreg.imzML")
    demo.write_synthetic_imzml(src, width=16, height=12, seed=2)
    ds = MSIDataset.from_imzml(src); ds.prime()
    peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
    fp = library.dataset_fingerprint(ds)
    npx = ds.n_pixels
    half = npx // 2
    regions = [{"name": "top", "color": "#f00", "segments": [], "mask": np.arange(npx) < half},
               {"name": "bot", "color": "#00f", "segments": [], "mask": np.arange(npx) >= half}]
    sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=npx, dataset_fingerprint=fp, named_regions=regions)
    p = session.save_session(session.managed_path(src, fp), sess)
    win.cohort.add(cohort.ref_from_session(p, group="control"))

    win.cembed_feat_combo.setCurrentIndex(0)
    win.cembed_prev_spin.setValue(0.0)
    win.cembed_tol_spin.setValue(50.0)
    win.cembed_method_combo.setCurrentText("TSNE")        # deterministic; no umap-learn dep
    win.cembed_unit_combo.setCurrentText("Pixels")
    win._cembed_sync_unit()
    assert win.cembed_regions_btn.isEnabled()             # Regions picker now usable for Pixels
    win.cembed_cap_spin.setValue(20000)                   # no capping → counts are real pixel counts

    # all ticked → the whole-slide pool (unchanged default behaviour)
    win._cembed_region_unchecked = set()
    win._cembed_run()
    n_whole = win._cembed_emb.coords.shape[0]
    assert n_whole > 0 and win._cembed_emb.px_row is not None    # geometry kept for back-map

    # untick 'bot' → the cloud is confined to 'top' pixels only (fewer points than the slide)
    win._cembed_region_unchecked = {"bot"}
    win._cembed_run()
    emb_top = win._cembed_emb
    assert 0 < emb_top.coords.shape[0] < n_whole
    assert emb_top.px_row is not None                     # still back-mappable to the tissue


def test_cohort_umap_single_slide_pixels_cluster_and_map(app, monkeypatch, tmp_path):
    """The per-pixel 'molecular histology' path on a SINGLE slide: Pixels runs on one sample,
    Colour by Cluster carves the embedding into phenotypes, the dense cloud renders as a
    density composite (an ImageItem + one labelled centroid per cluster), and 'Cluster map'
    paints those clusters back onto the slide's tissue (one panel per sample)."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="AtlasCohort")

    src = str(tmp_path / "atlas.imzML")
    demo.write_synthetic_imzml(src, width=18, height=14, seed=4)
    ds = MSIDataset.from_imzml(src); ds.prime()
    peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
    fp = library.dataset_fingerprint(ds)
    sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=ds.n_pixels, dataset_fingerprint=fp)
    p = session.save_session(session.managed_path(src, fp), sess)
    win.cohort.add(cohort.ref_from_session(p, group="control"))

    win.cembed_feat_combo.setCurrentIndex(0)
    win.cembed_prev_spin.setValue(0.0)
    win.cembed_tol_spin.setValue(50.0)
    win.cembed_method_combo.setCurrentText("TSNE")        # deterministic; no umap-learn dep
    win.cembed_unit_combo.setCurrentText("Pixels")
    win._cembed_sync_unit()
    assert win.b_cembed_run.isEnabled()                   # single-slide per-pixel now allowed
    win._cembed_run()

    emb = win._cembed_emb
    assert emb.px_row is not None and emb.shapes                  # geometry retained for back-map
    assert win.b_cembed_clustermap.isEnabled()

    # Colour by Cluster, density style → one ImageItem (raster) + one centroid scatter per cluster
    win.cembed_cluster_k.setValue(4)
    win.cembed_color_combo.setCurrentText("Cluster")             # triggers a redraw
    labels = win._cembed_cluster_labels(emb)
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == 4                                        # k-means returns exactly k
    assert len(win.cembed_plot.listDataItems()) == n_clusters     # legend centroids (not the raster)
    imgs = [it for it in win.cembed_plot.items() if isinstance(it, pg.ImageItem)]
    assert len(imgs) == 1                                         # the density raster

    # Dots style falls back to per-point scatter (one item per cluster, no raster)
    win.cembed_style_combo.setCurrentText("Dots")
    assert not [it for it in win.cembed_plot.items() if isinstance(it, pg.ImageItem)]
    assert len(win.cembed_plot.listDataItems()) == n_clusters

    # Cluster map: build the spatial back-map figure (one tissue axes for the lone slide)
    captured = {}
    win._show_figure_dialog = lambda fig, title: captured.update(fig=fig, title=title)
    win._cembed_cluster_map()
    assert "fig" in captured
    axes = captured["fig"].axes
    assert len(axes) >= 1
    # the tissue panel carries an image (the painted clusters)
    painted = [ax for ax in axes if ax.images]
    assert painted
    # The back-map fills the WHOLE tissue, not just the embedded (possibly capped) subset:
    # every candidate pixel on the slide gets a cluster colour (opaque), so a big slide isn't
    # left as a few-percent speckle. Here the slide is small (no cap), so all pixels are opaque.
    img = painted[0].images[0].get_array()
    n_tissue = len(emb.tissue_rc[emb.sample_names[0]][0])
    assert int((img[..., 3] > 0).sum()) == n_tissue


def test_cohort_joint_segmentation(app, monkeypatch, tmp_path):
    """The Cohort segmentation tab builds ONE tree over several imzML slides on a shared
    axis, so a cut assigns the same cluster ids to every slide; the Detail slider re-cuts
    them all together and a cluster can be promoted to a region on the active sample."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()

    def _sync_run(fn, *a, on_done=None, want_progress=False, want_stage=False, stages=None, busy="", modal=False, title=None, **k):
        if want_progress:
            k["progress"] = lambda *_: None
        res = fn(*a, **k)
        if on_done:
            on_done(res)
    win._run = _sync_run

    win.cohort = cohort.Cohort(name="JointSegCohort")
    srcs = []
    for i in range(3):
        src = str(tmp_path / f"js{i}.imzML")
        demo.write_synthetic_imzml(src, width=16, height=12, seed=i + 1)
        ds = MSIDataset.from_imzml(src); ds.prime()
        peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
        fp = library.dataset_fingerprint(ds)
        sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                     n_pixels=ds.n_pixels, dataset_fingerprint=fp)
        p = session.save_session(session.managed_path(src, fp), sess)
        win.cohort.add(cohort.ref_from_session(p, group="g" + str(i % 2)))
        srcs.append(src)

    # the active sample is one of the cohort slides → cluster→region is enabled
    win.ds = MSIDataset.from_imzml(srcs[0]); win.ds.prime()

    win._jseg_refresh_sample_list()
    assert win.jseg_sample_list.count() == 3
    assert len(win._jseg_checked_refs()) == 3            # all imzML samples ticked by default

    win.jseg_feat_combo.setCurrentIndex(0)               # consensus shared axis
    win.jseg_prev_spin.setValue(0.0)
    win.jseg_tol_spin.setValue(50.0)
    win.jseg_detail.setValue(5)
    win._jseg_run()

    jh = win._jseg_jh
    assert jh.n_samples == 3
    segs = win._jseg_segs
    assert len(segs) == 3
    # one panel + image per slide, all sharing the same cluster vocabulary
    assert len(win._jseg_items) == 3
    k = segs[0].n_clusters
    assert all(s.n_clusters == k for s in segs) and 2 <= k <= 5
    assert win.jseg_table.rowCount() == k                # one signature row per shared cluster
    assert win.b_jseg_png.isEnabled()

    # the Detail slider re-cuts every slide together to a coarser partition
    win.jseg_detail.setValue(3)
    assert win._jseg_segs[0].n_clusters <= 3
    assert all(s.n_clusters == win._jseg_segs[0].n_clusters for s in win._jseg_segs)

    # promote a cluster to a region on the active sample
    assert win._jseg_active_idx is not None and win._jseg_active_alignable()
    before = len(win.regions)
    win._jseg_cluster_to_region(0)
    assert len(win.regions) == before + 1
    assert win.regions[-1]["mask"] is not None and win.regions[-1]["mask"].any()


def test_switch_to_region_sample_activates_scope(app, monkeypatch, tmp_path):
    """Double-clicking a region sample whose file is already loaded just swaps to its
    feature scope (no reload) and selects the region."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="SwitchCohort")
    p = _write_session_scopes("/d/s.imzML", [_peak(700.0, 0.5)],
                              {"ROI 1": [_peak(885.5, 0.4)]}, fp="sw")
    ref = win.cohort.add(cohort.SampleRef(name="s · ROI 1", session_path=p, region="ROI 1"))
    monkeypatch.setattr(win, "_active_session_path", lambda: p)   # this file is loaded
    win._feature_scopes = {"All slide": [], "ROI 1": [_peak(885.5, 0.4)]}
    calls = {}
    monkeypatch.setattr(win, "_switch_feature_scope", lambda n: calls.__setitem__("scope", n))
    monkeypatch.setattr(win, "_select_region_by_name", lambda n: calls.__setitem__("region", n))
    win._switch_to_sample(ref.key())
    assert calls == {"scope": "ROI 1", "region": "ROI 1"}


def test_region_peaks_from_targets_measures_within_region(app):
    """A region with no feature list of its own is measured at the default list's m/z, but
    each peak's intensity comes from the REGION's own mean spectrum — so two regions of one
    slide give different values (valid replicates), and rel_intensity normalises to the
    region's own peak."""
    from types import SimpleNamespace
    axis = np.array([100.0, 200.0, 300.0, 400.0])
    # spectrum keyed by mask pixel-count: region L peaks at 200, region R peaks at 300
    specs = {2: np.array([1.0, 5.0, 2.0, 0.0]), 3: np.array([1.0, 1.0, 8.0, 0.0])}

    class _DS:
        def mean_spectrum(self, mask=None):
            return axis, specs[int(np.asarray(mask).sum())]

    win = SimpleNamespace(ds=_DS(), ppm=50.0)
    f = M.MainWindow._region_peaks_from_targets
    mask_l = np.array([True, True, False, False])      # 2 px
    mask_r = np.array([True, True, True, False])       # 3 px
    pl = {round(p["mz"]): p for p in f(win, mask_l, [200.0, 300.0])}
    pr = {round(p["mz"]): p for p in f(win, mask_r, [200.0, 300.0])}
    assert pl[200]["intensity"] == 5.0 and pl[300]["intensity"] == 2.0
    assert pr[200]["intensity"] == 1.0 and pr[300]["intensity"] == 8.0   # region-specific
    assert abs(pl[200]["rel_intensity"] - 1.0) < 1e-9                    # 5 / max(5)
    assert abs(pr[300]["rel_intensity"] - 1.0) < 1e-9                    # 8 / max(8)
    assert f(win, None, [200.0]) == [] and f(win, mask_l, []) == []      # nothing to measure


def test_cohort_add_region_builds_scope_from_default_list(app, monkeypatch, tmp_path):
    """Adding a region that has no feature list of its own builds a scope from the app-wide
    default feature list (measured within the region) and joins the cohort — no separate
    Find-peaks pass, and the whole-slide entry is dropped."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run                              # region measurement now runs off-thread
    win.cohort = cohort.Cohort(name="DefaultListCohort")
    win._feature_scopes = {"All slide": [_peak(750.0, 0.5)]}     # no per-region scopes
    win._feature_lists = {"L1": [_peak(750.0, 0.4), _peak(800.0, 0.3)]}
    win._default_feature_set = ("list", "L1")                    # default = the larger ★ list
    axis = np.array([700.0, 750.0, 800.0, 850.0])
    specs = {10: np.array([1.0, 6.0, 2.0, 0.0]), 20: np.array([1.0, 2.0, 9.0, 0.0])}

    class _DS:
        source = "/d/s.imzML"
        n_pixels = 100

        def mean_spectrum(self, mask=None):
            return axis, specs[int(np.asarray(mask).sum())]

    win.ds = _DS()
    win.regions = [{"name": "L"}, {"name": "R"}]
    masks = {"L": np.array([True] * 10 + [False] * 90),
             "R": np.array([True] * 20 + [False] * 80)}
    monkeypatch.setattr(win, "_region_pixel_mask", lambda rg, *a, **k: masks[rg["name"]])
    monkeypatch.setattr(win, "_active_session_path", lambda: str(tmp_path / "s.json"))
    monkeypatch.setattr(library, "dataset_fingerprint", lambda ds: "fpx")
    monkeypatch.setattr(win, "_flush_autosave", lambda: None)    # persistence tested elsewhere

    def _accept_all(self):
        for lw in self.findChildren(QtWidgets.QListWidget):
            for i in range(lw.count()):
                it = lw.item(i)
                if it.flags() & QtCore.Qt.ItemIsEnabled:
                    it.setCheckState(QtCore.Qt.Checked)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", _accept_all)
    win._cohort_add_regions()

    # both regions now carry a measured-at-default scope on the larger (750, 800) list
    assert set(win._feature_scopes) >= {"L", "R"}
    assert sorted(round(p["mz"]) for p in win._feature_scopes["L"]) == [750, 800]
    # region-specific intensities (L peaks at 750, R at 800), not the slide's
    by_l = {round(p["mz"]): p["intensity"] for p in win._feature_scopes["L"]}
    by_r = {round(p["mz"]): p["intensity"] for p in win._feature_scopes["R"]}
    assert by_l[750] == 6.0 and by_r[800] == 9.0
    # joined the cohort as two region replicates; whole-slide entry not present
    assert {s.region for s in win.cohort.samples} == {"L", "R"}
    assert all(s.region for s in win.cohort.samples)


def test_register_skips_whole_file_when_regions_present(app, monkeypatch, tmp_path):
    """Once a file contributes region samples, auto-registration must not also add it as a
    whole slide (that would double-count the same tissue in a comparison)."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="GuardCohort")
    p = _write_session_scopes("/d/s.imzML", [_peak(700.0, 0.5)],
                              {"ROI 1": [_peak(885.5, 0.4)]}, fp="fpx")
    win.cohort.add(cohort.SampleRef(name="s · ROI 1", session_path=p, region="ROI 1"))

    class _DS:
        source = "/d/s.imzML"
        n_pixels = 100
    win.ds = _DS()
    monkeypatch.setattr(win, "_active_session_path", lambda: p)
    monkeypatch.setattr(library, "dataset_fingerprint", lambda ds: "fpx")
    win._register_active_in_cohort()
    assert len(win.cohort.samples) == 1               # no whole-slide entry added
    assert all(s.region for s in win.cohort.samples)


def test_cohort_umap_preflight_relocates_moved_samples(app, monkeypatch, tmp_path):
    """A cohort moved to a new machine has stale absolute `source` paths. The Cohort UMAP
    pre-flight detects the missing files and (after the user picks the new data folder)
    relocates them in place by name and persists the healed roster — so the run proceeds
    instead of grinding to a 'None of the N samples could be loaded' error."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    win = M.MainWindow()

    moved = tmp_path / "moved"; moved.mkdir()
    win.cohort = cohort.Cohort(name="MovedCohort")
    for i in range(2):
        real = str(moved / f"s{i}.imzML")                 # the data's current (new) location
        demo.write_synthetic_imzml(real, width=12, height=10, seed=i + 1)
        ds = MSIDataset.from_imzml(real); ds.prime()
        peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
        fp = library.dataset_fingerprint(ds)
        sess = session.build_session(source=real, settings={"mode": "negative"}, peaks=peaks,
                                     n_pixels=ds.n_pixels, dataset_fingerprint=fp)
        p = session.save_session(session.managed_path(real, fp), sess)
        ref = cohort.ref_from_session(p, group="g")
        ref.source = str(tmp_path / "orig" / f"s{i}.imzML")   # stale: the OLD path, absent here
        win.cohort.add(ref)
    win._save_cohort()

    refs = win._cohort_included_refs()
    assert len(cohort.missing_source_samples(refs)) == 2      # both look moved

    monkeypatch.setattr(win, "_ask_relocate", lambda missing, total: True)
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_existing_directory",
                        lambda *a, **k: str(moved))

    assert win._cembed_preflight_paths(refs) is True
    assert cohort.missing_source_samples(refs) == []          # both relocated by name
    assert all(os.path.exists(r.source) for r in refs)
    saved = cohort.Cohort.load(cohort.cohort_path("MovedCohort"))
    assert all(os.path.exists(s.source) for s in saved.samples)   # healed roster persisted


def test_cohort_umap_preflight_aborts_when_user_declines(app, monkeypatch, tmp_path):
    """Declining relocation returns False so the run aborts up front — no slow grind into the
    loader, no cryptic deep error."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="DeclineCohort")
    win.cohort.add(cohort.SampleRef(name="gone", session_path="",
                                    source=str(tmp_path / "nope" / "x.imzML")))
    refs = win._cohort_included_refs()
    monkeypatch.setattr(win, "_ask_relocate", lambda *a: False)
    assert win._cembed_preflight_paths(refs) is False


def test_cohort_umap_preflight_region_session_missing_aborts(app, monkeypatch, tmp_path):
    """A region sample whose imzML exists but whose session is gone (moved machine) can't be
    fixed by a data-folder pick — the preflight must NOT offer relocation, must surface the
    precise reason, and must abort instead of grinding to 'None could be loaded'."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    win = M.MainWindow()
    raw = tmp_path / "slide.imzML"; raw.write_text("x")            # source present, session absent
    win.cohort = cohort.Cohort(name="RegionCohort")
    win.cohort.add(cohort.SampleRef(name="R", session_path=str(tmp_path / "gone" / "s.json"),
                                    source=str(raw), region="ROI 1"))
    refs = win._cohort_included_refs()
    seen = {"ask": False, "warn": False}
    monkeypatch.setattr(win, "_ask_relocate", lambda *a: seen.__setitem__("ask", True) or True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: seen.__setitem__("warn", True))
    assert win._cembed_preflight_paths(refs) is False
    assert seen["ask"] is False                                   # no folder relocate offered
    assert seen["warn"] is True                                   # precise reason surfaced instead


def test_cohort_umap_preflight_remaps_unchecked_after_relocate(app, monkeypatch, tmp_path):
    """Relocation rewrites source → changes key(); the Samples ▾ exclusion set must follow the
    relocated sample to its new key, or an unticked sample silently re-enters runs."""
    from smile_msi import demo
    from smile_msi.msi import MSIDataset

    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    win = M.MainWindow()
    moved = tmp_path / "moved"; moved.mkdir()
    win.cohort = cohort.Cohort(name="RemapCohort")
    for i in range(2):
        real = str(moved / f"s{i}.imzML")
        demo.write_synthetic_imzml(real, width=12, height=10, seed=i + 1)
        ds = MSIDataset.from_imzml(real); ds.prime()
        peaks = ds.pick_peaks(snr=3, min_rel_intensity=0.01)
        fp = library.dataset_fingerprint(ds)
        sess = session.build_session(source=real, settings={"mode": "negative"}, peaks=peaks,
                                     n_pixels=ds.n_pixels, dataset_fingerprint=fp)
        p = session.save_session(session.managed_path(real, fp), sess)
        ref = cohort.ref_from_session(p, group="g")
        ref.source = str(tmp_path / "orig" / f"s{i}.imzML")        # stale source
        win.cohort.add(ref)

    win._cohort_sample_unchecked = set()
    excluded = win.cohort.samples[1]
    old_key = excluded.key()
    win._cohort_sample_unchecked.add(old_key)                     # untick the 2nd sample
    refs = win._cohort_included_refs()                            # → only the 1st

    monkeypatch.setattr(win, "_ask_relocate", lambda *a: True)
    monkeypatch.setattr("smile_msi.gui.filedialogs.get_existing_directory",
                        lambda *a, **k: str(moved))
    assert win._cembed_preflight_paths(refs) is True
    new_key = excluded.key()
    assert new_key != old_key                                     # relocation changed the key
    assert new_key in win._cohort_sample_unchecked               # exclusion followed it
    assert old_key not in win._cohort_sample_unchecked
