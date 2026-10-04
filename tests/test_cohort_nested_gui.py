"""Cohort *nested stats* tab — drives the subject × compartment screen headless.

Builds 9 nerves × 3 compartments as region-samples (each compartment reading its own saved
feature scope), then runs the real tab wiring: mixed model, contrast picker, volcano,
per-subject dot plot, CSV export, and the small-n guards.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("statsmodels")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import cohort, session  # noqa: E402
from smile_msi.gui import main as M  # noqa: E402

COMPARTMENTS = ["endoneurium", "perineurium", "epineurium"]
IONS = [700.0, 720.0, 740.0, 760.0, 780.0, 800.0]
T_INT = 700.0            # planted group×compartment interaction (treated, epineurium only)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sync_run(fn, *a, on_done=None, want_progress=False, busy="", modal=False, title=None, **k):
    if want_progress:
        k["progress"] = lambda *_: None
    res = fn(*a, **k)
    if on_done:
        on_done(res)


def _peak(mz, rel):
    return {"mz": float(mz), "intensity": rel * 100, "snr": 10.0, "rel_intensity": float(rel)}


def _same(nerve, comp, k):
    return comp                                           # regions named identically everywhere


def _per_nerve(nerve, comp, k):
    return f"{nerve} {comp}"                              # ROI labels carry the nerve id


def _opaque(nerve, comp, k):
    return f"ROI {k + 1}"                                 # no token rule can recover this


def _build_cohort(win, n_a=3, n_b=6, seed=2026, region_namer=_same, label=True):
    """9 nerves, each a slide carrying one saved feature scope per compartment.

    ``region_namer(nerve, compartment, index)`` fixes both the session's feature-scope key and
    the ``SampleRef.region``, so the two can't drift apart the way a post-hoc rename would.

    ``label=True`` writes the per-sample mapping ('Auto-label' / 'Define compartments…' produce)
    onto every ref — ``meta['subject']`` and ``meta['compartment']`` — which is where the tab now
    reads the design from. ``label=False`` leaves the cohort unlabelled, for the tests that drive
    'Auto-label' itself.
    """
    rng = np.random.default_rng(seed)
    for i in range(n_a + n_b):
        nerve = f"n{i:02d}"
        grp = "normal" if i < n_a else "treated"
        off = rng.normal(1.0, 0.04)                       # per-nerve slide level
        scopes = {}
        for k, comp in enumerate(COMPARTMENTS):
            peaks = []
            for j, mz in enumerate(IONS):
                rel = (0.10 + 0.05 * j) * off * rng.normal(1.0, 0.03)
                if mz == T_INT and grp == "treated" and comp == "epineurium":
                    rel *= 4.0                            # the interaction
                peaks.append(_peak(mz, rel))
            scopes[region_namer(nerve, comp, k)] = peaks
        sess = session.build_session(source=f"/d/{nerve}.imzML", settings={"mode": "negative"},
                                     peaks=[_peak(m, 0.3) for m in IONS], n_pixels=300,
                                     dataset_fingerprint=nerve, feature_scopes=scopes)
        sp = session.save_session(session.managed_path(f"/d/{nerve}.imzML", nerve), sess)
        for k, comp in enumerate(COMPARTMENTS):
            region = region_namer(nerve, comp, k)
            meta = {"subject": nerve, "compartment": comp} if label else {}
            win.cohort.add(cohort.SampleRef(
                name=f"{nerve} · {region}", session_path=sp, source=f"/d/{nerve}.imzML",
                group=grp, region=region, meta=meta, n_pixels=300))


def _prepare(win):
    # Subject and compartment are no longer combos — they come from the per-sample mapping the
    # cohort was built (or Auto-labelled) with, so _prepare only sets the run knobs and A/B.
    win.nest_feat_combo.setCurrentIndex(0)                # consensus axis
    win.nest_prev_spin.setValue(0.0)
    win.nest_tol_spin.setValue(50.0)
    win.nest_summary_combo.setCurrentText("Saved peak values (no reload)")
    win.nest_ga.setCurrentText("normal")
    win.nest_gb.setCurrentText("treated")


def _win(monkeypatch, tmp_path, **kw):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="NestedCohort")
    _build_cohort(win, **kw)
    win._refresh_sample_tree()                            # populates group + subject combos
    return win


def test_nested_tab_exists_beside_the_cohort_tab(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    # the four cohort screens moved off the strip into the on-demand Cohort window (plan 24
    # Phase 5): they are tabs of win._cohort_tabs, and reveal_view opens that window.
    labels = [win._cohort_tabs.tabText(i) for i in range(win._cohort_tabs.count())]
    assert "Cohort nested stats" in labels and "Cohort" in labels
    win.reveal_view("Cohort nested stats")
    assert win._cohort_window is not None and win._cohort_window.isVisible()
    assert win._cohort_tabs.tabText(win._cohort_tabs.currentIndex()) == "Cohort nested stats"


def test_group_combos_populate_and_the_mapping_drives_subject_and_compartment(
        app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    assert {win.nest_ga.itemText(i) for i in range(win.nest_ga.count())} == \
        {"normal", "treated"}
    # subject and compartment are read from the per-sample mapping, not from token-rule combos
    assert win._nest_subject_by() == "subject"
    assert win._nest_comp_from() == "meta:compartment"
    assert win._nest_compartments() == sorted(COMPARTMENTS)
    assert len(win._nest_region_refs()) == 27
    assert win.b_nest_run.isEnabled()
    # scale-over defaults to on-tissue, so background can't bias between-subject contrasts
    assert win._nest_norm_scopes[win.nest_norm_scope_combo.currentText()] == "tissue"


def test_auto_label_takes_the_subject_from_the_slide_when_the_names_dont(
        app, monkeypatch, tmp_path):
    """One nerve per imzML with region names that don't carry the nerve id (endo/peri/epi): the
    subject has to come from the file, and Auto-label reads it there — no metadata import."""
    win = _win(monkeypatch, tmp_path, label=False)        # nothing labelled yet
    win._nest_update_design_summary()
    assert "Not labelled yet" in win.nest_info.text()

    win._nest_auto_apply()
    _prepare(win)
    win._nest_run()
    res = win._nest_res
    assert res.attrs["n_subjects_a"] == 3 and res.attrs["n_subjects_b"] == 6
    assert res.attrs["subject_by"] == "subject"
    assert res[np.isclose(res["mz"], T_INT)].iloc[0]["q_interaction"] <= 0.05


def test_auto_label_recovers_subjects_from_per_nerve_names_on_one_slide(
        app, monkeypatch, tmp_path):
    """The failure a per-slide subject can't see: if all nerves were sectioned onto one slide,
    the file names nothing. Auto-label takes the nerve from the ROI label instead."""
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve, label=False)
    for r in win.cohort.samples:                          # one shared slide, nerve in the ROI
        r.source = "/d/all_nerves.imzML"
    win._refresh_sample_tree()

    win._nest_update_design_summary()
    assert "Not labelled yet" in win.nest_info.text()     # ...and it says so, before any fit

    win._nest_auto_apply()
    win._nest_update_design_summary()
    assert "9 subjects" in win.nest_info.text()
    assert win._nest_subject_by() == "subject"


def test_design_summary_flags_a_subject_in_both_groups(app, monkeypatch, tmp_path):
    """The signature of wrong subject labels: the same 'subject' on both sides of the contrast.
    Here the compartment was mistakenly used as the subject id."""
    win = _win(monkeypatch, tmp_path)
    for r in win.cohort.samples:                          # mislabel: subject := compartment
        r.meta = {**r.meta, "subject": r.meta["compartment"]}
    win._refresh_sample_tree()
    win.nest_ga.setCurrentText("normal")
    win.nest_gb.setCurrentText("treated")
    win._nest_update_design_summary()
    info = win.nest_info.text()
    assert "appear in BOTH groups" in info
    assert "3 subjects" in info                           # the compartments, mistaken for nerves


def test_nested_design_summary_reports_the_design_before_any_fit(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win._nest_update_design_summary()
    info = win.nest_info.text()
    assert "27 region-samples" in info
    assert "9 subjects" in info
    assert "3 compartments" in info
    assert "normal n=3 vs treated n=6" in info


def test_nested_mixed_model_run_renders_table_volcano_and_dots(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win.nest_model_combo.setCurrentText("Mixed model (group × compartment)")
    win._nest_run()

    res = win._nest_res
    assert res is not None
    assert res.attrs["n_subjects_a"] == 3 and res.attrs["n_subjects_b"] == 6
    assert res.attrs["n_observations"] == 27
    assert res.attrs["compartments"] == sorted(COMPARTMENTS)
    assert win.nest_table.rowCount() == len(IONS)
    assert len(win.nest_volcano.listDataItems()) >= 1

    # the planted interaction is found, and only in the epineurium
    row = res[np.isclose(res["mz"], T_INT)].iloc[0]
    assert row["q_interaction"] <= 0.05
    assert row["p__epineurium"] < 0.05
    assert row["p__endoneurium"] > 0.05
    assert row["log2_fc__epineurium"] > 1.0
    # no other ion carries one
    assert (res[~np.isclose(res["mz"], T_INT)]["q_interaction"] > 0.05).all()

    # contrast picker offers the interaction, the averaged group effect, and each compartment
    labels = [win.nest_contrast_combo.itemText(i) for i in range(win.nest_contrast_combo.count())]
    assert labels[0] == "Interaction (group × compartment)"
    assert "Group (all compartments)" in labels
    assert all(f"Group within {c}" in labels for c in COMPARTMENTS)

    # per-subject dot plot: 3 compartments × 2 groups → 6 scatters + 6 group-mean bars
    win._nest_render_dots(T_INT)
    assert len(win.nest_dots.listDataItems()) >= 6
    assert f"{T_INT:.4f}" in win.nest_dots_caption.text()
    assert "interaction q" in win.nest_dots_caption.text()

    for b in win._nest_export_btns:
        assert b.isEnabled()


def test_nested_run_warns_that_a_rank_test_could_not_reject(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win._nest_run()
    info = win.nest_info.text()
    # 3-vs-6 over 6 features: BH admits at most 2, so an exact rank test is arithmetically
    # incapable of rejecting. The screen must say so before the user reads a q-value.
    assert info.startswith("⚠")
    assert "could not reject anything" in info
    assert "3-vs-6" in info
    assert "normal (n=3) vs treated (n=6) subjects" in info
    assert "27 profiles across 3 compartments" in info


def test_nested_switching_contrast_resorts_the_table_and_relabels_the_volcano(
        app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win._nest_run()
    win.nest_contrast_combo.setCurrentText("Group within epineurium")
    assert "Group within epineurium" in win.nest_volcano.getAxis("left").labelText
    # the table sorts by the shown contrast, so the planted ion tops it in the epineurium
    assert float(win.nest_table.item(0, 0).text()) == pytest.approx(T_INT, abs=0.01)

    win.nest_contrast_combo.setCurrentText("Group within endoneurium")
    assert float(win.nest_table.item(0, 0).text()) != pytest.approx(T_INT, abs=0.01)


def test_nested_stratified_model_skips_the_interaction_contrast(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win.nest_model_combo.setCurrentText("Stratified (test each compartment)")
    assert win.nest_test_combo.isEnabled()               # test picker only lives here
    win.nest_test_combo.setCurrentText("Moderated t (empirical Bayes)")
    win._nest_run()

    res = win._nest_res
    assert res.attrs["model"] == "stratified"
    assert "q_interaction" not in res.columns
    labels = [win.nest_contrast_combo.itemText(i) for i in range(win.nest_contrast_combo.count())]
    assert "Interaction (group × compartment)" not in labels
    row = res[np.isclose(res["mz"], T_INT)].iloc[0]
    assert row["q__epineurium"] <= 0.05
    assert row["q__endoneurium"] > 0.05


def test_run_refuses_when_subjects_are_unlabelled(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    for r in win.cohort.samples:                          # compartment stays, subject stripped
        r.meta = {"compartment": r.meta["compartment"]}
    assert win._nest_subject_by() == "subject"
    win._nest_run()
    assert getattr(win, "_nest_res", None) is None
    msg = win.statusBar().currentMessage()
    assert "have no subject label" in msg
    assert "every compartment must name its subject" in msg


# --------------------------------------------------------------------------- #
# per-nerve region names — the naming problem, and the ways out
# --------------------------------------------------------------------------- #
def test_per_nerve_names_are_unlabelled_until_auto_label(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve, label=False)
    _prepare(win)
    win._nest_update_design_summary()
    assert "Not labelled yet" in win.nest_info.text()
    win._nest_run()
    assert getattr(win, "_nest_res", None) is None
    assert "Not labelled yet" in win.statusBar().currentMessage()


def test_auto_label_collapses_per_nerve_names_to_shared_compartments(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve, label=False)
    win._nest_auto_apply()
    # 'n00 endoneurium', 'n01 endoneurium', … all collapse onto one 'endoneurium' compartment
    assert win._nest_compartments() == sorted(COMPARTMENTS)
    _prepare(win)
    win._nest_run()
    res = win._nest_res
    assert res.attrs["compartment_from"] == "meta:compartment"
    assert res.attrs["compartments"] == sorted(COMPARTMENTS)
    assert res[np.isclose(res["mz"], T_INT)].iloc[0]["q_interaction"] <= 0.05


def test_saved_mapping_survives_names_no_rule_could_parse(app, monkeypatch, tmp_path):
    """What 'Define compartments…' writes: an explicit per-sample compartment that survives
    region names ('ROI 1', 'ROI 2', …) no positional rule could recover."""
    win = _win(monkeypatch, tmp_path, region_namer=_opaque)   # label=True → mapping is stored
    assert win._nest_comp_from() == "meta:compartment"
    assert win._nest_compartments() == sorted(COMPARTMENTS)
    _prepare(win)
    win._nest_run()
    res = win._nest_res
    assert res.attrs["compartment_from"] == "meta:compartment"
    assert res[np.isclose(res["mz"], T_INT)].iloc[0]["q_interaction"] <= 0.05


def test_nest_apply_mapping_rescues_a_group_label_that_hides_the_compartment(
        app, monkeypatch, tmp_path):
    """The real-world case: Region='S01 endo' (nerve + compartment) and Group='Trt endo'
    (group + compartment). Neither column carries the normal-vs-treated axis on its own, so
    picking Group A/B directly would silently collapse the model to one compartment."""
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve)
    for r in win.cohort.samples:                         # Group='Normal endoneurium' etc.
        r.group = f"{'Normal' if r.group == 'normal' else 'Trt'} {r.region.split(' ', 1)[1]}"
    win._refresh_sample_tree()
    assert len(win.cohort.groups()) == 6                 # 2 groups × 3 compartments, tangled

    cmap = {r.region: r.region.split(" ", 1)[1] for r in win.cohort.samples}
    gmap = {g: g.split(" ", 1)[0].lower() for g in win.cohort.groups()}
    win._nest_apply_mapping(cmap, gmap)

    assert sorted(win.cohort.groups()) == ["normal", "trt"]
    assert win._nest_comp_from() == "meta:compartment"   # adopted without being asked
    assert win._nest_compartments() == sorted(COMPARTMENTS)
    info = win.nest_info.text()
    assert "3 compartments" in info and "9 subjects" in info

    _prepare(win)
    win.nest_ga.setCurrentText("normal")
    win.nest_gb.setCurrentText("trt")
    win._nest_run()
    res = win._nest_res
    assert res.attrs["n_subjects_a"] == 3 and res.attrs["n_subjects_b"] == 6
    assert res.attrs["compartments"] == sorted(COMPARTMENTS)
    assert res[np.isclose(res["mz"], T_INT)].iloc[0]["q_interaction"] <= 0.05


def test_nest_apply_mapping_persists_and_excludes_blank_compartments(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve)
    cmap = {r.region: ("" if "epineurium" in r.region else r.region.split(" ", 1)[1])
            for r in win.cohort.samples}
    win._nest_apply_mapping(cmap, {g: g for g in win.cohort.groups()})
    # a blank compartment removes the key rather than storing '' as a compartment name
    assert win._nest_compartments() == ["endoneurium", "perineurium"]
    epi = [r for r in win.cohort.samples if "epineurium" in r.region]
    assert all("compartment" not in (r.meta or {}) for r in epi)
    assert cohort.compartment_id(epi[0], "meta:compartment") == ""
    # ...and the mapping survives a reload of the saved roster
    reloaded = cohort.Cohort.from_dict(win.cohort.to_dict())
    mapped = [r for r in reloaded.samples if (r.meta or {}).get("compartment")]
    assert len(mapped) == 18


def test_nest_design_dialog_seeds_the_mapping_and_applies_it_on_accept(
        app, monkeypatch, tmp_path):
    """Drive the real dialog: accepting it without editing must apply the seeded guess, which
    for 'n00 endoneurium' / 'Normal endoneurium' is exactly the mapping we want."""
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve)
    for r in win.cohort.samples:
        r.group = f"{'Normal' if r.group == 'normal' else 'Trt'} {r.region.split(' ', 1)[1]}"
    win._refresh_sample_tree()

    monkeypatch.setattr(QtWidgets.QDialog, "exec",
                        lambda self: QtWidgets.QDialog.Accepted)
    win._nest_design_dialog()

    assert win._nest_compartments() == sorted(COMPARTMENTS)
    assert sorted(win.cohort.groups()) == ["Normal", "Trt"]
    assert "3 compartment(s)" in win.statusBar().currentMessage()
    # the Subject column seeds from the region's first token, so every nerve is its own subject
    assert {(r.meta or {}).get("subject") for r in win.cohort.samples} == \
        {f"n{i:02d}" for i in range(9)}
    assert win._nest_subject_by() == "subject"             # ...and the combo adopts it
    win._nest_update_design_summary()
    assert "9 subjects" in win.nest_info.text()


def test_nest_design_dialog_cancel_changes_nothing(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve)
    before = [(r.group, dict(r.meta or {})) for r in win.cohort.samples]
    monkeypatch.setattr(QtWidgets.QDialog, "exec",
                        lambda self: QtWidgets.QDialog.Rejected)
    win._nest_design_dialog()
    assert [(r.group, dict(r.meta or {})) for r in win.cohort.samples] == before


def test_nest_design_dialog_without_region_samples_explains_itself(app, monkeypatch, tmp_path):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win.cohort = cohort.Cohort(name="Empty")
    seen = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: seen.append(a[2])))
    win._nest_design_dialog()
    assert seen and "no region-samples" in seen[0]


def test_nested_run_refuses_group_labels_that_embed_the_compartment(app, monkeypatch, tmp_path):
    """The run that produced 231 rows of `converged = FALSE`: Group A='Normal endo',
    B='Trt endo' restricts the table to endo rows while the model still expects three
    compartments, so every design matrix is rank-deficient."""
    win = _win(monkeypatch, tmp_path, region_namer=_per_nerve)
    for r in win.cohort.samples:                          # 6 tangled groups
        comp = r.region.split(" ", 1)[1]
        r.group = f"{'Normal' if r.group == 'normal' else 'Trt'} {comp}"
    win._refresh_sample_tree()
    _prepare(win)
    win.nest_ga.setCurrentText("Normal endoneurium")
    win.nest_gb.setCurrentText("Trt endoneurium")

    st = win._nest_design_state()
    assert st["n_a"] == 3 and st["n_b"] == 6              # the counts still look fine...
    assert sorted(st["missing_in_ab"]) == ["epineurium", "perineurium"]   # ...but two are absent

    win._nest_run()
    assert getattr(win, "_nest_res", None) is None
    msg = win.statusBar().currentMessage()
    assert "contain no epineurium, perineurium samples" in msg
    assert "group labels embed the compartment" in msg
    assert "Define compartments…" in msg
    # the summary says the same thing, from the same resolver
    assert "group labels embed the compartment" in win.nest_info.text()


def test_nested_run_refuses_a_compartment_missing_from_one_group(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    for r in win.cohort.samples:                          # normal nerves have no epineurium
        if r.group == "normal" and r.region == "epineurium":
            r.group = "excluded"
    win._refresh_sample_tree()
    _prepare(win)
    st = win._nest_design_state()
    assert st["empty_cells"] == ["epineurium"]
    win._nest_run()
    assert getattr(win, "_nest_res", None) is None
    assert "missing from one of the two groups" in win.statusBar().currentMessage()


def test_nested_design_state_and_run_guard_agree(app, monkeypatch, tmp_path):
    """The healthy design must be accepted by the same resolver that rejects the broken ones."""
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    st = win._nest_design_state()
    assert win._nest_blocking_problem(st) == ""
    assert st["missing_in_ab"] == [] and st["empty_cells"] == [] and st["shared"] == []
    assert st["n_subjects"] == 9
    win._nest_run()
    assert win._nest_res is not None
    assert win._nest_res.attrs["n_testable"] > 0


def test_nested_mixed_model_names_the_missing_compartment_in_its_warning():
    from smile_msi import hierstats as hs

    # 6 subjects, but only the endo compartment is present while three are modelled
    X = np.random.default_rng(3).normal(0, 1, (6, 4))
    g = np.array(["a"] * 3 + ["b"] * 3)
    c = np.array(["endo"] * 6)
    s = np.array([f"n{i}" for i in range(6)])
    res = hs.nested_mixed_model(X, g, c, s, "a", "b", compartments=["endo", "epi", "peri"])
    assert not res["converged"].any()
    assert "epi, peri are absent" in res.attrs["warning"]
    assert "rank-deficient" in res.attrs["warning"]
    assert "group labels don't embed the compartment" in res.attrs["warning"]


# --------------------------------------------------------------------------- #
# lipid class roll-up
# --------------------------------------------------------------------------- #
def test_nested_class_button_follows_the_model_that_ran(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    assert not win.b_nest_class.isEnabled()             # nothing to roll up yet
    win._nest_run()
    assert win.b_nest_class.isEnabled()

    win.nest_model_combo.setCurrentText("Stratified (test each compartment)")
    win._nest_run()
    # the stratified path reports no standard errors, so the class statistic can't be formed
    assert not win.b_nest_class.isEnabled()
    win._nest_class_dialog()
    assert "needs the mixed model's standard errors" in win.statusBar().currentMessage()


def test_nested_class_dialog_builds_a_table_and_exports_it(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win._nest_run()
    # annotate_df left most ions unidentified in this synthetic cohort; plant a class so the
    # roll-up has something to test, exactly as a real annotated result would.
    res = win._nest_res
    res["best_class"] = ["Sulfatide"] * 4 + [None] * (len(res) - 4)
    res["id_confidence"] = [60.0] * 4 + [45.0] * (len(res) - 4)

    seen = {}
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: seen.setdefault("shown", True))
    win._nest_class_dialog()
    assert seen.get("shown")

    out = cohort.nested_class_comparison(win._nest_tbl, res, contrast="endoneurium", min_ions=3)
    assert list(out["class"]) == ["Sulfatide"]
    row = out.iloc[0]
    assert row["n_ions"] == 4
    assert row["n_effective"] <= 4
    assert row["p_value"] >= row["p_uncorrected"] - 1e-12

    outp = tmp_path / "class_test.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(outp), "CSV (*.csv)")))
    win._nest_class_export(out)
    text = outp.read_text(encoding="utf-8")
    assert "# contrast: endoneurium" in text
    assert "# center_observations: True" in text
    assert "Wu & Smyth" in text
    assert "p_uncorrected" in text


def test_nested_class_dialog_needs_a_result(app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    win._nest_class_dialog()
    assert "Run a nested comparison first" in win.statusBar().currentMessage()


def test_nest_seed_helpers_guess_the_mapping():
    from smile_msi.gui.cohortview import CohortMixin
    assert CohortMixin._nest_seed_compartment("S01 endo") == "endo"
    assert CohortMixin._nest_seed_compartment("s02_peri") == "peri"
    assert CohortMixin._nest_seed_compartment("") == ""
    # 'Trt endo' → 'Trt' once 'endo' is a known compartment; a clean label is left alone
    seed = CohortMixin._nest_seed_group
    known = {"endo", "epi", "peri"}
    assert seed(None, "Trt endo", known) == "Trt"
    assert seed(None, "Normal epi", known) == "Normal"
    assert seed(None, "normal", known) == "normal"
    assert seed(None, "endo", known) == "endo"           # nothing left → keep the label


def test_nested_run_refuses_one_subject_per_group(app, monkeypatch, tmp_path):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    win = M.MainWindow()
    win._run = _sync_run
    win.cohort = cohort.Cohort(name="Thin")
    _build_cohort(win, n_a=1, n_b=6)
    win._refresh_sample_tree()
    _prepare(win)
    win._nest_run()
    # 1 nerve × 3 compartments must NOT read as n=3 — the guard counts subjects, not rows
    assert "Each group needs ≥2 subjects" in win.statusBar().currentMessage()
    assert "normal has 1" in win.statusBar().currentMessage()


def test_nested_csv_export_stamps_the_model_and_the_feasibility_verdict(
        app, monkeypatch, tmp_path):
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    win._nest_run()
    out = tmp_path / "nested.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")))
    win._nest_export_csv()
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # a table of q-values whose design is unrecorded is the artifact this screen prevents
    assert "# model: lmm" in text
    assert "# subject_by: subject" in text
    assert "# transform: log2" in text
    assert "# df_method: bw" in text
    assert "linear mixed model" in text
    assert "# rank_test_feasibility:" in text
    assert "q_interaction" in text.splitlines()[-len(IONS) - 1]


def test_nested_run_records_an_audit_step(app, monkeypatch, tmp_path):
    """The run must stamp the audit trail with the design, not just the numbers — that record
    is what provenance.methods_paragraph turns into the methods prose. (win.prov is None with
    no slide loaded, so capture the call rather than the store.)"""
    win = _win(monkeypatch, tmp_path)
    _prepare(win)
    seen = {}

    def _capture(kind, *, label="", params=None, regions=None, **k):
        seen["kind"], seen["params"], seen["regions"] = kind, params or {}, regions
        return {}

    monkeypatch.setattr(win, "record_step", _capture)
    win._nest_run()

    assert seen["kind"] == "cohort_nested_comparison"
    p = seen["params"]
    assert p["model"] == "lmm" and p["subject_by"] == "subject"
    assert p["n_subjects_a"] == 3 and p["n_subjects_b"] == 6
    assert p["n_observations"] == 27
    assert p["compartments"] == sorted(COMPARTMENTS)
    assert p["df_method"] == "bw"
    assert p["summary"] == "saved" and p["transform"] == "log2"
    # every region-sample that fed a group is recorded, tagged A or B
    roles = [role for role, _info in seen["regions"]]
    assert roles.count("A") == 9 and roles.count("B") == 18
