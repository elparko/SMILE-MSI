"""Cohort engine tests — pure, headless (no Qt, no real imzML).

Builds throwaway managed sessions under a tmp ``SMILE_MSI_HOME`` and exercises the
roster model, session discovery, the metadata-only sample x feature table, and the
between-group comparison.
"""
import os

import numpy as np
import pytest

from smile_msi import cohort, session


def _peak(mz, rel, intensity=None):
    return {"mz": float(mz), "intensity": float(intensity if intensity is not None else rel * 100),
            "snr": 10.0, "rel_intensity": float(rel)}


def _write_session(source, peaks, fp, n_pixels=100):
    """Write a managed session for ``source`` and return its path."""
    sess = session.build_session(source=source, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=n_pixels, dataset_fingerprint=fp)
    path = session.managed_path(source, fp)
    return session.save_session(path, sess)


def _write_session_scopes(source, peaks, scopes, fp, n_pixels=100):
    """Write a managed session that also carries per-region ``feature_scopes`` — what a
    region sample reads its intensities from."""
    sess = session.build_session(source=source, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=n_pixels, dataset_fingerprint=fp, feature_scopes=scopes)
    path = session.managed_path(source, fp)
    return session.save_session(path, sess)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    return tmp_path


def test_cohort_save_load_roundtrip(home):
    c = cohort.Cohort(name="Nerve study")
    c.add(cohort.SampleRef(name="A", session_path="/x/a.json", group="control", n_pixels=50))
    c.add(cohort.SampleRef(name="B", session_path="/x/b.json", group="treated", n_pixels=60))
    # re-adding the same session is idempotent (keyed on session path)
    c.add(cohort.SampleRef(name="A-again", session_path="/x/a.json"))
    assert len(c.samples) == 2
    assert c.groups() == ["control", "treated"]
    assert set(c.by_group()) == {"control", "treated"}

    path = c.save()
    assert path == cohort.cohort_path("Nerve study")
    again = cohort.Cohort.load(path)
    assert again.name == "Nerve study"
    assert [s.name for s in again.samples] == ["A", "B"]
    assert again.find("/x/b.json").group == "treated"

    # set_group + remove
    assert again.set_group("/x/a.json", "treated")
    assert again.find("/x/a.json").group == "treated"
    assert again.remove("/x/a.json")
    assert len(again.samples) == 1

    listing = cohort.list_cohorts()
    assert any(r["name"] == "Nerve study" for r in listing)


def test_group_by_folder(home):
    c = cohort.Cohort()
    c.add(cohort.SampleRef(name="a", session_path="/x/a.json", source="/data/ctrl/a.imzML"))
    c.add(cohort.SampleRef(name="b", session_path="/x/b.json", source="/data/ctrl/b.imzML"))
    c.add(cohort.SampleRef(name="c", session_path="/x/c.json", source="/data/treated/c.imzML"))
    assert c.group_by_folder() == 3
    assert c.by_group().keys() == {"ctrl", "treated"}
    assert [s.group for s in c.samples] == ["ctrl", "ctrl", "treated"]
    # idempotent — re-running changes nothing
    assert c.group_by_folder() == 0


def test_group_by_pattern(home):
    c = cohort.Cohort()
    c.add(cohort.SampleRef(name="WT_mouse1", session_path="/x/1.json"))
    c.add(cohort.SampleRef(name="WT_mouse2", session_path="/x/2.json"))
    c.add(cohort.SampleRef(name="KO_mouse3", session_path="/x/3.json"))
    c.add(cohort.SampleRef(name="nomatch", session_path="/x/4.json"))
    # capture group becomes the label; the non-matching sample keeps its (empty) group
    assert c.group_by_pattern(r"^(WT|KO)") == 3
    assert [s.group for s in c.samples] == ["WT", "WT", "KO", ""]
    # whole-match fallback when the pattern has no capture group
    c2 = cohort.Cohort()
    c2.add(cohort.SampleRef(name="batch7-slideA", session_path="/x/a.json"))
    assert c2.group_by_pattern(r"batch\d+") == 1
    assert c2.samples[0].group == "batch7"
    with pytest.raises(__import__("re").error):
        c.group_by_pattern("(")


def test_group_by_meta(home):
    c = cohort.Cohort()
    c.add(cohort.SampleRef(name="a", session_path="/x/a.json", meta={"sex": "M"}))
    c.add(cohort.SampleRef(name="b", session_path="/x/b.json", meta={"sex": "F"}))
    c.add(cohort.SampleRef(name="c", session_path="/x/c.json"))   # no 'sex' → unchanged
    assert c.meta_keys() == ["sex"]
    assert c.group_by_meta("sex") == 2
    assert [s.group for s in c.samples] == ["M", "F", ""]


def test_apply_metadata(home):
    c = cohort.Cohort()
    c.add(cohort.SampleRef(name="slide1.imzML", session_path="/x/1.json",
                           source="/data/slide1.imzML"))
    c.add(cohort.SampleRef(name="slide2.imzML", session_path="/x/2.json",
                           source="/data/slide2.imzML"))
    # match on the basename-without-extension; fill meta + group from columns
    rows = [{"sample": "slide1", "group": "ctrl", "age": "12"},
            {"sample": "slide2", "group": "treated", "age": "20"},
            {"sample": "missing", "group": "x", "age": "0"}]
    n = c.apply_metadata(rows, key_col="sample", group_col="group")
    assert n == 2
    assert [s.group for s in c.samples] == ["ctrl", "treated"]
    assert c.find("/data/slide1.imzML").meta == {"age": "12"}
    assert c.meta_keys() == ["age"]


def test_discover_samples(home):
    _write_session("/data/slide1.imzML", [_peak(700.5, 0.9)], fp="fp1")
    _write_session("/data/slide2.imzML", [_peak(701.5, 0.5)], fp="fp2")
    refs = cohort.discover_samples()
    assert len(refs) == 2
    names = {r.name for r in refs}
    assert names == {"slide1.imzML", "slide2.imzML"}
    assert all(r.session_path.endswith(".json") for r in refs)


def test_batch_feature_table_matches_within_ppm(home):
    # slide A has a peak ~5 ppm off the target; slide B is missing it entirely
    p_a = _write_session("/d/a.imzML", [_peak(885.5500, 0.80), _peak(700.0, 0.30)], fp="a")
    p_b = _write_session("/d/b.imzML", [_peak(700.0, 0.60)], fp="b")
    refs = [cohort.ref_from_session(p_a, group="g1"),
            cohort.ref_from_session(p_b, group="g2")]
    targets = [885.5500, 700.0]

    tbl = cohort.batch_feature_table(refs, targets, tol_ppm=20.0, value="rel_intensity")
    assert list(tbl.index) == ["a.imzML", "b.imzML"]
    assert list(tbl["group"]) == ["g1", "g2"]
    cols = [c for c in tbl.columns if c != "group"]
    # 885.55 present in A only
    assert tbl.loc["a.imzML", cols[0]] == pytest.approx(0.80)
    assert np.isnan(tbl.loc["b.imzML", cols[0]])
    # 700.0 present in both
    assert tbl.loc["a.imzML", cols[1]] == pytest.approx(0.30)
    assert tbl.loc["b.imzML", cols[1]] == pytest.approx(0.60)

    # a target far from any peak (> tol) is all-NaN
    tbl2 = cohort.batch_feature_table(refs, [900.0], tol_ppm=5.0)
    assert tbl2.iloc[:, 1].isna().all()


def test_batch_feature_table_disambiguates_duplicate_names(home):
    # two different slides that happen to share a basename get distinct session files
    p1 = _write_session("/run1/sample.imzML", [_peak(800.0, 0.5)], fp="run1")
    p2 = _write_session("/run2/sample.imzML", [_peak(800.0, 0.7)], fp="run2")
    refs = [cohort.ref_from_session(p1), cohort.ref_from_session(p2)]
    tbl = cohort.batch_feature_table(refs, [800.0])
    assert list(tbl.index) == ["sample.imzML", "sample.imzML #2"]


def test_group_comparison(home):
    # feature t1 is up in g2; feature t2 is flat between groups
    t1, t2 = 750.0, 600.0
    refs = []
    for i, rel in enumerate([0.10, 0.12, 0.11]):
        p = _write_session(f"/d/ctrl{i}.imzML", [_peak(t1, rel), _peak(t2, 0.5)], fp=f"c{i}")
        refs.append(cohort.ref_from_session(p, group="control"))
    for i, rel in enumerate([0.80, 0.75, 0.85]):
        p = _write_session(f"/d/syn{i}.imzML", [_peak(t1, rel), _peak(t2, 0.5)], fp=f"s{i}")
        refs.append(cohort.ref_from_session(p, group="treated"))

    tbl = cohort.batch_feature_table(refs, [t1, t2])
    res = cohort.group_comparison(tbl, "control", "treated")
    assert set(res.columns) >= {"mz", "n_A", "n_B", "mean_A", "mean_B", "log2_fc",
                                "p_value", "q_value"}
    assert res.attrs["a_label"] == "control" and res.attrs["b_label"] == "treated"
    row1 = res[res["mz"] == t1].iloc[0]
    assert row1["n_A"] == 3 and row1["n_B"] == 3
    assert row1["log2_fc"] > 1.0                      # ~8x up in treated
    row2 = res[res["mz"] == t2].iloc[0]
    assert abs(row2["log2_fc"]) < 0.1                 # flat
    # the discriminating feature sorts to the top (lowest p)
    assert res.iloc[0]["mz"] == t1


def test_group_comparison_parametric_methods(home):
    # same setup as test_group_comparison; Welch/Student t-tests are an opt-in
    # alternative to the default Mann-Whitney and must still flag the up feature.
    import numpy as np
    from scipy.stats import ttest_ind

    t1, t2 = 750.0, 600.0
    refs = []
    for i, rel in enumerate([0.10, 0.12, 0.11]):
        p = _write_session(f"/d/ctrl{i}.imzML", [_peak(t1, rel), _peak(t2, 0.5)], fp=f"c{i}")
        refs.append(cohort.ref_from_session(p, group="control"))
    for i, rel in enumerate([0.80, 0.75, 0.85]):
        p = _write_session(f"/d/syn{i}.imzML", [_peak(t1, rel), _peak(t2, 0.5)], fp=f"s{i}")
        refs.append(cohort.ref_from_session(p, group="treated"))
    tbl = cohort.batch_feature_table(refs, [t1, t2])

    for method, label in [("welch", "Welch's t-test"), ("student", "Student's t-test")]:
        res = cohort.group_comparison(tbl, "control", "treated", method=method)
        assert res.attrs["test"] == label
        assert res.iloc[0]["mz"] == t1                # discriminating feature still on top
        row1 = res[res["mz"] == t1].iloc[0]
        # p matches scipy run directly on the per-sample values for that feature
        col = f"mz_{t1:.4f}"
        a = tbl[tbl["group"] == "control"][col].to_numpy()
        b = tbl[tbl["group"] == "treated"][col].to_numpy()
        _, p = ttest_ind(a, b, equal_var=(method == "student"))
        assert abs(row1["p_value"] - p) < 1e-9

    with pytest.raises(ValueError):
        cohort.group_comparison(tbl, "control", "treated", method="bogus")


def _paired_refs():
    """6 donors, each contributing a control and a treated nerve. Feature t1 is higher in the
    treated nerve **within every donor** by a distinct positive amount (a consistent paired
    effect masked by large between-donor variance, with distinct within-pair deltas so the
    Wilcoxon signed-rank runs its exact test); feature t2 is flat. Returns (refs, t1, t2)."""
    t1, t2 = 750.0, 600.0
    ctrl_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    deltas = [0.05, 0.08, 0.11, 0.09, 0.07, 0.06]     # all >0, all distinct magnitudes
    refs = []
    for i, (base, d) in enumerate(zip(ctrl_levels, deltas)):
        subj = f"donor{i}"
        pc = _write_session(f"/d/ctrl{i}.imzML", [_peak(t1, base), _peak(t2, 0.5)], fp=f"c{i}")
        rc = cohort.ref_from_session(pc, group="control"); rc.meta["subject"] = subj
        pt = _write_session(f"/d/trt{i}.imzML", [_peak(t1, base + d), _peak(t2, 0.5)], fp=f"t{i}")
        rt = cohort.ref_from_session(pt, group="treated"); rt.meta["subject"] = subj
        refs += [rc, rt]
    return refs, t1, t2


def test_group_comparison_paired_wilcoxon(home):
    # A consistent within-donor +0.10 shift that the *unpaired* rank test can't separate
    # (control/treated ranges fully overlap) is significant once paired on the donor.
    refs, t1, t2 = _paired_refs()
    tbl = cohort.batch_feature_table(refs, [t1, t2])

    unpaired = cohort.group_comparison(tbl, "control", "treated")
    assert unpaired[unpaired["mz"] == t1].iloc[0]["q_value"] > 0.05   # overlap → not significant

    res = cohort.group_comparison(tbl, "control", "treated", paired=True, pair_by="subject")
    assert res.attrs["paired"] is True and res.attrs["pair_by"] == "subject"
    assert res.attrs["n_pairs"] == 6
    assert res.attrs["test"].startswith("Wilcoxon")
    row1 = res[res["mz"] == t1].iloc[0]
    assert row1["n_A"] == 6 and row1["n_B"] == 6        # pairs, not raw sample counts
    assert row1["log2_fc"] > 0                          # treated up within every donor
    assert row1["q_value"] <= 0.05                      # 6 concordant pairs → p=0.03125
    # the flat feature has all-zero within-pair diffs → undefined test → untested (NaN), not 1.0
    row2 = res[res["mz"] == t2].iloc[0]
    assert np.isnan(row2["p_value"])


def test_group_comparison_paired_ttest_and_guards(home):
    from scipy.stats import ttest_rel

    refs, t1, t2 = _paired_refs()
    tbl = cohort.batch_feature_table(refs, [t1, t2])

    res = cohort.group_comparison(tbl, "control", "treated", method="paired-t", pair_by="subject",
                                  paired=True)
    assert res.attrs["test"] == "Paired t-test"
    # p matches scipy ttest_rel on the donor-aligned per-sample values for t1
    col = f"mz_{t1:.4f}"
    a = tbl[tbl["group"] == "control"][col].to_numpy()
    b = tbl[tbl["group"] == "treated"][col].to_numpy()
    _, p = ttest_rel(a, b)
    assert abs(res[res["mz"] == t1].iloc[0]["p_value"] - p) < 1e-9

    # paired=True without a pairing key is a hard error (never silently mispairs)
    with pytest.raises(ValueError):
        cohort.group_comparison(tbl, "control", "treated", paired=True)


def test_group_comparison_reports_prevalence(home):
    # t_all is detected in every sample; t_some in only 1 control + 1 treated (2 of 6).
    t_all, t_some = 700.0, 800.0
    refs = []
    for grp, tag in (("control", "c"), ("treated", "t")):
        for i in range(3):
            peaks = [_peak(t_all, 0.5)] + ([_peak(t_some, 0.4)] if i == 0 else [])
            p = _write_session(f"/d/{tag}{i}.imzML", peaks, fp=f"{tag}{i}")
            refs.append(cohort.ref_from_session(p, group=grp))
    tbl = cohort.batch_feature_table(refs, [t_all, t_some])      # missing='nan' → absent = NaN
    res = cohort.group_comparison(tbl, "control", "treated")
    assert res.attrs["n_compared"] == 6
    r_all = res[res["mz"] == t_all].iloc[0]
    assert r_all["n_detected"] == 6 and r_all["prevalence"] == 1.0
    # a sparse ion still gets a prevalence even though it's untested (< 2 detections per group)
    r_some = res[res["mz"] == t_some].iloc[0]
    assert r_some["n_detected"] == 2 and abs(r_some["prevalence"] - 2 / 6) < 1e-9
    assert np.isnan(r_some["p_value"])


def test_consensus_targets(home):
    # three near-identical m/z across samples collapse to one centroid; a singleton
    # present in only one sample is dropped at min_prevalence > 1/3
    p1 = _write_session("/d/x.imzML", [_peak(885.500, 0.5), _peak(700.0, 0.4)], fp="x")
    p2 = _write_session("/d/y.imzML", [_peak(885.510, 0.6)], fp="y")
    p3 = _write_session("/d/z.imzML", [_peak(885.505, 0.7)], fp="z")
    refs = [cohort.ref_from_session(p) for p in (p1, p2, p3)]

    allt = cohort.consensus_targets(refs, tol_ppm=30.0, min_prevalence=0.0)
    assert any(abs(t - 885.505) < 0.02 for t in allt)
    assert any(abs(t - 700.0) < 0.02 for t in allt)

    common = cohort.consensus_targets(refs, tol_ppm=30.0, min_prevalence=0.5)
    # 885.5 cluster is in all 3 samples (kept); 700.0 only in 1/3 (dropped)
    assert any(abs(t - 885.505) < 0.02 for t in common)
    assert not any(abs(t - 700.0) < 0.02 for t in common)


def test_features_present_flags_foreign_axis(home):
    """features_present answers the cheap pre-flight 'do any targets occur in these slides'
    peaks?' so the GUI can warn before a long pooled load. A matching axis overlaps; a foreign
    axis leaves n_with_peaks>0 but n_overlapping==0; empty targets short-circuit."""
    p1 = _write_session("/d/a.imzML", [_peak(885.50, 0.5), _peak(700.0, 0.4)], fp="a")
    p2 = _write_session("/d/b.imzML", [_peak(885.51, 0.6)], fp="b")
    refs = [cohort.ref_from_session(p) for p in (p1, p2)]

    # A target within tol of a real peak on ≥1 slide → overlap found (early-exits at 1).
    n_with, n_over = cohort.features_present(refs, [885.505], tol_ppm=30.0)
    assert n_with >= 1 and n_over == 1

    # A foreign axis matches nothing → both slides have peaks, none overlap.
    assert cohort.features_present(refs, [10000.0, 20000.0], tol_ppm=30.0) == (2, 0)

    # Empty targets short-circuits without reading anything.
    assert cohort.features_present(refs, []) == (0, 0)


def test_region_samples_distinct_keys_and_roundtrip(home):
    # several regions of one slide (same session_path) must coexist — key folds in the
    # region name; the whole-slide entry keys on the plain path (back-compat)
    c = cohort.Cohort(name="Region study")
    whole = cohort.SampleRef(name="slide", session_path="/x/a.json")
    s1 = cohort.SampleRef(name="slide · ROI 1", session_path="/x/a.json", region="ROI 1")
    s2 = cohort.SampleRef(name="slide · ROI 2", session_path="/x/a.json", region="ROI 2")
    assert len({whole.key(), s1.key(), s2.key()}) == 3
    assert whole.scope() is None and s1.scope() == "ROI 1"
    c.add(whole); c.add(s1); c.add(s2)
    c.add(cohort.SampleRef(name="dup", session_path="/x/a.json", region="ROI 1"))  # idempotent
    assert len(c.samples) == 3

    again = cohort.Cohort.load(c.save())
    assert {s.key() for s in again.samples} == {whole.key(), s1.key(), s2.key()}
    got = again.find(s1.key())
    assert got.region == "ROI 1" and got.name == "slide · ROI 1"


def test_batch_table_reads_region_scopes(home):
    # one slide, two regions differing at a marker m/z; the cohort reads each region's
    # saved feature scope, not the slide-level peaks
    p = _write_session_scopes(
        "/d/slide.imzML", peaks=[_peak(700.0, 0.5)],          # slide-level: unused by regions
        scopes={"ROI 1": [_peak(885.5, 0.2)], "ROI 2": [_peak(885.5, 0.9)]}, fp="slide")
    r1 = cohort.SampleRef(name="slide · ROI 1", session_path=p, region="ROI 1", group="a")
    r2 = cohort.SampleRef(name="slide · ROI 2", session_path=p, region="ROI 2", group="b")
    tbl = cohort.batch_feature_table([r1, r2], [885.5])
    col = [c for c in tbl.columns if c != "group"][0]
    assert tbl.loc["slide · ROI 1", col] == pytest.approx(0.2)
    assert tbl.loc["slide · ROI 2", col] == pytest.approx(0.9)

    # a region whose scope is gone (renamed/never built) is *skipped and reported* — not a
    # phantom all-NaN row that would silently inflate the compared-sample count (and no crash)
    ghost = cohort.SampleRef(name="slide · ghost", session_path=p, region="ghost")
    tbl2 = cohort.batch_feature_table([r1, ghost], [885.5])
    assert list(tbl2.index) == ["slide · ROI 1"]
    assert tbl2.loc["slide · ROI 1", col] == pytest.approx(0.2)
    assert any(s["name"] == "slide · ghost" and "no saved peaks" in s["reason"]
               for s in tbl2.attrs["skipped"])


def test_region_samples_as_replicates_in_group_comparison(home):
    # two slides, each split into two regions; the regions are the replicates. Marker up
    # in the treated slide's regions.
    t = 750.0
    p1 = _write_session_scopes("/d/s1.imzML", [_peak(t, 0.1)],
                               {"L": [_peak(t, 0.10)], "R": [_peak(t, 0.12)]}, fp="s1")
    p2 = _write_session_scopes("/d/s2.imzML", [_peak(t, 0.1)],
                               {"L": [_peak(t, 0.80)], "R": [_peak(t, 0.85)]}, fp="s2")
    refs = [cohort.SampleRef(name="s1·L", session_path=p1, region="L", group="control"),
            cohort.SampleRef(name="s1·R", session_path=p1, region="R", group="control"),
            cohort.SampleRef(name="s2·L", session_path=p2, region="L", group="treated"),
            cohort.SampleRef(name="s2·R", session_path=p2, region="R", group="treated")]
    tbl = cohort.batch_feature_table(refs, [t])
    assert len(tbl) == 4
    res = cohort.group_comparison(tbl, "control", "treated")
    row = res[res["mz"] == t].iloc[0]
    assert row["n_A"] == 2 and row["n_B"] == 2       # two regions per group
    assert row["log2_fc"] > 1.0                       # up in treated


def test_consensus_targets_reads_region_scopes(home):
    # consensus axis is built from region scopes when refs are region samples
    p = _write_session_scopes("/d/slide.imzML", [_peak(700.0, 0.5)],
                              {"L": [_peak(885.5, 0.5)], "R": [_peak(885.51, 0.6)]}, fp="slide")
    refs = [cohort.SampleRef(name="L", session_path=p, region="L"),
            cohort.SampleRef(name="R", session_path=p, region="R")]
    targets = cohort.consensus_targets(refs, tol_ppm=30.0, min_prevalence=0.0)
    assert any(abs(t - 885.505) < 0.02 for t in targets)
    assert not any(abs(t - 700.0) < 0.02 for t in targets)   # slide-level peak isn't read


def test_add_idempotent_on_source(home):
    # the duplication bug: a slide whose fingerprint drifted got a *new* managed session
    # path on each load. Keying identity on the stable source path keeps add() idempotent
    # regardless of the differing session paths.
    src = "/data/slideX.imzML"
    c = cohort.Cohort(name="W")
    c.add(cohort.SampleRef(name="x", session_path="/sessions/x__aaa.json", source=src))
    c.add(cohort.SampleRef(name="x", session_path="/sessions/x__bbb.json", source=src))
    assert len(c.samples) == 1
    # a *region* sample of the same slide still coexists (key folds in the region name)
    c.add(cohort.SampleRef(name="x·ROI", session_path="/sessions/x__ccc.json",
                           source=src, region="ROI"))
    assert len(c.samples) == 2


def test_dedupe_collapses_same_source_keeping_newest(home):
    # repair a roster minted before the fix: three entries for one slide, differing only by
    # session path / fingerprint, collapse to one — the newest managed session wins and any
    # group label is carried over.
    import os as _os
    src = "/data/slide.imzML"
    p_old = _write_session(src, [_peak(700.0, 0.5)], fp="good")
    p_new = _write_session(src, [_peak(700.0, 0.5)], fp="drifted")
    assert p_old != p_new                                  # different fp → different file
    _os.utime(p_old, (1000, 1000))
    _os.utime(p_new, (2000, 2000))                         # newer
    c = cohort.Cohort(name="W")
    c.samples = [
        cohort.SampleRef(name="slide", session_path=p_old, source=src, group="control"),
        cohort.SampleRef(name="slide", session_path=p_new, source=src),
    ]
    assert c.samples[0].key() == c.samples[1].key()        # same identity (source)
    removed = c.dedupe()
    assert removed == 1
    assert len(c.samples) == 1
    kept = c.samples[0]
    assert kept.session_path == p_new                      # newest managed session wins
    assert kept.group == "control"                         # group carried over from the dup


def test_key_falls_back_to_session_path_without_source():
    # back-compat: whole-slide refs that predate the source field still key on session path
    a = cohort.SampleRef(name="a", session_path="/x/a.json")
    b = cohort.SampleRef(name="b", session_path="/x/b.json")
    assert a.key() != b.key()
    assert a.key() == "/x/a.json"


def test_home_dir_defaults_to_smile_msi(tmp_path, monkeypatch):
    # with no override, the store lives at ~/.smile-msi
    monkeypatch.delenv("SMILE_MSI_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from smile_msi import library
    assert library.home_dir() == str(tmp_path / ".smile-msi")


def test_home_dir_env_override_honored(tmp_path, monkeypatch):
    # an explicit override (used by tests and power users) is honoured verbatim
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "custom"))
    monkeypatch.setenv("HOME", str(tmp_path))
    from smile_msi import library
    assert library.home_dir() == str(tmp_path / "custom")


def test_cross_slide_normalization():
    """Per-sample normalization equalizes slide-to-slide total signal before group stats."""
    import numpy as np
    from smile_msi import cohort
    X = np.array([[10., 20., 30.], [100., 200., 300.], [5., np.nan, 15.]])  # row2 = 10x row1
    none = cohort._normalize_samples(X.copy(), "none")
    assert np.allclose(none, X, equal_nan=True)
    med = cohort._normalize_samples(X.copy(), "median")
    meds = np.nanmedian(np.where(med > 0, med, np.nan), axis=1)
    assert np.allclose(meds, meds[0])                      # all samples share a median
    tic = cohort._normalize_samples(X.copy(), "sum")
    assert np.isfinite(tic[~np.isnan(X)]).all()


def test_match_values_equivalent_to_scalar_match():
    """Vectorized _match_values must be bit-for-bit equal to per-target _match — including
    the boundary cases and the prefer-lower-index-on-tie rule."""
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(200, 1200, size=400))
    val = rng.uniform(0, 1, size=mz.size)
    targets = np.concatenate([mz + rng.uniform(-0.02, 0.02, mz.size),   # near real peaks
                              rng.uniform(200, 1200, 100),               # random
                              [mz[0] - 5, mz[-1] + 5, 0.0]])             # boundaries + invalid
    for tol in (5.0, 20.0, 50.0):
        vec = cohort._match_values(mz, val, targets, tol)
        for k, t in enumerate(targets):
            j = cohort._match(mz, float(t), tol)
            exp = val[j] if j >= 0 else np.nan
            assert (np.isnan(vec[k]) and np.isnan(exp)) or vec[k] == exp


def test_tested_only_fdr_and_detection_gate(home):
    # A sparse ion detected in only ONE sample of a group must NOT be tested (q=NaN), and
    # must be excluded from the FDR denominator so it can't inflate a real feature's q.
    t_real, t_sparse, t_flat = 750.0, 600.0, 500.0
    refs = []
    for i, rel in enumerate([0.10, 0.12, 0.11, 0.10]):
        peaks = [_peak(t_real, rel), _peak(t_flat, 0.5)]
        if i == 0:
            peaks.append(_peak(t_sparse, 0.9))      # sparse: present in just one control
        p = _write_session(f"/d/ctrl{i}.imzML", peaks, fp=f"c{i}")
        refs.append(cohort.ref_from_session(p, group="control"))
    for i, rel in enumerate([0.80, 0.85, 0.78, 0.82]):
        p = _write_session(f"/d/syn{i}.imzML", [_peak(t_real, rel), _peak(t_flat, 0.5)], fp=f"s{i}")
        refs.append(cohort.ref_from_session(p, group="treated"))

    tbl = cohort.batch_feature_table(refs, [t_real, t_sparse, t_flat])
    res = cohort.group_comparison(tbl, "control", "treated")
    sparse_row = res[res["mz"] == t_sparse].iloc[0]
    assert sparse_row["n_A"] == 1 and np.isnan(sparse_row["p_value"])   # gated out, not tested
    assert np.isnan(sparse_row["q_value"])
    assert res.attrs["n_testable"] == 2                                  # real + flat only
    # the discriminating feature's q uses the tested-only denominator (2), NOT 3 — so the
    # sparse, untestable feature can't dilute it. BH: real has the smallest p of 2 tested →
    # q = p · 2/1; had the sparse column been (wrongly) counted, it would be p · 3/1.
    real_row = res[res["mz"] == t_real].iloc[0]
    assert real_row["q_value"] == pytest.approx(real_row["p_value"] * 2, rel=1e-6)
    assert real_row["q_value"] < real_row["p_value"] * 3


def test_group_comparison_warns_when_nothing_testable(home):
    # one sample per group → MWU can't run; result is all-NaN p/q with a clear warning,
    # not a clean '0 significant' negative
    pa = _write_session("/d/a.imzML", [_peak(700.0, 0.5)], fp="a")
    pb = _write_session("/d/b.imzML", [_peak(700.0, 0.9)], fp="b")
    refs = [cohort.ref_from_session(pa, group="control"),
            cohort.ref_from_session(pb, group="treated")]
    tbl = cohort.batch_feature_table(refs, [700.0])
    res = cohort.group_comparison(tbl, "control", "treated")
    assert res.attrs["n_testable"] == 0
    assert "warning" in res.attrs
    assert res["p_value"].isna().all() and res["q_value"].isna().all()


def test_batch_table_missing_zero_fill_and_skip_reporting(home):
    # missing='zero' fills absent ions with 0 (below-detection), removing presence bias;
    # a corrupt/missing session is skipped and reported, not silently dropped from the count
    pa = _write_session("/d/a.imzML", [_peak(700.0, 0.5)], fp="a")        # has 700, lacks 800
    pb = _write_session("/d/b.imzML", [_peak(800.0, 0.5)], fp="b")        # has 800, lacks 700
    refs = [cohort.ref_from_session(pa, group="g"),
            cohort.ref_from_session(pb, group="g"),
            cohort.SampleRef(name="gone", session_path="/d/missing.json", source="/d/m.imzML")]
    tbl = cohort.batch_feature_table(refs, [700.0, 800.0], missing="zero")
    assert len(tbl) == 2                                                   # the missing one isn't a row
    cols = [c for c in tbl.columns if c != "group"]
    assert tbl.loc["a.imzML", cols[1]] == 0.0                             # absent 800 → 0, not NaN
    assert any(s["name"] == "gone" and "missing/corrupt" in s["reason"]
               for s in tbl.attrs["skipped"])


def test_shared_sessions_cache_reused(home, monkeypatch):
    # consensus_targets + batch_feature_table share one sessions dict → each JSON parsed once
    for i in range(3):
        _write_session(f"/d/s{i}.imzML", [_peak(700.0 + i, 0.5)], fp=f"s{i}")
    refs = cohort.discover_samples()
    calls = {"n": 0}
    real = session.load_session
    monkeypatch.setattr(session, "load_session",
                        lambda p, *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(p, *a, **k))[1])
    shared: dict = {}
    cohort.consensus_targets(refs, sessions=shared)
    after_consensus = calls["n"]
    cohort.batch_feature_table(refs, [700.0], sessions=shared)
    assert calls["n"] == after_consensus            # batch reused the cache: no extra loads


def test_section_loader_bounds_resident_cubes(home, monkeypatch, tmp_path):
    # the OOM fix: primed with the full roster, the loader evicts each cube the moment its
    # last sample is consumed, so resident RAM never exceeds ~one cube — even across N slides
    from smile_msi import msi
    live = {"n": 0}

    class FakeDS:
        def __init__(self, src):
            self.source = src; self.released = False
        def to_ram(self):
            live["n"] += 1
        def prime(self):
            pass
        def release(self):
            if not self.released:
                live["n"] -= 1; self.released = True

    def fake_from_imzml(path, lazy=True, **k):
        return FakeDS(path)
    monkeypatch.setattr(msi.MSIDataset, "from_imzml", staticmethod(fake_from_imzml))

    srcs = []
    for i in range(4):
        f = tmp_path / f"s{i}.imzML"; f.write_text(""); srcs.append(str(f))
    refs = [cohort.SampleRef(name=f"s{i}", session_path="", source=srcs[i]) for i in range(4)]

    loader = cohort.SectionLoader()
    loader.prime(refs)
    peak = 0
    for r in refs:
        loader.load(r)
        peak = max(peak, live["n"])
    assert peak == 1                       # never more than one cube resident at a time
    loader.close()
    assert live["n"] == 0                   # every cube evicted

    # two region-samples that SHARE a source load the cube once and free it after the second
    region_refs = [cohort.SampleRef(name="r1", session_path="", source=srcs[0], region=""),
                   cohort.SampleRef(name="r2", session_path="", source=srcs[0], region="")]
    loader2 = cohort.SectionLoader()
    loader2.prime(region_refs)
    loader2.load(region_refs[0]); assert live["n"] == 1
    loader2.load(region_refs[1]); assert live["n"] == 1     # reused, not reloaded
    loader2.close(); assert live["n"] == 0


def test_section_loader_reconstructs_cluster_backed_region(home, monkeypatch, tmp_path):
    # a region defined by segmentation clusters saves an EMPTY mask; the loader must rebuild
    # its pixels from `segments` over the session's labels, else it's silently dropped from
    # the embedding while the group comparison (feature_scopes) keeps it (rank 6)
    from smile_msi import msi
    labels = [0, 1, 2, 1, 0, 2]                       # 6 pixels; region 'Nerve' = clusters {1,2}
    src = str(tmp_path / "slide.imzML"); open(src, "w").close()
    sess = session.build_session(
        source=src, settings={}, peaks=[_peak(700.0, 0.5)], n_pixels=6, dataset_fingerprint="fp",
        labels=labels, n_clusters=3,
        named_regions=[{"name": "Nerve", "color": "#fff", "segments": [1, 2], "mask": None}])
    sp = session.save_session(session.managed_path(src, "fp"), sess)

    class FakeDS:
        n_pixels = 6
        def to_ram(self): pass
        def prime(self): pass
    monkeypatch.setattr(msi.MSIDataset, "from_imzml",
                        staticmethod(lambda path, lazy=True, **k: FakeDS()))

    ref = cohort.SampleRef(name="Nerve", session_path=sp, source=src, region="Nerve")
    loader = cohort.SectionLoader()
    got = loader.load(ref)
    assert got is not None
    _ds, pix = got
    assert sorted(pix.tolist()) == [1, 2, 3, 5]       # pixels whose label ∈ {1, 2}


def test_section_loader_reuses_cube_sidecar(home, tmp_path):
    """A cohort member with a fingerprint-matching .cache.npz sidecar reuses its cube +
    prime stats instead of recomputing — so SectionLoader skips the prime pass and a lazy
    ion render gets a ready cube (rank-4 cohort fast-path). Mismatched fingerprints fall
    back cleanly."""
    from smile_msi import demo, library
    from smile_msi.msi import MSIDataset

    src = str(tmp_path / "sec.imzML")
    demo.write_synthetic_imzml(src, width=10, height=8, seed=3)
    ds = MSIDataset.from_imzml(src); ds.prime(); ds.build_mz_cube()
    fp = library.dataset_fingerprint(ds)
    sp = session.managed_path(src, fp)
    session.save_session(sp, session.build_session(
        source=src, settings={}, peaks=[_peak(700.0, 0.5)],
        n_pixels=ds.n_pixels, dataset_fingerprint=fp))
    session.save_cube(sp, ds._cube, mean=ds._mean, pix=ds._pix, fingerprint=fp)

    ref = cohort.SampleRef(name="sec", session_path=sp, source=src,
                           fingerprint=fp, n_pixels=ds.n_pixels)
    loader = cohort.SectionLoader(dense=False)        # lazy: no to_ram/prime of its own
    out = loader.load(ref)
    assert out is not None
    got, _ = out
    assert got._cube is not None                      # cube reused from the sidecar
    assert got._mean is not None and got._pix is not None   # prime pass skipped

    # a fingerprint mismatch (wrong key) must NOT reuse — falls back to a fresh lazy ds
    bad = cohort.SampleRef(name="sec2", session_path=session.managed_path(src, "WRONGFP"),
                           source=src)
    loader2 = cohort.SectionLoader(dense=False)
    got2, _ = loader2.load(bad)
    assert got2._cube is None


def test_consensus_width_cap_no_chaining(home):
    # a dense run of peaks each within tol of its neighbour must NOT chain into one giant
    # cluster spanning many × tol — the width cap splits them
    peaks = [_peak(m, 0.5) for m in (700.000, 700.010, 700.020, 700.030, 700.040)]
    p = _write_session("/d/dense.imzML", peaks, fp="d")
    refs = [cohort.ref_from_session(p)]
    # neighbour gaps ≈14 ppm (<20) but the run spans ~57 ppm end-to-end; width cap forbids one cluster
    targets = cohort.consensus_targets(refs, tol_ppm=20.0, min_prevalence=0.0)
    assert len(targets) >= 2                          # not fused into a single centroid


# --------------------------------------------------------------------------- #
# relocation — healing stale paths after a cohort is moved / copied
# --------------------------------------------------------------------------- #
def test_missing_source_samples_flags_only_moved_imzml(tmp_path):
    present = tmp_path / "here.imzML"
    present.write_text("x")
    samples = [
        cohort.SampleRef(name="present", session_path="", source=str(present)),
        cohort.SampleRef(name="moved", session_path="", source=r"C:\old\gone.imzML"),
        cohort.SampleRef(name="synthetic", session_path="", source=""),                 # not flagged
        cohort.SampleRef(name="csv", session_path="", source=str(tmp_path / "x.csv")),   # not imzML
    ]
    assert [m.name for m in cohort.missing_source_samples(samples)] == ["moved"]
    assert cohort.source_exists(samples[0])
    assert not any(cohort.source_exists(s) for s in samples[1:])   # moved + synthetic + csv


def test_relocate_samples_by_basename(tmp_path):
    newroot = tmp_path / "moved"
    (newroot / "sub").mkdir(parents=True)
    a = newroot / "sub" / "slideA.imzML"; a.write_text("x")       # nested → found by walk
    b = newroot / "slideB.imzML"; b.write_text("x")
    samples = [
        cohort.SampleRef(name="A", session_path="", source=r"C:\old\data\slideA.imzML"),
        cohort.SampleRef(name="B", session_path="", source=r"C:\old\data\slideB.imzML"),
        cohort.SampleRef(name="C", session_path="", source=r"C:\old\data\missing.imzML"),
    ]
    report = cohort.relocate_samples(samples, [str(newroot)])
    assert {n for n, _o, _new in report.fixed} == {"A", "B"}
    assert samples[0].source == str(a)
    assert samples[1].source == str(b)
    assert [n for n, _src in report.unresolved] == ["C"]
    assert samples[2].source == r"C:\old\data\missing.imzML"      # untouched when not found
    assert cohort.missing_source_samples(samples) == [samples[2]]


def test_relocate_samples_ambiguous_is_left_untouched(tmp_path):
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True); (root / "b").mkdir()
    (root / "a" / "dup.imzML").write_text("x")
    (root / "b" / "dup.imzML").write_text("x")                    # same basename, two folders
    s = cohort.SampleRef(name="D", session_path="", source=r"C:\old\dup.imzML")
    report = cohort.relocate_samples([s], [str(root)])
    assert not report.fixed
    assert report.ambiguous == [("D", "dup.imzML", 2)]
    assert s.source == r"C:\old\dup.imzML"                        # never guesses


def test_key_is_separator_and_case_agnostic():
    import os as _os
    a = cohort.SampleRef(name="A", session_path="", source="C:/data/slide.imzML")
    b = cohort.SampleRef(name="B", session_path="", source="C:\\data\\slide.imzML")
    assert a.key() == b.key()                        # \ and / unify on every host
    if _os.name == "nt":                             # casing folds only on Windows
        assert cohort.SampleRef(name="", session_path="", source="C:\\D\\S.imzML").key() == \
               cohort.SampleRef(name="", session_path="", source="c:\\d\\s.imzml").key()
    assert cohort.SampleRef(name="", session_path="", source="/x/y", region="ROI 1").key().endswith("::ROI 1")


def test_folder_is_separator_agnostic():
    assert cohort.SampleRef(name="", session_path="", source="C:\\runs\\batch7\\slide.imzML").folder() == "batch7"
    assert cohort.SampleRef(name="", session_path="", source="/runs/batch7/slide.imzML").folder() == "batch7"


def test_dedupe_carries_all_labels_from_dropped(tmp_path):
    older = tmp_path / "old.json"; older.write_text("{}")
    newer = tmp_path / "new.json"; newer.write_text("{}")
    os.utime(older, (1000, 1000)); os.utime(newer, (2000, 2000))   # newer mtime wins
    src = "/data/slide.imzML"
    labelled = cohort.SampleRef(name="A", source=src, session_path=str(older),
                                batch="b2", meta={"age": "71"}, region_n_pixels=50)
    blank = cohort.SampleRef(name="B", source=src, session_path=str(newer))
    c = cohort.Cohort(name="X", samples=[labelled, blank])
    assert labelled.key() == blank.key()
    assert c.dedupe() == 1
    keep = c.samples[0]
    assert keep.session_path == str(newer)           # newer-mtime ref kept
    assert keep.batch == "b2"                         # but the dropped ref's labels survive
    assert keep.meta == {"age": "71"}
    assert keep.region_n_pixels == 50


def test_relocate_refuses_basename_collision(tmp_path):
    root = tmp_path / "data"; root.mkdir()
    (root / "slide.imzML").write_text("x")           # exactly ONE slide.imzML on the new machine
    s1 = cohort.SampleRef(name="runA", session_path="", source=r"C:\runA\slide.imzML")
    s2 = cohort.SampleRef(name="runB", session_path="", source=r"C:\runB\slide.imzML")
    report = cohort.relocate_samples([s1, s2], [str(root)])
    assert not report.fixed                           # neither relocated — basename can't disambiguate
    assert {a[0] for a in report.ambiguous} == {"runA", "runB"}
    assert s1.source == r"C:\runA\slide.imzML" and s2.source == r"C:\runB\slide.imzML"  # untouched
    # but a basename needed by exactly ONE sample still relocates cleanly
    s3 = cohort.SampleRef(name="solo", session_path="", source=r"C:\old\slide.imzML")
    rep2 = cohort.relocate_samples([s3], [str(root)])
    assert [n for n, _o, _n in rep2.fixed] == ["solo"]


def test_unloadable_flags_region_with_missing_session(tmp_path):
    raw = tmp_path / "slide.imzML"; raw.write_text("x")
    whole = cohort.SampleRef(name="W", session_path="", source=str(raw))
    region = cohort.SampleRef(name="R", source=str(raw),
                              session_path=r"C:\old\s.json", region="ROI 1")
    bad_src = cohort.SampleRef(name="X", session_path="", source=r"C:\old\gone.imzML")
    problems = {r.name: why for r, why in cohort.unloadable_samples([whole, region, bad_src])}
    assert "W" not in problems                        # whole slide with present imzML loads
    assert "session file not found" in problems["R"]  # region needs its session, which is gone
    assert "source file not found" in problems["X"]


def test_loader_skips_wrong_slide_by_pixel_count(home, tmp_path):
    from smile_msi import demo
    src = str(tmp_path / "real.imzML")
    demo.write_synthetic_imzml(src, width=10, height=8, seed=1)    # 80 pixels
    wrong = cohort.SampleRef(name="x", session_path="", source=src, n_pixels=999)   # roster expects a different slide
    assert cohort.SectionLoader(dense=False).load(wrong) is None   # identity guard skips it
    right = cohort.SampleRef(name="x", session_path="", source=src, n_pixels=80)
    assert cohort.SectionLoader(dense=False).load(right) is not None


def test_relocate_repoints_session_path_and_rewrites_session_source(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    newroot = tmp_path / "data"; newroot.mkdir()
    raw = newroot / "slide.imzML"; raw.write_text("x")
    # a managed session exists locally (under the new home), but the cohort still points its
    # session_path at the OLD machine's location
    local_sess = os.path.join(session.sessions_dir(), "slide.json")
    session.save_session(local_sess, {"source": r"C:\old\slide.imzML", "n_pixels": 1})
    s = cohort.SampleRef(name="E",
                         session_path=r"C:\old\.smile-msi\sessions\slide.json",
                         source=r"C:\old\slide.imzML")
    report = cohort.relocate_samples([s], [str(newroot)], fix_sessions=True)
    assert s.source == str(raw)
    assert s.session_path == local_sess                          # repointed to this machine
    assert report.repointed_sessions == 1
    assert session.load_session(local_sess)["source"] == str(raw)  # session JSON healed too


def test_find_moved_source_requires_unique_match_with_ibd(tmp_path):
    from smile_msi.cohort import find_moved_source
    a = tmp_path / "a"; b = tmp_path / "b"; c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "slide.imzML").write_bytes(b"x"); (a / "slide.ibd").write_bytes(b"y")
    (c / "slide.imzML").write_bytes(b"x")                    # no .ibd → not a usable copy
    missing = str(tmp_path / "old" / "slide.imzML")
    assert find_moved_source(missing, [str(a), str(b), str(c), "", str(tmp_path / "nope")]) == str(a / "slide.imzML")
    assert find_moved_source(missing, [str(b)]) is None
    (b / "slide.imzML").write_bytes(b"x"); (b / "slide.ibd").write_bytes(b"y")
    assert find_moved_source(missing, [str(a), str(b)]) is None    # ambiguous → never guess
    assert find_moved_source(missing, [str(a), str(a)]) == str(a / "slide.imzML")  # same root twice
