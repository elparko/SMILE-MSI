"""Cohort-level nested designs — ``pseudobulk_table`` / ``nested_design`` / ``nested_comparison``.

These exercise the wiring, not the statistics (see ``test_hierstats.py`` for those): the
subject/compartment derivation, the transform, and — the one that would silently corrupt every
result — that the pixel summary is normalized against the **whole slide**, never against the
region's own pixels.
"""
import numpy as np
import pandas as pd
import pytest

from smile_msi import cohort

COMPARTMENTS = ["endoneurium", "perineurium", "epineurium"]


# --------------------------------------------------------------------------- #
# pseudobulk_table — pixel-level summaries
# --------------------------------------------------------------------------- #
class _StubDS:
    """Minimal MSIDataset stand-in: a raw (pixel × target) block + per-pixel TIC.

    ``norm_factors`` reproduces the real ``_finalize_norm`` contract — divide the per-pixel TIC
    by the mean TIC over ``scope`` (the whole slide when ``scope`` is None) — so the tests can
    tell a whole-slide scale from an on-tissue one.
    """

    def __init__(self, raw, tic):
        self._raw = np.asarray(raw, dtype=float)
        self._tic = np.asarray(tic, dtype=float)
        self.n_pixels = self._raw.shape[0]
        self.norm_requested = []          # every `norm=` features_for_rows was called with
        self.factor_calls = []            # every (method, scope-or-None) norm_factors saw

    def features_for_rows(self, targets, rows, tol_ppm=20.0, reduce="sum", norm="none"):
        self.norm_requested.append(norm)
        return self._raw[np.asarray(rows, dtype=int)]

    def norm_factors(self, method, scope=None):
        self.factor_calls.append((method, None if scope is None else tuple(np.asarray(scope))))
        src = self._tic if scope is None else self._tic[np.asarray(scope, dtype=int)]
        pos = src[src > 0]
        scale = float(pos.mean()) if pos.size else 1.0
        return self._tic / scale


class _StubLoader:
    def __init__(self, mapping):
        self._mapping = mapping           # ref.name -> (ds, pix) or None

    def load(self, ref):
        return self._mapping.get(ref.name)


def _ref(name, group="", region="", subject=None):
    meta = {} if subject is None else {"subject": subject}
    return cohort.SampleRef(name=name, session_path=f"/x/{name}.json",
                            source=f"/d/{name}.imzML", group=group, region=region, meta=meta)


def test_pseudobulk_table_never_lets_the_streamed_path_normalize():
    """The TRUST_AUDIT Confounder-#1 trap: `features_for_rows(norm=...)` on a lazy dataset
    scales by the requested rows' own mean, which would divide each compartment by its own
    mean TIC and erase the between-compartment signal. We must extract raw and apply
    slide-wide factors ourselves."""
    raw = np.array([[1.0], [1.0], [1.0], [10.0], [10.0], [10.0]])
    ds = _StubDS(raw, np.ones(6))
    r = _ref("slideA", group="facial", region="epineurium", subject="n1")
    loader = _StubLoader({"slideA": (ds, np.array([3, 4, 5]))})

    tbl = cohort.pseudobulk_table([r], [700.0], norm="tic", summary="median", loader=loader,
                                  norm_scope="slide")
    assert ds.norm_requested == ["none"]               # raw extraction, always
    assert ds.factor_calls == [("tic", None)]          # whole-slide scale
    assert tbl.iloc[0, 1] == pytest.approx(10.0)
    assert tbl.attrs["n_pixels"] == [3]
    assert tbl.attrs["norm_scope"] == "slide"


def test_pseudobulk_table_tissue_scope_is_immune_to_background(monkeypatch):
    """The whole-slide mean TIC drifts with how much empty matrix a section sits in, injecting
    a per-slide multiplicative bias into between-subject contrasts. Scoping the scale to the
    on-tissue pixels removes it: the same tissue on a mostly-background slide must summarize
    to the same value as on a tightly-cropped one."""
    tissue_pix = np.array([0, 1])
    # identical tissue (TIC 100), but slide B carries 6 background pixels instead of 2
    raw_a = np.array([[10.0], [10.0], [0.1], [0.1]])
    tic_a = np.array([100.0, 100.0, 1.0, 1.0])
    raw_b = np.vstack([np.array([[10.0], [10.0]]), np.full((6, 1), 0.1)])
    tic_b = np.concatenate([[100.0, 100.0], np.full(6, 1.0)])

    def run(scope, raw, tic, name):
        ds = _StubDS(raw, tic)
        r = _ref(name, group="g", region="endo", subject="n")
        monkeypatch.setattr(cohort, "_scope_rows_by_source",
                            lambda refs, sessions=None: {f"/d/{name}.imzML": tissue_pix})
        return cohort.pseudobulk_table([r], [700.0], norm="tic", summary="median",
                                       loader=_StubLoader({name: (ds, tissue_pix)}),
                                       norm_scope=scope).iloc[0, 1]

    a_t, b_t = run("tissue", raw_a, tic_a, "A"), run("tissue", raw_b, tic_b, "B")
    assert a_t == pytest.approx(b_t)                   # background-independent
    assert a_t == pytest.approx(10.0)                  # TIC 100 / mean-on-tissue 100

    # ...and the whole-slide scale is not: B's extra background drags its mean TIC from
    # (100+100+1+1)/4 = 50.5 down to (200+6)/8 = 25.75, shrinking every normalized intensity
    # on that slide by the same factor. Between-subject contrasts inherit that bias.
    a_s, b_s = run("slide", raw_a, tic_a, "A"), run("slide", raw_b, tic_b, "B")
    assert a_s == pytest.approx(10.0 * 50.5 / 100.0)
    assert b_s == pytest.approx(10.0 * 25.75 / 100.0)
    assert b_s / a_s == pytest.approx(25.75 / 50.5, rel=1e-6)


def test_pseudobulk_table_rejects_an_unknown_norm_scope():
    with pytest.raises(ValueError, match="unknown norm_scope"):
        cohort.pseudobulk_table([], [700.0], norm_scope="tissue-ish")


# --------------------------------------------------------------------------- #
# unique feature columns — two m/z that agree to 4 dp must not share a column
# --------------------------------------------------------------------------- #
def test_feature_columns_disambiguate_targets_that_round_together():
    cols = cohort.feature_columns([700.12345, 700.12347, 800.0])
    assert cols == ["mz_700.1235", "mz_700.1235#2", "mz_800.0000"]
    assert len(set(cols)) == 3
    assert cohort.targets_from_columns(cols) == [700.1235, 700.1235, 800.0]


def test_colliding_targets_do_not_double_the_group_n():
    """A duplicate column name made `table[feat_cols]` return extra columns, which silently
    doubled n_A/n_B for that feature in group_comparison — pseudoreplication from a format
    string — and crashed the nested path with an opaque pandas length error."""
    targets = [700.12345, 700.12347, 800.0]
    cols = cohort.feature_columns(targets)
    rows = [{"group": "a" if i < 3 else "b", **{c: 1.0 + i for c in cols}} for i in range(6)]
    tbl = pd.DataFrame(rows, index=[f"s{i}" for i in range(6)], columns=["group", *cols])
    tbl.attrs["targets"] = targets

    res = cohort.group_comparison(tbl, "a", "b")
    assert list(res["n_A"]) == [3, 3, 3]
    assert list(res["n_B"]) == [3, 3, 3]
    assert sorted(res["mz"]) == sorted(targets)          # the exact m/z, not the rounded ones


def _duplicated_columns_table():
    """The shape a naive `f"mz_{t:.4f}"` used to produce: two targets, one column name."""
    tbl = pd.DataFrame([[g, 1.0, 2.0] for g in ("a", "a", "b", "b")],
                       index=[f"s{i}" for i in range(4)],
                       columns=["group", "mz_700.1235", "mz_700.1235"])
    tbl.attrs["targets"] = [700.12345, 700.12347]
    return tbl


def test_group_comparison_refuses_a_table_whose_columns_are_not_unique():
    with pytest.raises(ValueError, match="not unique"):
        cohort.group_comparison(_duplicated_columns_table(), "a", "b")


def test_nested_comparison_refuses_a_table_whose_columns_are_not_unique():
    tbl = _duplicated_columns_table()
    tbl.attrs["refs"] = [_ref(f"s{i}", group=("a" if i < 2 else "b"),
                              region="endo", subject=f"n{i}") for i in range(4)]
    with pytest.raises(ValueError, match="not unique"):
        cohort.nested_comparison(tbl, "a", "b", subject_by="subject")


def test_hierstats_refuses_ids_that_do_not_match_the_matrix():
    from smile_msi import hierstats as hs

    X = np.zeros((6, 3))
    g = np.array(["a"] * 3 + ["b"] * 3)
    c = np.array(["endo"] * 6)
    s = np.array([f"n{i}" for i in range(6)])
    with pytest.raises(ValueError, match="ids has 2 entries but X has 3"):
        hs.nested_mixed_model(X, g, c, s, "a", "b", ids=[1.0, 2.0])
    with pytest.raises(ValueError, match="ids has 2 entries but X has 3"):
        hs.stratified_comparison(X, g, c, s, "a", "b", ids=[1.0, 2.0])


def test_subject_id_from_slide_or_metadata():
    r = _ref("s01", subject="N-07")
    assert cohort.subject_id(r, "subject") == "n-07"
    # the usual nested MSI design: one nerve per file, no metadata import needed
    assert cohort.subject_id(r, cohort.SUBJECT_BY_SLIDE) == "s01.imzml"
    assert cohort.subject_id(_ref("s01"), "subject") == ""    # absent key → unresolvable


def test_subject_id_from_the_region_name_when_nerves_share_a_slide():
    """The slide names nothing when several nerves were sectioned onto it; the ROI label does."""
    r = _ref("slide", region="S01 endo")
    assert cohort.subject_id(r, "region:first") == "s01"
    assert cohort.subject_id(_ref("slide", region="endo_S02"), "region:last") == "s02"
    assert cohort.subject_id(_ref("slide", region=""), "region:first") == ""
    # a single-token region names no nerve — 'endo' is a compartment, not a subject
    assert cohort.subject_id(_ref("slide", region="endo"), "region:first") == "endo"
    with pytest.raises(ValueError, match="unknown subject_by"):
        cohort.subject_id(r, "region:middle")


@pytest.mark.parametrize("region,mode,expect", [
    ("s01 endo", "exact", "s01 endo"),
    ("s01 endo", "last", "endo"),
    ("s01 endo", "first", "s01"),
    ("endo_s01", "last", "s01"),
    ("epi-S02", "first", "epi"),
    ("endo", "last", "endo"),
    ("", "last", ""),
])
def test_compartment_id_token_rules(region, mode, expect):
    assert cohort.compartment_id(_ref("x", region=region), mode) == expect


def test_compartment_id_from_a_saved_mapping():
    """The mapping the 'Define compartments…' dialog writes — the only source that copes with
    region names no token rule can parse."""
    r = cohort.SampleRef(name="x", session_path="/x/x.json", region="ROI 2",
                         meta={"compartment": "perineurium"})
    assert cohort.compartment_id(r, "meta:compartment") == "perineurium"
    assert cohort.compartment_id(r, "exact") == "ROI 2"
    # an unmapped sample reports no compartment; callers must exclude it, not invent one
    bare = cohort.SampleRef(name="y", session_path="/x/y.json", region="ROI 3")
    assert cohort.compartment_id(bare, "meta:compartment") == ""


def test_compartment_id_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown compartment_from"):
        cohort.compartment_id(_ref("x", region="a b"), "middle")


def test_pseudobulk_table_median_is_robust_to_hot_pixels_where_mean_is_not():
    raw = np.array([[1.0], [1.0], [1.0], [1.0], [1000.0]])   # one matrix hot-pixel
    ds = _StubDS(raw, np.ones(5))
    r = _ref("s", group="g", region="endoneurium", subject="n1")
    pix = np.arange(5)

    med = cohort.pseudobulk_table([r], [700.0], norm="none", summary="median",
                                  loader=_StubLoader({"s": (_StubDS(raw, np.ones(5)), pix)}))
    mean = cohort.pseudobulk_table([r], [700.0], norm="none", summary="mean",
                                   loader=_StubLoader({"s": (ds, pix)}))
    assert med.iloc[0, 1] == pytest.approx(1.0)
    assert mean.iloc[0, 1] == pytest.approx(200.8)
    assert med.attrs["summary"] == "median"
    assert "median" in med.attrs["value"]


def test_pseudobulk_table_skips_unloadable_and_empty_regions():
    ds = _StubDS(np.array([[5.0], [7.0]]), np.ones(2))
    refs = [_ref("ok", group="g", region="endo", subject="n1"),
            _ref("gone", group="g", region="endo", subject="n2"),
            _ref("empty", group="g", region="endo", subject="n3")]
    loader = _StubLoader({"ok": (ds, np.array([0, 1])), "gone": None,
                          "empty": (_StubDS(np.array([[1.0]]), np.ones(1)), np.array([], int))})
    tbl = cohort.pseudobulk_table(refs, [700.0], norm="none", loader=loader)
    assert list(tbl.index) == ["ok"]
    reasons = {s["name"]: s["reason"] for s in tbl.attrs["skipped"]}
    assert set(reasons) == {"gone", "empty"}
    assert "not resolvable" in reasons["gone"] and "no pixels" in reasons["empty"]
    assert len(tbl.attrs["refs"]) == 1


def test_pseudobulk_table_rejects_an_unknown_summary():
    with pytest.raises(ValueError, match="unknown summary"):
        cohort.pseudobulk_table([], [700.0], summary="mode")


# --------------------------------------------------------------------------- #
# nested_design
# --------------------------------------------------------------------------- #
def _nested_table(rng, n_a=3, n_b=6, n_feat=6, interaction=2.5):
    """A pseudobulk table: (n_a + n_b) subjects × 3 compartments, raw intensity scale.

    Baselines span ~3 orders of magnitude, as real ion intensities do. That matters: the
    τ pseudocount is the 5th percentile of the nonzero block, so a fixture with one flat
    baseline would put τ next to every value and compress the fold changes toward zero.
    """
    refs, rows, index = [], [], []
    targets = [700.0 + 1.0 * j for j in range(n_feat)]
    cols = [f"mz_{t:.4f}" for t in targets]
    base = np.geomspace(50.0, 20000.0, n_feat)
    base[0] = 5000.0                                      # the planted feature sits well above τ
    for i in range(n_a + n_b):
        subj = f"nerve{i:02d}"
        grp = "facial" if i < n_a else "synkinetic"
        off = rng.normal(1.0, 0.05)                       # multiplicative subject effect
        for comp in COMPARTMENTS:
            y = base * off * rng.normal(1.0, 0.03, n_feat)
            if grp == "synkinetic" and comp == "epineurium":
                y[0] *= 2.0 ** interaction                # interaction on feature 0
            refs.append(_ref(f"{subj}::{comp}", group=grp, region=comp, subject=subj))
            rows.append({"group": grp, **dict(zip(cols, y))})
            index.append(f"{subj}::{comp}")
    tbl = pd.DataFrame(rows, index=index, columns=["group", *cols])
    tbl.attrs["targets"] = targets
    tbl.attrs["refs"] = refs
    tbl.attrs["skipped"] = []
    tbl.attrs["summary"] = "median"
    return tbl


def test_nested_design_reads_subject_from_meta_and_compartment_from_region():
    tbl = _nested_table(np.random.default_rng(1))
    subject, compartment = cohort.nested_design(tbl, "subject")
    assert len(subject) == 27 and len(compartment) == 27
    assert set(compartment) == set(COMPARTMENTS)
    assert len(set(subject)) == 9
    assert subject[0] == "nerve00"                        # lower-cased meta value


def test_nested_design_refuses_rows_with_no_subject_key():
    tbl = _nested_table(np.random.default_rng(2))
    tbl.attrs["refs"][4].meta = {}                        # one compartment forgot its nerve
    with pytest.raises(ValueError, match="could not be traced to a subject"):
        cohort.nested_design(tbl, "subject")


def test_nested_design_collapses_per_nerve_region_names():
    """The failure the mapping/token rules exist for: region names that embed the nerve id make
    every compartment appear in exactly one subject."""
    tbl = _nested_table(np.random.default_rng(4))
    for r in tbl.attrs["refs"]:                           # 'endoneurium' → 'nerve00 endoneurium'
        r.region = f"{r.meta['subject']} {r.region}"
    _s, comp_exact = cohort.nested_design(tbl, "subject", compartment_from="exact")
    assert len(set(comp_exact)) == 27                     # one 'compartment' per nerve: broken
    _s, comp_last = cohort.nested_design(tbl, "subject", compartment_from="last")
    assert set(comp_last) == set(COMPARTMENTS)            # collapsed back onto 3


def test_nested_design_requires_row_aligned_refs():
    tbl = _nested_table(np.random.default_rng(3))
    tbl.attrs["refs"] = tbl.attrs["refs"][:5]
    with pytest.raises(ValueError, match="aligned to rows"):
        cohort.nested_design(tbl, "subject")


# --------------------------------------------------------------------------- #
# nested_comparison
# --------------------------------------------------------------------------- #
def test_nested_comparison_lmm_finds_the_interaction_and_stamps_provenance():
    tbl = _nested_table(np.random.default_rng(2026))
    res = cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                   model="lmm", compartments=COMPARTMENTS)
    assert res.attrs["model"] == "lmm"
    assert res.attrs["transform"] == "log2"
    assert res.attrs["tau"] > 0
    assert res.attrs["n_subjects_a"] == 3 and res.attrs["n_subjects_b"] == 6
    assert res.attrs["n_observations"] == 27
    assert res.attrs["subject_by"] == "subject"
    assert len(res.attrs["subjects_a"]) == 3
    assert res.attrs["compartments"] == COMPARTMENTS
    assert list(res["mz"])[:1] == [700.0]

    row = res[res["mz"] == 700.0].iloc[0]
    assert row["q_interaction"] <= 0.05
    assert row["p__epineurium"] < 0.05
    assert row["p__endoneurium"] > 0.05
    assert row["log2_fc__epineurium"] > 2.0
    # every other feature is null in the interaction family
    assert (res[res["mz"] != 700.0]["q_interaction"] > 0.05).all()


def test_nested_comparison_carries_the_small_n_feasibility_verdict():
    tbl = _nested_table(np.random.default_rng(5), n_feat=50)
    res = cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                   model="lmm", compartments=COMPARTMENTS)
    fea = res.attrs["feasibility"]
    assert fea["n_a"] == 3 and fea["n_b"] == 6 and fea["n_features"] == 50
    assert fea["feasible"] is False
    assert fea["p_floor"] == pytest.approx(2 / 84, abs=1e-4)
    assert "cannot return p below" in fea["reason"]


def test_nested_comparison_stratified_matches_the_lmm_on_the_planted_feature():
    tbl = _nested_table(np.random.default_rng(7), n_feat=20)
    strat = cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                     model="stratified", method="modt",
                                     compartments=COMPARTMENTS)
    assert strat.attrs["model"] == "stratified"
    row = strat[strat["mz"] == 700.0].iloc[0]
    assert row["q__epineurium"] <= 0.05
    assert row["q__endoneurium"] > 0.05
    per = strat.attrs["per_compartment"]["epineurium"]
    assert per["n_A"] == 3 and per["n_B"] == 6
    assert per["feasibility"]["feasible"] is False


def test_nested_comparison_log2_transform_is_applied_and_can_be_disabled():
    tbl = _nested_table(np.random.default_rng(9), n_feat=8)
    log = cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                   compartments=COMPARTMENTS, transform="log2")
    raw = cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                   compartments=COMPARTMENTS, transform="none")
    assert log.attrs["transform"] == "log2" and raw.attrs["transform"] == "none"
    # the model coefficient lives on the modelling scale, so it differs; the τ-regularized
    # log2 fold change is computed from raw means either way and must agree exactly.
    assert log["log2_fc__epineurium"].to_numpy() == pytest.approx(
        raw["log2_fc__epineurium"].to_numpy(), nan_ok=True)
    assert abs(log["effect__epineurium"].iloc[0]) < abs(raw["effect__epineurium"].iloc[0])


def test_nested_comparison_rejects_bad_arguments():
    tbl = _nested_table(np.random.default_rng(11))
    with pytest.raises(ValueError, match="must differ"):
        cohort.nested_comparison(tbl, "facial", "facial", subject_by="subject")
    with pytest.raises(ValueError, match="no samples labelled"):
        cohort.nested_comparison(tbl, "nope", "nada", subject_by="subject")
    with pytest.raises(ValueError, match="unknown model"):
        cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject", model="glm")
    with pytest.raises(ValueError, match="unknown transform"):
        cohort.nested_comparison(tbl, "facial", "synkinetic", subject_by="subject",
                                 transform="sqrt")
