"""Cohort-level lipid-class roll-up — :func:`smile_msi.cohort.nested_class_comparison`.

Wires a nested mixed-model result to :func:`smile_msi.hierstats.camera`: the per-ion
standardized effects become the statistic, and the residuals of the *same* design supply the
inter-ion correlation that decides how much those ions are really worth.
"""
import numpy as np
import pandas as pd
import pytest

from smile_msi import cohort

COMPARTMENTS = ["endo", "epi", "peri"]


def _ref(name, group, region, subject):
    return cohort.SampleRef(name=name, session_path=f"/x/{subject}.json",
                            source=f"/d/{subject}.imzML", group=group, region=region,
                            meta={"subject": subject})


def _cohort_with_a_class(rng, n_a=3, n_b=7, n_bg=40, m_cls=19, rho=0.6, endo_shift=-0.9):
    """10 nerves × 3 compartments. `m_cls` ions form a correlated class (a chain-length series:
    one shared myelin factor drives them all) that is depleted in group B's endo only."""
    subjects = [f"n{i:02d}" for i in range(n_a + n_b)]
    groups = ["normal"] * n_a + ["trt"] * n_b
    n_feat = m_cls + n_bg
    base = np.r_[np.full(m_cls, 11.0), rng.uniform(5.0, 13.0, n_bg)]   # log2 baselines
    refs, rows, index = [], [], []
    targets = [700.0 + 0.7 * j for j in range(n_feat)]
    cols = cohort.feature_columns(targets)
    for s, g in zip(subjects, groups):
        off = rng.normal(0.0, 0.35)                       # per-nerve level
        for comp in COMPARTMENTS:
            # one latent factor shared by the whole class → exchangeable correlation ≈ rho
            factor = rng.normal(size=1)
            y = base + off + rng.normal(0.0, 0.30, n_feat)
            y[:m_cls] += np.sqrt(rho) * factor * 0.8
            if g == "trt" and comp == "endo":
                y[:m_cls] += endo_shift                   # the class effect
            refs.append(_ref(f"{s} {comp}", g, comp, s))
            rows.append({"group": g, **dict(zip(cols, np.exp2(y)))})   # raw intensity scale
            index.append(f"{s} {comp}")
    tbl = pd.DataFrame(rows, index=index, columns=["group", *cols])
    tbl.attrs["targets"] = targets
    tbl.attrs["refs"] = refs
    tbl.attrs["skipped"] = []
    tbl.attrs["summary"] = "median"
    return tbl, m_cls


def _annotate(res, m_cls, conf=60.0):
    res = res.copy()
    res.attrs.update(res.attrs)
    cls = np.array(["Sulfatide"] * m_cls + ["Other"] * (len(res) - m_cls), dtype=object)
    # only the class ions and a few background ions carry an ID at all
    cls[m_cls + 5:] = None
    res["best_class"] = cls
    res["id_confidence"] = np.r_[np.full(m_cls, conf), np.full(len(res) - m_cls, 45.0)]
    return res


def _run(rng_seed=2026, **kw):
    rng = np.random.default_rng(rng_seed)
    tbl, m = _cohort_with_a_class(rng, **kw)
    res = cohort.nested_comparison(tbl, "normal", "trt", subject_by="subject",
                                   compartments=COMPARTMENTS)
    return tbl, _annotate(res, m), m


# --------------------------------------------------------------------------- #
def test_class_test_finds_the_planted_class_and_reports_its_correlation():
    tbl, res, m = _run(rho=0.25, endo_shift=-1.2)
    out = cohort.nested_class_comparison(tbl, res, contrast="endo")
    row = out[out["class"] == "Sulfatide"].iloc[0]

    assert row["n_ions"] == m
    assert row["direction"] == "down"
    assert row["median_log2_fc"] < -0.9
    # the ions really are correlated, and the test says by how much
    assert row["r_bar"] > 0.2
    assert row["vif"] > 3
    assert row["n_effective"] < m / 3
    # the adjustment costs significance — that is the entire point
    assert row["p_value"] > row["p_uncorrected"]
    assert row["p_value"] <= 0.05                 # a real, large effect still survives it


def test_class_test_p_value_saturates_because_the_class_is_worth_a_couple_of_ions():
    """A rank test can only say 'every class ion is at the bottom'. Once it does, a bigger
    effect buys nothing — with n_effective ≈ 2 there is a floor on the achievable p, however
    enormous the depletion. The uncorrected p keeps shrinking and is therefore a lie."""
    small = cohort.nested_class_comparison(*_run(rho=0.35, endo_shift=-1.2)[:2], contrast="endo")
    large = cohort.nested_class_comparison(*_run(rho=0.35, endo_shift=-3.0)[:2], contrast="endo")
    s = small[small["class"] == "Sulfatide"].iloc[0]
    l = large[large["class"] == "Sulfatide"].iloc[0]
    assert l["median_log2_fc"] < s["median_log2_fc"] - 1.5      # a far bigger effect
    assert l["p_value"] == pytest.approx(s["p_value"], rel=0.15)  # ...same adjusted p
    assert l["n_effective"] < 3


def test_class_test_centering_isolates_class_specific_correlation():
    """Every ion of one section shares that section's overall scale. Left in, an entirely
    uncorrelated class reports r̄ ≈ 0.6 and every class is deflated by the same nuisance."""
    tbl, res, _m = _run(rho=0.0)                  # NO class-specific coupling planted
    on = cohort.nested_class_comparison(tbl, res, contrast="endo", center_observations=True)
    off = cohort.nested_class_comparison(tbl, res, contrast="endo", center_observations=False)
    r_on = on[on["class"] == "Sulfatide"].iloc[0]
    r_off = off[off["class"] == "Sulfatide"].iloc[0]
    assert abs(r_on["r_bar"]) < 0.05 and r_on["n_effective"] > 15
    assert r_off["r_bar"] > 0.4 and r_off["n_effective"] < 3
    assert on.attrs["center_observations"] is True and off.attrs["center_observations"] is False


def test_class_test_deflates_a_correlated_class_with_no_effect():
    """No planted shift: the class drifts as a block, and the uncorrected p is not to be
    believed. This is the failure mode the whole function exists to prevent."""
    tbl, res, _m = _run(rng_seed=7, endo_shift=0.0, rho=0.8)
    out = cohort.nested_class_comparison(tbl, res, contrast="endo")
    row = out[out["class"] == "Sulfatide"].iloc[0]
    assert row["r_bar"] > 0.4
    assert row["p_value"] > row["p_uncorrected"]
    assert row["p_value"] > 0.05                  # correctly NOT significant
    assert row["q_value"] > 0.05


def test_class_test_is_immune_to_a_uniform_per_compartment_scale_offset():
    """A competitive test compares the class to the background, so a shift applied to EVERY
    ion cancels. A self-contained test would report it as a finding."""
    rng = np.random.default_rng(4)
    tbl, m = _cohort_with_a_class(rng, endo_shift=0.0)
    feat = [c for c in tbl.columns if c != "group"]
    endo_trt = np.array([r.region == "endo" and r.group == "trt"
                          for r in tbl.attrs["refs"]])
    tbl.loc[endo_trt, feat] = tbl.loc[endo_trt, feat] * 2 ** 0.6   # +0.6 log2, ALL ions
    res = _annotate(cohort.nested_comparison(tbl, "normal", "trt", subject_by="subject",
                                             compartments=COMPARTMENTS), m)
    out = cohort.nested_class_comparison(tbl, res, contrast="endo")
    row = out[out["class"] == "Sulfatide"].iloc[0]
    assert row["p_value"] > 0.05                  # the offset does not become a class hit
    # ...even though every per-ion effect is shifted positive
    assert np.nanmedian(res["effect__endo"]) > 0.4


def test_class_test_confidence_filter_drops_low_confidence_ions():
    tbl, res, m = _run()
    res.loc[res.index[:8], "id_confidence"] = 20.0     # 8 of the 19 are dubious IDs
    out = cohort.nested_class_comparison(tbl, res, contrast="endo", min_confidence=40)
    row = out[out["class"] == "Sulfatide"].iloc[0]
    assert row["n_ions"] == m - 8
    full = cohort.nested_class_comparison(tbl, res, contrast="endo", min_confidence=0)
    assert full[full["class"] == "Sulfatide"].iloc[0]["n_ions"] == m
    assert out.attrs["min_confidence"] == 40.0


def test_class_test_honours_min_ions_and_a_fixed_correlation():
    tbl, res, m = _run()
    out = cohort.nested_class_comparison(tbl, res, contrast="endo", min_ions=6)
    assert "Other" not in set(out["class"])            # the 5-ion background class is dropped
    fixed = cohort.nested_class_comparison(tbl, res, contrast="endo", inter_ion_cor=0.01)
    row = fixed[fixed["class"] == "Sulfatide"].iloc[0]
    assert row["r_bar"] == pytest.approx(0.01)
    assert row["vif"] == pytest.approx(1 + (m - 1) * 0.01)


def test_class_test_group_contrast_and_attrs():
    tbl, res, _m = _run()
    out = cohort.nested_class_comparison(tbl, res, contrast="group")
    assert "median_log2_fc" not in out.columns         # no per-compartment fold change to show
    assert out.attrs["contrast"] == "group"
    assert out.attrs["statistic"] == "effect_group / se_group"
    assert out.attrs["a_label"] == "normal" and out.attrs["b_label"] == "trt"
    assert out.attrs["n_ions_background"] > out.attrs["n_ions_annotated"]
    assert "Wu & Smyth" in out.attrs["test"]


def test_class_test_rejects_bad_inputs():
    tbl, res, _m = _run()
    with pytest.raises(ValueError, match="no contrast"):
        cohort.nested_class_comparison(tbl, res, contrast="nerve")
    with pytest.raises(ValueError, match="annotate the result first"):
        cohort.nested_class_comparison(tbl, res.drop(columns=["best_class"]), contrast="endo")
    with pytest.raises(ValueError, match="not row-aligned"):
        cohort.nested_class_comparison(tbl, res.sort_values("p_group"), contrast="endo")


def test_class_test_refuses_a_stratified_result():
    rng = np.random.default_rng(9)
    tbl, m = _cohort_with_a_class(rng)
    strat = _annotate(cohort.nested_comparison(tbl, "normal", "trt", subject_by="subject",
                                               model="stratified", compartments=COMPARTMENTS), m)
    with pytest.raises(ValueError, match="only the mixed model reports"):
        cohort.nested_class_comparison(tbl, strat, contrast="endo")
