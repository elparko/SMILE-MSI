"""Nested/hierarchical cohort statistics — :mod:`smile_msi.hierstats`.

The design under test throughout is the one the module exists for: subjects split into two
unbalanced groups, each subject measured in several compartments.
"""
from math import comb

import numpy as np
import pytest

from smile_msi import hierstats as hs


# --------------------------------------------------------------------------- #
# Small-n feasibility
# --------------------------------------------------------------------------- #
def test_exact_rank_p_floor_matches_permutation_count():
    # The floor is 2 / (number of distinct label assignments), by construction.
    assert hs.exact_rank_p_floor(3, 6) == pytest.approx(2.0 / comb(9, 3))
    assert hs.exact_rank_p_floor(3, 6) == pytest.approx(2.0 / 84)
    assert hs.exact_rank_p_floor(3, 3) == pytest.approx(2.0 / 20)   # balanced 3v3 can't beat 0.1
    assert hs.exact_rank_p_floor(4, 4) == pytest.approx(2.0 / 70)
    assert hs.exact_rank_p_floor(1, 5) == pytest.approx(2.0 / 6)    # 1-vs-N is nearly vacuous
    assert hs.exact_rank_p_floor(1, 1) == 1.0                       # clamped: 2/2 would be 1.0
    assert hs.exact_rank_p_floor(0, 5) == 1.0


def test_exact_rank_p_floor_is_symmetric_and_clamped():
    assert hs.exact_rank_p_floor(3, 6) == hs.exact_rank_p_floor(6, 3)
    assert 0.0 < hs.exact_rank_p_floor(2, 2) <= 1.0


def test_bh_feasibility_rejects_3v6_over_many_features():
    """The headline guard: 3-vs-6 clears an uncorrected 0.05 but cannot survive BH."""
    fea = hs.bh_feasibility(3, 6, n_features=500)
    assert fea["p_floor"] == pytest.approx(0.0238, abs=1e-3)
    assert fea["p_floor"] < 0.05                     # passes a naive "is p<0.05 reachable" check
    assert fea["max_features"] == 2                  # floor(0.05 / 0.0238)
    assert fea["feasible"] is False
    assert "cannot return p below" in fea["reason"]


def test_bh_feasibility_accepts_a_design_that_can_reject():
    fea = hs.bh_feasibility(8, 8, n_features=10)
    assert fea["feasible"] is True
    assert "reason" not in fea
    # And a single-feature 3v6 contrast IS decidable — the problem is multiplicity, not n alone.
    assert hs.bh_feasibility(3, 6, n_features=2)["feasible"] is True


# --------------------------------------------------------------------------- #
# Empirical-Bayes variance shrinkage
# --------------------------------------------------------------------------- #
def test_trigamma_inverse_round_trips():
    from scipy.special import polygamma

    y = np.array([0.05, 0.5, 1.0, 3.0, 25.0, 400.0])
    assert hs._trigamma_inverse(polygamma(1, y)) == pytest.approx(y, rel=1e-6)


def test_fit_f_dist_recovers_planted_prior():
    """Simulate the hierarchical variance model limma assumes and recover (d0, s0²)."""
    rng = np.random.default_rng(20260709)
    G, d, d0, s2_prior = 6000, 7, 4.0, 0.35
    sigma2 = s2_prior * d0 / rng.chisquare(d0, G)          # scaled inverse chi-square
    s2 = sigma2 * rng.chisquare(d, G) / d                  # observed residual variances
    df_prior, s2_hat = hs.fit_f_dist(s2, np.full(G, float(d)))
    assert df_prior == pytest.approx(d0, rel=0.20)
    assert s2_hat == pytest.approx(s2_prior, rel=0.15)


def test_squeeze_var_shrinks_toward_the_prior():
    rng = np.random.default_rng(7)
    d = 6.0
    s2 = 0.4 * rng.chisquare(d, 2000) / d
    s2_post, df_prior, s2_prior = hs.squeeze_var(s2, np.full(2000, d))
    assert np.isfinite(df_prior) and df_prior > 0
    # every posterior sits strictly between its own variance and the prior
    lo, hi = np.minimum(s2, s2_prior), np.maximum(s2, s2_prior)
    assert np.all(s2_post >= lo - 1e-12) and np.all(s2_post <= hi + 1e-12)
    # and the spread is compressed
    assert s2_post.std() < s2.std()


def test_squeeze_var_with_no_excess_spread_collapses_to_a_common_variance():
    """Identical residual variances ⇒ the data support one variance ⇒ infinite prior weight.

    Guards the limma ``fitFDist`` detail that the ``evar ≤ 0`` branch takes the arithmetic
    mean of s2, not ``exp(emean)``: the latter would return ≈0.29 here, not 0.25.
    """
    s2 = np.full(50, 0.25)
    s2_post, df_prior, s2_prior = hs.squeeze_var(s2, np.full(50, 7.0))
    assert not np.isfinite(df_prior)
    assert s2_prior == pytest.approx(0.25)
    assert s2_post == pytest.approx(np.full(50, 0.25))


def test_squeeze_var_ignores_zero_and_nonfinite_variances_when_fitting():
    s2 = np.array([0.0, np.nan, np.inf, 0.3, 0.5, 0.2, 0.44])
    df = np.full(s2.size, 5.0)
    s2_post, df_prior, s2_prior = hs.squeeze_var(s2, df)
    assert np.isfinite(s2_prior) and s2_prior > 0
    # the degenerate features still receive the prior rather than NaN
    assert s2_post[0] == pytest.approx(s2_prior)
    assert np.all(np.isfinite(s2_post))


# --------------------------------------------------------------------------- #
# Moderated t
# --------------------------------------------------------------------------- #
def _two_group(rng, n_a, n_b, n_feat, shift_idx=(), shift=2.0, sd=0.5):
    XA = rng.normal(0.0, sd, (n_a, n_feat))
    XB = rng.normal(0.0, sd, (n_b, n_feat))
    for j in shift_idx:
        XB[:, j] += shift
    return XA, XB


def test_moderated_t_finds_the_planted_features_at_3v6():
    rng = np.random.default_rng(11)
    XA, XB = _two_group(rng, 3, 6, 300, shift_idx=(0, 1, 2), shift=2.5, sd=0.4)
    res = hs.moderated_t_two_group(XA, XB)
    assert res.attrs["n_testable"] == 300
    hits = set(np.where(res["q_value"].to_numpy() <= 0.05)[0])
    assert {0, 1, 2} <= hits
    # and the exact rank test, on the identical data, cannot find any of them
    fea = hs.bh_feasibility(3, 6, 300)
    assert not fea["feasible"]


def test_moderated_t_effect_is_the_difference_of_means():
    rng = np.random.default_rng(3)
    XA, XB = _two_group(rng, 4, 4, 20)
    res = hs.moderated_t_two_group(XA, XB)
    assert res["effect"].to_numpy() == pytest.approx(XB.mean(0) - XA.mean(0))
    assert np.all(res["n_A"] == 4) and np.all(res["n_B"] == 4)


def test_moderated_t_gains_degrees_of_freedom_over_student():
    """df_total = residual df + df_prior — the power the prior buys, made explicit."""
    rng = np.random.default_rng(5)
    XA, XB = _two_group(rng, 3, 6, 400)
    res = hs.moderated_t_two_group(XA, XB)
    d0 = res.attrs["df_prior"]
    assert d0 > 0
    residual_df = 3 + 6 - 2
    assert np.all(res["df_total"].to_numpy() > residual_df)
    if np.isfinite(d0):
        assert res["df_total"].to_numpy() == pytest.approx(residual_df + d0)


def test_moderated_t_matches_student_t_when_the_prior_is_the_data():
    """With a common variance the posterior is that variance, and the moderated statistic is
    the pooled-variance t referred to the normal — check the t statistic against it directly."""
    XA = np.array([[0.0], [1.0], [2.0]])
    XB = np.array([[3.0], [4.0], [5.0]])
    # single feature ⇒ fit_f_dist takes the n==1 branch ⇒ df_prior = inf, s2_prior = s2
    res = hs.moderated_t_two_group(XA, XB)
    s2 = 1.0                                          # both groups have var 1.0 (ddof=1)
    expect_t = (4.0 - 1.0) / (np.sqrt(s2) * np.sqrt(1 / 3 + 1 / 3))
    assert res["t"].iloc[0] == pytest.approx(expect_t)
    assert not np.isfinite(res.attrs["df_prior"])


def test_moderated_t_marks_sparse_features_untested_and_keeps_them_out_of_bh():
    rng = np.random.default_rng(9)
    XA, XB = _two_group(rng, 3, 6, 10, shift_idx=(0,), shift=3.0, sd=0.3)
    XA[:, 5] = np.nan                                 # feature 5 absent from every A subject
    XA[1:, 6] = np.nan                                # feature 6 has only 1 A subject
    res = hs.moderated_t_two_group(XA, XB, min_n=2)
    assert res.attrs["n_testable"] == 8
    assert not res["tested"].iloc[5] and not res["tested"].iloc[6]
    assert np.isnan(res["p_value"].iloc[5]) and np.isnan(res["q_value"].iloc[6])
    # BH ran over 8 features, not 10: the smallest q equals the smallest p times 8.
    tested = res[res["tested"]].sort_values("p_value")
    assert tested["q_value"].iloc[0] == pytest.approx(tested["p_value"].iloc[0] * 8)


def test_moderated_t_ids_and_column_contract():
    XA, XB = _two_group(np.random.default_rng(1), 3, 3, 4)
    res = hs.moderated_t_two_group(XA, XB, ids=[700.5, 701.5, 702.5, 703.5])
    assert list(res["mz"]) == [700.5, 701.5, 702.5, 703.5]
    for c in ("n_A", "n_B", "mean_A", "mean_B", "effect", "s2_raw", "s2_post", "t",
              "df_total", "p_value", "q_value", "tested"):
        assert c in res.columns
    with pytest.raises(ValueError, match="ids has"):
        hs.moderated_t_two_group(XA, XB, ids=[1.0, 2.0])
    with pytest.raises(ValueError, match="same number of feature columns"):
        hs.moderated_t_two_group(XA, XB[:, :2])


# --------------------------------------------------------------------------- #
# Nested mixed model
# --------------------------------------------------------------------------- #
COMPARTMENTS = ["endoneurium", "perineurium", "epineurium"]


INTERACTION_SHIFT = 2.5      # feature 0: treated − normal, epineurium only (log2 units)
MAIN_SHIFT = 1.8             # feature 1: treated − normal, every compartment


def _nested_cohort(rng, n_a=3, n_b=6, n_feat=8, subject_sd=0.5):
    """``n_a + n_b`` subjects × 3 compartments, on the log2 modelling scale.

    Feature 0 carries a group×compartment interaction (the group effect exists only in the
    epineurium); feature 1 carries a uniform group main effect; the rest are null. Each
    subject gets a random intercept shared across features — the residual slide effect that
    normalization never fully removes, and the thing the ``(1 | subject)`` term exists to
    absorb. Each *feature* gets its own residual variance (drawn from a chi-square, as the
    empirical-Bayes prior assumes) so the shrinkage has real spread to work on, and its own
    baseline abundance spanning ~3 orders of magnitude, so the τ-regularized fold change
    isn't crushed by the pseudocount the way it would be with one flat baseline.
    """
    subjects = [f"nerve{i:02d}" for i in range(n_a + n_b)]
    groups = ["normal"] * n_a + ["treated"] * n_b
    comp_effect = {"endoneurium": 0.0, "perineurium": 0.8, "epineurium": -0.5}
    base = rng.uniform(4.0, 14.0, n_feat)             # log2 baseline: raw spans 16 … 16384
    base[0] = base[1] = 12.0                          # the planted features sit well above τ
    sd = np.sqrt(0.09 * rng.chisquare(6, n_feat) / 6)  # per-feature residual sd, ~0.3
    offs = {s: rng.normal(0.0, subject_sd) for s in subjects}
    rows, g_lab, c_lab, s_lab = [], [], [], []
    for s, g in zip(subjects, groups):
        for c in COMPARTMENTS:
            y = base + offs[s] + comp_effect[c] + rng.normal(0.0, 1.0, n_feat) * sd
            if g == "treated" and c == "epineurium":
                y[0] += INTERACTION_SHIFT             # interaction: epineurium only
            if g == "treated":
                y[1] += MAIN_SHIFT                    # main effect: every compartment
            rows.append(y)
            g_lab.append(g); c_lab.append(c); s_lab.append(s)
    return (np.vstack(rows), np.array(g_lab), np.array(c_lab), np.array(s_lab))


def test_bw_dfs_match_the_classical_repeated_measures_values():
    # 27 observations, 9 subjects, 6 fixed effects (intercept + group + 2 comp + 2 interaction)
    df_between, df_within = hs._bw_dfs(27, 9, 6)
    assert (df_between, df_within) == (7.0, 14.0)


def test_nested_design_columns_are_ordered_as_documented():
    g = np.array([0.0, 0.0, 1.0, 1.0])
    c = np.array([0, 1, 0, 1])
    D = hs._nested_design(g, c, n_comp=2)
    assert D.shape == (4, 4)                       # intercept, group, comp[1], group:comp[1]
    assert D[:, 0] == pytest.approx([1, 1, 1, 1])
    assert D[:, 1] == pytest.approx([0, 0, 1, 1])
    assert D[:, 2] == pytest.approx([0, 1, 0, 1])
    assert D[:, 3] == pytest.approx([0, 0, 0, 1])  # interaction fires only for group=1, comp=1


def test_nested_mixed_model_recovers_a_planted_interaction():
    X, g, c, s = _nested_cohort(np.random.default_rng(2026))
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated", compartments=COMPARTMENTS)
    assert res.attrs["n_subjects"] == 9
    assert res.attrs["n_observations"] == 27
    assert res.attrs["df_between"] == 7.0 and res.attrs["df_within"] == 14.0
    assert bool(res["converged"].all())

    # feature 0: interaction present, and the simple effect lives in the epineurium alone
    assert res["q_interaction"].iloc[0] <= 0.05
    assert res["p__epineurium"].iloc[0] < 0.05
    assert res["p__endoneurium"].iloc[0] > 0.05
    assert res["effect__epineurium"].iloc[0] == pytest.approx(INTERACTION_SHIFT, abs=0.6)
    assert res["effect__endoneurium"].iloc[0] == pytest.approx(0.0, abs=0.6)

    # feature 1: uniform main effect, no interaction
    assert res["p_group"].iloc[1] < 0.05
    assert res["p_interaction"].iloc[1] > 0.05
    assert res["effect_group"].iloc[1] == pytest.approx(MAIN_SHIFT, abs=0.6)

    # null features: neither
    for j in range(2, X.shape[1]):
        assert res["q_interaction"].iloc[j] > 0.05
        assert res["q_group"].iloc[j] > 0.05


def test_nested_mixed_model_group_effect_is_the_mean_simple_effect():
    X, g, c, s = _nested_cohort(np.random.default_rng(4))
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated", compartments=COMPARTMENTS)
    simple = np.vstack([res[f"effect__{c}"].to_numpy() for c in COMPARTMENTS])
    assert res["effect_group"].to_numpy() == pytest.approx(simple.mean(axis=0), abs=1e-8)


def test_nested_mixed_model_bh_families_are_separate():
    """Each q column is BH over its OWN p column — checked against statsmodels' multipletests,
    an implementation we didn't write, so a shared bug can't hide the error."""
    from statsmodels.stats.multitest import multipletests

    X, g, c, s = _nested_cohort(np.random.default_rng(6), n_feat=40)
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated", compartments=COMPARTMENTS)
    for pcol, qcol in [("p_interaction", "q_interaction"), ("p_group", "q_group"),
                       ("p__endoneurium", "q__endoneurium"), ("p__epineurium", "q__epineurium")]:
        p, q = res[pcol].to_numpy(), res[qcol].to_numpy()
        ok = np.isfinite(p)
        assert ok.sum() == 40
        expect = multipletests(p[ok], method="fdr_bh")[1]
        assert q[ok] == pytest.approx(expect, rel=1e-9)
        assert np.all(q[ok] >= p[ok] - 1e-12)
    # The interaction family is not the group family: feature 1 is a main effect only, so it
    # tops the group ranking and is unremarkable in the interaction ranking. (Its q_group is
    # NOT ≤0.05 here — a between-subject effect at 3-vs-6 rarely clears BH over 40 features,
    # unlike the within-subject interaction. That asymmetry is the design's, not the model's.)
    assert res["p_group"].iloc[1] == res["p_group"].min()
    assert res["p_interaction"].iloc[1] > 0.05
    assert res["q_group"].iloc[1] < res["q_interaction"].iloc[1]


def test_nested_mixed_model_reports_log2_fc_from_the_raw_scale():
    X, g, c, s = _nested_cohort(np.random.default_rng(8), n_feat=60)
    raw = np.exp2(X)                               # X is the log2 modelling scale
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated",
                                compartments=COMPARTMENTS, raw=raw)
    # log2_fc is the τ-regularized ratio of raw means, NOT the model coefficient: the mean of
    # exponentials exceeds the exponential of the mean, so it won't equal effect__ exactly.
    # It must still be large and positive where the effect is, and near zero where it isn't.
    assert res["log2_fc__epineurium"].iloc[0] > 1.8
    assert abs(res["log2_fc__endoneurium"].iloc[0]) < 0.8
    assert res["log2_fc__epineurium"].iloc[0] > res["log2_fc__endoneurium"].iloc[0] + 1.5
    # feature 1's uniform shift shows up in every compartment
    for comp in COMPARTMENTS:
        assert res[f"log2_fc__{comp}"].iloc[1] > 1.2


def test_nested_mixed_model_refuses_an_unestimable_design():
    X, g, c, s = _nested_cohort(np.random.default_rng(10), n_a=1, n_b=6)
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated", compartments=COMPARTMENTS)
    assert not res["converged"].any()               # 1 subject in group A cannot be fitted
    assert res.attrs["n_testable"] == 0
    assert "subjects per group" in res.attrs["warning"]
    assert res["p_interaction"].isna().all()


def test_nested_mixed_model_drops_a_missing_compartment_cleanly():
    X, g, c, s = _nested_cohort(np.random.default_rng(12))
    X[c == "epineurium", 3] = np.nan               # feature 3 never seen in the epineurium
    res = hs.nested_mixed_model(X, g, c, s, "normal", "treated", compartments=COMPARTMENTS)
    assert not res["converged"].iloc[3]
    assert np.isnan(res["p_interaction"].iloc[3])
    assert res["converged"].iloc[0]                 # its neighbours are unaffected
    assert res.attrs["n_testable"] == X.shape[1] - 1


def test_nested_mixed_model_z_df_is_anticonservative_versus_between_within():
    """The reason `bw` is the default: the normal reference always gives a smaller p."""
    X, g, c, s = _nested_cohort(np.random.default_rng(14))
    bw = hs.nested_mixed_model(X, g, c, s, "normal", "treated",
                               compartments=COMPARTMENTS, df_method="bw")
    z = hs.nested_mixed_model(X, g, c, s, "normal", "treated",
                              compartments=COMPARTMENTS, df_method="z")
    pb, pz = bw["p__epineurium"].to_numpy(), z["p__epineurium"].to_numpy()
    assert np.all(pz <= pb + 1e-12)
    assert np.any(pz < pb)


# --------------------------------------------------------------------------- #
# Stratified comparison
# --------------------------------------------------------------------------- #
def test_stratified_moderated_t_finds_the_compartment_specific_effect():
    X, g, c, s = _nested_cohort(np.random.default_rng(16), n_feat=60)
    res = hs.stratified_comparison(X, g, c, s, "normal", "treated",
                                   compartments=COMPARTMENTS, method="modt")
    assert res["q__epineurium"].iloc[0] <= 0.05     # interaction feature: hit in epineurium
    assert res["q__endoneurium"].iloc[0] > 0.05     # ...and nowhere else
    assert res["q__endoneurium"].iloc[1] <= 0.05    # main-effect feature: hit everywhere
    assert res["q__epineurium"].iloc[1] <= 0.05
    for comp in COMPARTMENTS:
        assert res.attrs["per_compartment"][comp]["n_A"] == 3
        assert res.attrs["per_compartment"][comp]["n_B"] == 6


def test_stratified_rank_test_warns_that_it_cannot_reject():
    X, g, c, s = _nested_cohort(np.random.default_rng(18), n_feat=60)
    res = hs.stratified_comparison(X, g, c, s, "normal", "treated",
                                   compartments=COMPARTMENTS, method="mwu")
    assert "warning" in res.attrs
    assert "cannot return p below" in res.attrs["warning"]
    # and, as advertised, nothing clears FDR even for the strongly planted feature
    assert not (res["q__epineurium"].to_numpy() <= 0.05).any()
    assert res["q__epineurium"].iloc[0] > 0.05


def test_stratified_comparison_records_per_compartment_feasibility():
    X, g, c, s = _nested_cohort(np.random.default_rng(19), n_feat=60)
    res = hs.stratified_comparison(X, g, c, s, "normal", "treated",
                                   compartments=COMPARTMENTS, method="modt")
    fea = res.attrs["per_compartment"]["endoneurium"]["feasibility"]
    assert fea["n_a"] == 3 and fea["n_b"] == 6 and fea["feasible"] is False
    assert res.attrs["n_subjects"] == 9
