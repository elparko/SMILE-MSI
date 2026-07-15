"""Correlation-adjusted competitive set testing — :func:`smile_msi.hierstats.camera`.

The point of the module is that ``m`` correlated features are worth far fewer than ``m``
independent ones. These tests pin that: the rank variance must reduce to the classical
Wilcoxon formula at ``r̄ = 0``, ``r̄`` must be recovered from planted correlation, and a set of
correlated null features must NOT be called significant the way an uncorrected test calls it.
"""
import numpy as np
import pytest

from smile_msi import hierstats as hs


# --------------------------------------------------------------------------- #
# inter-feature correlation
# --------------------------------------------------------------------------- #
def _correlated_block(rng, n_obs, m, rho):
    """m features with an exchangeable correlation of `rho` (one shared factor + noise)."""
    shared = rng.normal(size=(n_obs, 1))
    indep = rng.normal(size=(n_obs, m))
    return np.sqrt(rho) * shared + np.sqrt(1 - rho) * indep


def test_inter_feature_correlation_recovers_a_planted_rho():
    rng = np.random.default_rng(11)
    for rho in (0.0, 0.2, 0.5, 0.8):
        X = _correlated_block(rng, 400, 30, rho)
        # residualize against an intercept only, then unit-normalize
        R, ok = hs.standardized_residuals(X, np.ones((400, 1)))
        assert ok.all()
        r = hs.inter_feature_correlation(R, np.arange(30))
        assert r == pytest.approx(rho, abs=0.06)


def test_inter_feature_correlation_matches_the_explicit_mean_of_the_corr_matrix():
    rng = np.random.default_rng(2)
    X = _correlated_block(rng, 60, 8, 0.4)
    R, _ = hs.standardized_residuals(X, np.ones((60, 1)))
    fast = hs.inter_feature_correlation(R, np.arange(8))
    C = np.corrcoef(R.T)
    slow = (C.sum() - 8) / (8 * 7)
    assert fast == pytest.approx(slow, abs=1e-9)


def test_inter_feature_correlation_is_zero_for_a_singleton():
    R = np.eye(4)[:, :1]
    assert hs.inter_feature_correlation(R, [0]) == 0.0


def test_standardized_residuals_removes_the_design_and_flags_degenerate_columns():
    rng = np.random.default_rng(5)
    D = np.column_stack([np.ones(20), np.repeat([0, 1], 10)])
    Y = rng.normal(size=(20, 3))
    Y[:, 2] = D @ [1.0, 2.0]                       # perfectly explained → no residual
    R, ok = hs.standardized_residuals(Y, D)
    assert ok.tolist() == [True, True, False]
    assert np.allclose(D.T @ R[:, :2], 0, atol=1e-10)      # residuals orthogonal to the design
    assert np.allclose(np.linalg.norm(R[:, :2], axis=0), 1.0)
    assert np.allclose(R[:, 2], 0.0)


# --------------------------------------------------------------------------- #
# rank-sum variance
# --------------------------------------------------------------------------- #
def test_rank_sum_variance_reduces_to_the_classical_formula_at_zero_correlation():
    """The arcsin expression must collapse to n1*n2*(n+1)/12 — otherwise every uncorrelated
    set is tested against the wrong null."""
    rng = np.random.default_rng(7)
    stat = rng.normal(size=40)
    in_set = np.zeros(40, bool); in_set[:9] = True
    from scipy.stats import mannwhitneyu

    p_less, p_greater = hs.rank_sum_test_with_correlation(stat, in_set, correlation=0.0)
    ref = mannwhitneyu(stat[in_set], stat[~in_set], alternative="two-sided",
                       method="asymptotic", use_continuity=True).pvalue
    assert min(1.0, 2 * min(p_less, p_greater)) == pytest.approx(ref, rel=2e-2)


def test_rank_sum_p_grows_monotonically_with_correlation():
    rng = np.random.default_rng(9)
    stat = np.r_[rng.normal(1.2, 1, 15), rng.normal(0, 1, 100)]
    in_set = np.zeros(115, bool); in_set[:15] = True
    ps = [min(1.0, 2 * min(*hs.rank_sum_test_with_correlation(stat, in_set, correlation=c)))
          for c in (0.0, 0.05, 0.2, 0.5)]
    assert ps == sorted(ps)                        # more correlation ⇒ less significance
    assert ps[0] < 0.01 and ps[-1] > ps[0] * 5


def test_rank_sum_handles_ties_and_degenerate_sets():
    stat = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    in_set = np.array([True, True, False, False, False, False])
    p_less, p_greater = hs.rank_sum_test_with_correlation(stat, in_set)
    assert 0.0 <= p_less <= 1.0 and 0.0 <= p_greater <= 1.0
    assert hs.rank_sum_test_with_correlation(stat, np.zeros(6, bool)) == (1.0, 1.0)
    assert hs.rank_sum_test_with_correlation(stat, np.ones(6, bool)) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def _lipidome(rng, n_obs=30, n_bg=200, m=19, rho=0.3, shift=0.0):
    """A background lipidome plus one correlated 'class' of m ions, optionally shifted."""
    bg = rng.normal(size=(n_obs, n_bg))
    cls = _correlated_block(rng, n_obs, m, rho)
    Y = np.column_stack([cls, bg])
    stat = Y.mean(axis=0) * np.sqrt(n_obs)         # a per-feature statistic
    stat[:m] += shift
    return Y, stat, {"class": np.arange(m)}


def test_camera_deflates_a_correlated_null_set_that_the_naive_test_calls_significant():
    """The headline behaviour. A correlated set drifts as a block, so its mean statistic is far
    from zero by chance — and an uncorrected competitive test believes it."""
    rng = np.random.default_rng(20260709)
    naive_hits = adj_hits = 0
    for _ in range(60):
        Y, stat, sets = _lipidome(rng, rho=0.5, shift=0.0)     # NO real effect
        R, _ = hs.standardized_residuals(Y, np.ones((Y.shape[0], 1)))
        out = hs.camera(stat, sets, resid_unit=R)
        naive_hits += int(out["p_uncorrected"].iloc[0] <= 0.05)
        adj_hits += int(out["p_value"].iloc[0] <= 0.05)
    # the uncorrected test's false-positive rate blows past its nominal 5%
    assert naive_hits >= 12          # ≈20%+ of null sets called significant
    assert adj_hits <= 4             # the adjusted one stays near nominal
    assert adj_hits < naive_hits / 3


def test_camera_still_finds_a_real_shift_in_a_correlated_set():
    rng = np.random.default_rng(3)
    Y, stat, sets = _lipidome(rng, rho=0.3, shift=-2.5)
    R, _ = hs.standardized_residuals(Y, np.ones((Y.shape[0], 1)))
    out = hs.camera(stat, sets, resid_unit=R)
    row = out.iloc[0]
    assert row["direction"] == "down"
    assert row["p_value"] <= 0.05
    assert row["p_uncorrected"] < row["p_value"]     # adjustment costs power, as it must


def test_camera_reports_the_correlation_the_inflation_and_the_effective_count():
    rng = np.random.default_rng(4)
    Y, stat, sets = _lipidome(rng, m=19, rho=0.3)
    R, _ = hs.standardized_residuals(Y, np.ones((Y.shape[0], 1)))
    row = hs.camera(stat, sets, resid_unit=R).iloc[0]
    assert row["n_features"] == 19
    assert row["r_bar"] == pytest.approx(0.3, abs=0.12)
    assert row["vif"] == pytest.approx(1 + 18 * row["r_bar_used"])
    assert row["n_effective"] == pytest.approx(19 / row["vif"])
    assert row["n_effective"] < 8                    # 19 sulfatides are worth a handful


def test_camera_floors_the_inflation_at_no_adjustment_by_default():
    """A negative r̄ would make the test MORE significant than independence. A homologous
    series cannot plausibly be anti-correlated, so treat it as noise unless asked otherwise."""
    rng = np.random.default_rng(6)
    Y = rng.normal(size=(30, 60))
    Y[:, 1] = -Y[:, 0]                              # force a negative pair inside the set
    stat = Y.mean(axis=0)
    sets = {"s": np.arange(4)}
    R, _ = hs.standardized_residuals(Y, np.ones((30, 1)))
    floored = hs.camera(stat, sets, resid_unit=R).iloc[0]
    assert floored["r_bar"] < 0
    assert floored["r_bar_used"] == pytest.approx(0.0)
    assert floored["vif"] == pytest.approx(1.0)

    free = hs.camera(stat, sets, resid_unit=R, min_vif=0.0).iloc[0]
    assert free["r_bar_used"] == pytest.approx(free["r_bar"])
    assert free["vif"] < 1.0                        # anti-conservative — opt-in only


def test_camera_fixed_correlation_overrides_the_estimate():
    rng = np.random.default_rng(8)
    Y, stat, sets = _lipidome(rng, rho=0.4)
    R, _ = hs.standardized_residuals(Y, np.ones((30, 1)))
    row = hs.camera(stat, sets, resid_unit=R, inter_feature_cor=0.01).iloc[0]
    assert row["r_bar"] == pytest.approx(0.01)      # limma's preset, not the data's 0.4
    assert row["vif"] == pytest.approx(1 + 18 * 0.01)


def test_camera_parametric_and_rank_agree_in_direction_and_both_deflate():
    rng = np.random.default_rng(12)
    Y, stat, sets = _lipidome(rng, rho=0.35, shift=-2.0)
    R, _ = hs.standardized_residuals(Y, np.ones((30, 1)))
    a = hs.camera(stat, sets, resid_unit=R, use_ranks=True).iloc[0]
    b = hs.camera(stat, sets, resid_unit=R, use_ranks=False).iloc[0]
    assert a["direction"] == b["direction"] == "down"
    for row in (a, b):
        assert row["p_value"] > row["p_uncorrected"]
    assert "rank-sum" in hs.camera(stat, sets, resid_unit=R).attrs["test"]
    assert "parametric" in hs.camera(stat, sets, resid_unit=R, use_ranks=False).attrs["test"]


def test_camera_skips_small_sets_and_bh_adjusts_across_the_rest():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(15)
    stat = rng.normal(size=120)
    sets = {"tiny": [0, 1], "a": np.arange(2, 12), "b": np.arange(12, 30),
            "c": np.arange(30, 45)}
    out = hs.camera(stat, sets, min_size=3)
    assert list(out["set"]) and "tiny" not in set(out["set"])
    expect = multipletests(out["p_value"], method="fdr_bh")[1]
    assert out["q_value"].to_numpy() == pytest.approx(expect, rel=1e-9)
    assert out.attrs["n_background"] == 120


def test_camera_without_residuals_assumes_independence():
    rng = np.random.default_rng(17)
    stat = rng.normal(size=80)
    out = hs.camera(stat, {"s": np.arange(10)})
    assert out["r_bar"].iloc[0] == 0.0
    assert out["vif"].iloc[0] == 1.0
    assert out["p_value"].iloc[0] == pytest.approx(out["p_uncorrected"].iloc[0])
